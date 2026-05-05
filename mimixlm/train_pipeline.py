"""
train_pipeline.py
--------
General-purpose training script built on transformer_minimal.py.

Supports two task modes:
  pretrain : standard language-model pre-training (next-token prediction on all tokens)
  sft      : supervised fine-tuning on instruction/chat data (loss only on assistant turns)

Data source modes (auto-detected from path):
  1. Single .txt file          — full file content as one text  (pretrain only)
  2. Directory of .txt files   — each file as one text, all merged  (pretrain only)
  3. Single .jsonl file        — text / conversation records
  4. Directory of .jsonl files — all files merged

JSONL formats supported:
  pretrain : {"text": "..."} or {"content": "..."}
  sft      : alpaca  — {"instruction": ..., "input": ..., "output": ...}
             sharegpt — {"conversations": [{"from": "human"/"gpt", "value": ...}, ...]}

Usage:
    # Pre-training
    python train.py --task pretrain --data data/corpus/ --output model/my_model

    # SFT (alpaca format)
    python train.py --task sft --chat-format alpaca \\
        --data data/alpaca.jsonl --output model/sft_model \\
        --resume model/pretrain_model

    # SFT (sharegpt format)
    python train.py --task sft --chat-format sharegpt \\
        --data data/sharegpt.jsonl --output model/sft_model
"""

import argparse
import dataclasses
import json
import math
import os
import random
import shutil
import time
import urllib.request
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset
from typing import Literal, Optional

from transformer_minimal import (
    ModelConfig,
    Transformer,
)
from tokenizer_minimal import Tokenizer, BPETokenizer
from chat_cli import ChatFormatter
import logger as _logger_module

# ── Logger setup ─────────────────────────────────────────────────────────────
# Reuse the singleton logger instance from logger.py to avoid duplicate handlers.
# File handler is attached lazily in main() once the task_name is known.
logger = _logger_module.logger

def _init_file_logger(task_name: "str | None" = None) -> None:
    """Attach a file handler to the module logger.

    Args:
        task_name : used as the log file stem.  When None or empty, a
                    timestamp (``YYYYMMDD_HHMMSS``) is used instead.
    """
    if not task_name:
        task_name = time.strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(_logger_module.LOG_DIR, f"{task_name}.log")
    _logger_module.add_file_handlers(log_path)

# ─────────────────────────────────────────────────────────────────────────────
# TrainConfig
# ─────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class TrainConfig:
    """
    Hyperparameters that control the training loop.

    Separating training config from ModelConfig keeps the model definition
    clean and makes it easy to swap optimiser settings without touching the
    architecture.

    Args:
        batch_size                : number of samples per gradient update step
        lr                        : peak learning rate (reached at the end of warmup)
        warmup_steps              : number of linear warm-up steps; 0 = start at peak lr
        max_grad_norm             : gradient clipping threshold (0 = no clipping)
        log_interval              : print a loss line every N steps; None = auto (total/100)
        num_epochs                : number of full passes over the dataset
        gradient_accumulation_steps : accumulate gradients over N micro-batches before
                                    each optimizer step; effective batch = batch_size * N
        mixed_precision           : floating-point dtype for autocast; one of
                                    "no" (disabled), "fp16", "bf16"
        gradient_checkpointing    : if True, enable activation checkpointing on every
                                    decoder (and encoder) layer to trade compute for memory
        use_ddp                   : if True, wrap the model with DistributedDataParallel.
                                    Requires the process group to be initialised before
                                    calling train() (e.g. via dist.init_process_group).
        save_every_steps          : save a checkpoint every N optimizer steps; 0 = only save at end
        max_checkpoints           : maximum number of recent checkpoints to keep; 0 = keep all
        eval_every_steps          : run eval on eval_dataloader every N optimizer steps; 0 = disabled
    """
    batch_size:                  int        = 64
    lr:                          float      = 1e-3
    warmup_steps:                int        = 400
    lr_scheduler:                str        = "rsqrt"   # "rsqrt" | "cosine"
    max_grad_norm:               float      = 1.0
    log_interval:                int | None = None
    num_epochs:                  int        = 1
    gradient_accumulation_steps: int        = 1
    mixed_precision:             str        = "no"   # "no" | "fp16" | "bf16"
    gradient_checkpointing:      bool       = False
    use_ddp:                     bool       = False
    save_every_steps:            int        = 0      # save a checkpoint every N optimizer steps; 0 = only save at end
    max_checkpoints:             int        = 3      # maximum number of recent checkpoints to keep; 0 = keep all
    eval_every_steps:            int        = 0      # run eval on eval_dataloader every N optimizer steps; 0 = disabled
    device:                      str        = "cpu"

# ─────────────────────────────────────────────────────────────────────────────
# Training utilities (moved from transformer_minimal.py)
# ─────────────────────────────────────────────────────────────────────────────


class TransformerDataset(Dataset):
    """
    Wraps a list of pre-built samples for use with DataLoader.
    Each sample is a dict with keys depending on model_type:
      encoder_decoder : {"src": LongTensor, "tgt_in": LongTensor, "tgt_out": LongTensor}
      lm              : {"tgt_in": LongTensor, "tgt_out": LongTensor}
    """

    def __init__(self, samples: list[dict]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]


def collate_fn(batch: list[dict], pad_id: int) -> dict:
    """Pad each field in the batch to the same length and stack into tensors.

    ``loss_mask`` fields are padded with 0 (ignore) rather than *pad_id* so
    that padded positions are automatically excluded from the SFT loss.
    """
    keys = batch[0].keys()
    result = {}
    for key in keys:
        seqs = [item[key] for item in batch]
        max_len = max(s.size(0) for s in seqs)
        fill_value = 0 if key == "loss_mask" else pad_id
        padded = torch.stack(
            [F.pad(s, (0, max_len - s.size(0)), value=fill_value) for s in seqs]
        )
        result[key] = padded
    return result


def make_dataloader(
    samples: list[dict],
    batch_size: int,
    pad_id: int,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Build a DataLoader from a list of sample dicts."""
    dataset = TransformerDataset(samples)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda batch: collate_fn(batch, pad_id),
        num_workers=num_workers,
        pin_memory=False,
    )


# ─────────────────────────────────────────────
# 8a. Training Callbacks
# ─────────────────────────────────────────────

class TrainingCallback:
    """
    Base class for training callbacks.

    Subclass and override :meth:`on_step` and/or :meth:`on_save` to inject
    custom logic at regular training steps or whenever a checkpoint is saved.

    Attributes:
        every_n_steps : call :meth:`on_step` every N optimizer steps.
                        0 = never call on_step (only on_save is used).
        on_save_only  : if True, skip on_step entirely and only respond to
                        on_save events (equivalent to every_n_steps=0).

    Example — log a custom metric every 100 steps::

        class MetricLogger(TrainingCallback):
            every_n_steps = 100
            def on_step(self, step: int, logs: dict) -> None:
                logger.info(f"step={step} loss={logs.get('loss', '?'):.4f}")
    """

    every_n_steps: int = 0  # 0 = on_save only

    def on_step(self, step: int, logs: dict) -> None:
        """
        Called after every optimizer step when ``step % every_n_steps == 0``.

        Args:
            step : current global optimizer step (1-based)
            logs : dict with at least ``{"loss": float, "lr": float}``
        """

    def on_save(self, checkpoint_dir: str, step: int) -> None:
        """
        Called immediately after a checkpoint directory has been written.

        Args:
            checkpoint_dir : absolute path to the newly created checkpoint dir
            step           : global optimizer step at which the save occurred
        """


class CopyTokenizerCallback(TrainingCallback):
    """
    Default callback: copies tokenizer files from *save_path* into each
    checkpoint sub-directory whenever a checkpoint is saved.

    Files copied (if they exist in *save_path*):
      - ``tokenizer.json``
      - ``bpe_ranks.json``   (BPETokenizer)
      - ``chat_template.json``

    This ensures every checkpoint is self-contained and can be loaded with
    ``Tokenizer.from_pretrained(checkpoint_dir)`` without referencing the
    root save directory.

    Args:
        save_path : root training output directory that contains the tokenizer
                    files (the same directory passed to :func:`train`).

    Usage::

        callback = CopyTokenizerCallback("checkpoints/my_run")
        train(..., save_path="checkpoints/my_run", callbacks=[callback])
    """

    TOKENIZER_FILES = ("tokenizer.json", "bpe_ranks.json", "chat_template.json")

    def __init__(self, save_path: str) -> None:
        self.save_path = save_path

    def on_save(self, checkpoint_dir: str, step: int) -> None:
        copied: list[str] = []
        for filename in self.TOKENIZER_FILES:
            source_path = os.path.join(self.save_path, filename)
            if os.path.isfile(source_path):
                shutil.copy2(source_path, os.path.join(checkpoint_dir, filename))
                copied.append(filename)
        if copied:
            logger.info(f"  CopyTokenizerCallback: copied {copied} -> {checkpoint_dir}")


def _save_checkpoint(
    model,
    cfg,
    train_cfg,
    save_dir: str,
    step: int,
    max_checkpoints: int,
    saved_paths: list,
    callbacks: "list[TrainingCallback] | None" = None,
) -> None:
    """
    Save a checkpoint in ``checkpoint-{step}`` sub-directory inside *save_dir*,
    then fire the ``on_save`` hook on every registered callback.

    Keeps at most *max_checkpoints* recent checkpoints by deleting the oldest
    one whenever the limit is exceeded.  Pass ``max_checkpoints=0`` to keep all.

    Tokenizer / chat-template files are no longer saved here directly; use
    :class:`CopyTokenizerCallback` (or a custom :class:`TrainingCallback`) to
    copy them into each checkpoint directory via the ``on_save`` hook.

    Args:
        model           : raw (non-DDP) model instance
        cfg             : ModelConfig used to build the model
        train_cfg       : TrainConfig used for training
        save_dir        : root directory; each checkpoint gets its own sub-dir
        step            : current global optimizer step (used in the folder name)
        max_checkpoints : maximum number of checkpoints to retain (0 = unlimited)
        saved_paths     : mutable list tracking checkpoint dirs in order;
                          updated in-place by this function
        callbacks       : list of :class:`TrainingCallback` instances whose
                          ``on_save`` method is called after the checkpoint is
                          written
    """
    checkpoint_dir = os.path.join(save_dir, f"checkpoint-{step}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save model weights
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, "model.pt"))

    # Save ModelConfig as JSON (fully self-contained for from_pretrained)
    with open(os.path.join(checkpoint_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(dataclasses.asdict(cfg), f, indent=2)

    saved_paths.append(checkpoint_dir)
    logger.info(f"  Checkpoint saved -> {checkpoint_dir}")

    # Fire on_save hooks
    for callback in (callbacks or []):
        callback.on_save(checkpoint_dir, step)

    # Evict oldest checkpoint if over the limit
    if max_checkpoints > 0 and len(saved_paths) > max_checkpoints:
        oldest = saved_paths.pop(0)
        if os.path.isdir(oldest):
            shutil.rmtree(oldest)
            logger.info(f"  Checkpoint evicted: {oldest}")


def train_lm(
    model: Transformer,
    train_cfg: TrainConfig,
    dataloader: DataLoader,
    save_path: Optional[str] = None,
    eval_dataloader: Optional[DataLoader] = None,
    callbacks: "list[TrainingCallback] | None" = None,
) -> None:
    """
        save_path       : if provided, save model checkpoint here after training
        eval_dataloader : optional DataLoader for evaluation; required when
                          ``train_cfg.eval_every_steps > 0``
        callbacks       : list of :class:`TrainingCallback` instances.
                          - ``on_step(step, logs)`` is called after every
                            optimizer step when ``step % callback.every_n_steps == 0``
                            (skipped when ``every_n_steps == 0``).
                          - ``on_save(checkpoint_dir, step)`` is called after
                            every checkpoint write (mid-training and final).
                          Pass :class:`CopyTokenizerCallback` here to copy
                          tokenizer files into each checkpoint directory.
    """
    cfg = model.cfg

    # ── Gradient checkpointing ───────────────────────────────────────────────
    # Wrap every encoder/decoder layer's forward with torch.utils.checkpoint so
    # activations are recomputed during backward instead of stored in memory.
    if train_cfg.gradient_checkpointing:
        def _make_checkpointed_forward(layer):
            original_forward = layer.forward
            def checkpointed_forward(*args, **kwargs):
                # checkpoint requires at least one tensor input with requires_grad.
                # use_reentrant=False avoids issues with kwargs in newer PyTorch.
                return grad_checkpoint(original_forward, *args, use_reentrant=False, **kwargs)
            return checkpointed_forward

        for layer in list(model.encoder_layers) + list(model.decoder_layers):
            layer.forward = _make_checkpointed_forward(layer)

    # Normalise device: accept plain strings like "cpu", "cuda:0", "mps"
    device = torch.device(train_cfg.device) if isinstance(train_cfg.device, str) else train_cfg.device

    # Cast model to the dtype specified in ModelConfig before moving to device
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    model_dtype = dtype_map.get(cfg.dtype, torch.float32)
    model.to(dtype=model_dtype)
    model.to(device)

    # ── DDP wrapping ─────────────────────────────────────────────────────────
    is_ddp       = train_cfg.use_ddp and dist.is_available() and dist.is_initialized()
    ddp_rank     = dist.get_rank() if is_ddp else 0
    is_main_rank = ddp_rank == 0

    if is_ddp:
        model = DDP(model, device_ids=[device] if device.type == "cuda" else None)

    # Unwrapped model reference for parameter counting and checkpointing
    raw_model = model.module if is_ddp else model

    model.train()

    # ── Mixed precision setup ────────────────────────────────────────────────
    # mixed_precision (AMP) only makes sense when weights are float32.
    # If cfg.dtype is already float16/bfloat16, the weights are already low-
    # precision and autocast would be a no-op; skip it and warn the user.
    if cfg.dtype != "float32" and train_cfg.mixed_precision != "no":
        logger.info(
            f"  [Warning] mixed_precision='{train_cfg.mixed_precision}' is ignored "
            f"because cfg.dtype='{cfg.dtype}' (model weights are already low-precision). "
            f"AMP is only meaningful when weights are float32."
        )
        use_amp = False
        scaler  = None
        amp_dtype = model_dtype
    else:
        amp_dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16}
        use_amp   = train_cfg.mixed_precision in amp_dtype_map
        amp_dtype = amp_dtype_map.get(train_cfg.mixed_precision, torch.float32)
        # GradScaler is only needed for fp16 (bf16 doesn't require loss scaling)
        scaler    = torch.amp.GradScaler(device.type) if train_cfg.mixed_precision == "fp16" else None

    # ── Scheduler & optimiser ────────────────────────────────────────────────
    num_epochs  = train_cfg.num_epochs
    accum_steps = max(train_cfg.gradient_accumulation_steps, 1)
    # Optimizer steps per epoch = ceil(batches / accum_steps)
    opt_steps_per_epoch = max(len(dataloader) // accum_steps, 1)
    total_opt_steps     = num_epochs * opt_steps_per_epoch
    warmup              = train_cfg.warmup_steps

    optimizer = torch.optim.Adam(
        raw_model.parameters(), lr=train_cfg.lr, betas=(0.9, 0.98), eps=1e-9
    )
    _warmup = max(warmup, 1)
    if train_cfg.lr_scheduler == "cosine":
        def _cosine_with_warmup(step: int) -> float:
            if step < _warmup:
                return (step + 1) / _warmup
            progress = (step - _warmup) / max(total_opt_steps - _warmup, 1)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_cosine_with_warmup)
    else:
        # Default: inverse-square-root (rsqrt) schedule
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: min(
                (step + 1) / _warmup,
                (_warmup / (step + 1)) ** 0.5,
            ),
        )

    # reduction="sum" + manual normalisation lets us detect all-PAD batches
    # and skip them instead of producing NaN loss.
    criterion = nn.CrossEntropyLoss(ignore_index=cfg.pad_idx, reduction="sum")

    # ── Logging ──────────────────────────────────────────────────────────────
    if is_main_rank:
        num_kv      = cfg.num_kv_heads if cfg.num_kv_heads > 0 else cfg.num_heads
        num_params  = sum(p.numel() for p in raw_model.parameters())
        num_trained = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
        logger.info(f"Training | epochs: {num_epochs} | batches/epoch: {len(dataloader)} | "
              f"accum_steps: {accum_steps} | opt_steps: {total_opt_steps} | device: {device}")
        logger.info(f"mode: {cfg.model_type} | pos={cfg.pos_encoding}(base={cfg.rope_base}) | "
              f"ffn={cfg.ffn_type}/{cfg.activation} | norm={cfg.norm_type}/{cfg.norm_layer}")
        logger.info(f"heads={cfg.num_heads} | kv_heads={num_kv} | "
              f"qk_norm={cfg.qk_norm} | v_norm={cfg.v_norm} | "
              f"learnable_residual={cfg.learnable_residual} | tie_weights={cfg.tie_weights}")
        logger.info(f"Parameters: {num_params:,} total | {num_trained:,} trainable | "
              f"lr={train_cfg.lr} | warmup={train_cfg.warmup_steps} | "
              f"max_grad_norm={train_cfg.max_grad_norm} | "
              f"amp={train_cfg.mixed_precision} | "
              f"grad_ckpt={train_cfg.gradient_checkpointing} | ddp={is_ddp}")

    log_interval = (
        train_cfg.log_interval
        if train_cfg.log_interval is not None
        else max(total_opt_steps // 100, 1)
    )

    global_opt_step       = 0   # counts optimizer (not micro-batch) steps
    micro_batch_idx       = 0   # counts every micro-batch across the epoch
    saved_checkpoint_paths: list[str] = []   # tracks checkpoint dirs for eviction

    train_start_time  = time.monotonic()
    last_log_time     = train_start_time   # timestamp of the last log print
    last_log_step     = 0                  # global_opt_step at the last log print

    for epoch in range(1, num_epochs + 1):
        epoch_loss       = 0.0
        accum_loss       = 0.0   # accumulated loss across micro-batches
        epoch_start_time = time.monotonic()

        for batch in dataloader:
            micro_batch_idx += 1
            is_last_accum   = (micro_batch_idx % accum_steps == 0)

            # Move batch tensors to device
            batch = {key: tensor.to(device) for key, tensor in batch.items()}

            # ── Forward pass (with optional autocast) ────────────────────────
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                if cfg.model_type == "lm":
                    logits = model(batch["tgt_in"])
                else:
                    logits = model(batch["src"], batch["tgt_in"])

                targets     = batch["tgt_out"].view(-1)

                # loss_mask: 1 = compute loss, 0 = ignore (used for SFT to
                # restrict loss to assistant turns only).  Falls back to the
                # standard pad-token mask when the field is absent.
                if "loss_mask" in batch:
                    token_mask = batch["loss_mask"].view(-1).bool()
                else:
                    token_mask = targets != cfg.pad_idx

                num_targets = token_mask.sum()

                if num_targets == 0:
                    # Skip all-PAD / all-masked batches
                    continue

                # Divide loss by accum_steps so the accumulated gradient equals
                # the mean gradient over the full effective batch.
                # Upcast logits to fp32 before cross-entropy: log_softmax inside
                # cross_entropy can overflow in fp16/bf16 on large vocabularies.
                flat_logits = logits.float().view(-1, cfg.vocab_size)[token_mask]
                flat_targets = targets[token_mask]
                loss = criterion(flat_logits, flat_targets) / num_targets
                scaled_loss = loss / accum_steps

            # ── Backward pass ────────────────────────────────────────────────
            if scaler is not None:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            accum_loss += loss.item()

            # ── Optimizer step (every accum_steps micro-batches) ─────────────
            if is_last_accum:
                if scaler is not None:
                    scaler.unscale_(optimizer)

                if train_cfg.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(
                        raw_model.parameters(), max_norm=train_cfg.max_grad_norm
                    )

                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad()

                avg_accum_loss  = accum_loss / accum_steps
                epoch_loss     += avg_accum_loss
                accum_loss      = 0.0
                global_opt_step += 1

                if is_main_rank and global_opt_step % log_interval == 0:
                    now            = time.monotonic()
                    elapsed_since_log = now - last_log_time
                    steps_since_log   = global_opt_step - last_log_step
                    steps_per_sec  = steps_since_log / max(elapsed_since_log, 1e-6)
                    steps_remaining = total_opt_steps - global_opt_step
                    eta_seconds    = steps_remaining / max(steps_per_sec, 1e-6)
                    eta_str        = (
                        f"{int(eta_seconds // 3600):02d}h"
                        f"{int(eta_seconds % 3600 // 60):02d}m"
                        f"{int(eta_seconds % 60):02d}s"
                    )
                    logger.info(
                        f"  Epoch {epoch} | OptStep {global_opt_step:>6}/{total_opt_steps} | "
                        f"Loss: {avg_accum_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f} | "
                        f"{steps_per_sec:.2f} steps/s | ETA: {eta_str}"
                    )
                    last_log_time = now
                    last_log_step = global_opt_step

                # ── on_step callbacks ────────────────────────────────────
                if is_main_rank:
                    step_logs = {"loss": avg_accum_loss, "lr": scheduler.get_last_lr()[0]}
                    for callback in (callbacks or []):
                        if callback.every_n_steps > 0 and global_opt_step % callback.every_n_steps == 0:
                            callback.on_step(global_opt_step, step_logs)

                # ── Mid-training checkpoint (every save_every_steps steps) ──
                if (
                    is_main_rank
                    and save_path is not None
                    and train_cfg.save_every_steps > 0
                    and global_opt_step % train_cfg.save_every_steps == 0
                ):
                    _save_checkpoint(
                        raw_model, cfg, train_cfg, save_path,
                        global_opt_step, train_cfg.max_checkpoints,
                        saved_checkpoint_paths,
                        callbacks=callbacks,
                    )

                # ── Periodic evaluation (every eval_every_steps steps) ───────
                if (
                    is_main_rank
                    and eval_dataloader is not None
                    and train_cfg.eval_every_steps > 0
                    and global_opt_step % train_cfg.eval_every_steps == 0
                ):
                    raw_model.eval()
                    eval_start_time = time.monotonic()
                    eval_loss_total = 0.0
                    eval_steps = 0
                    with torch.no_grad():
                        for eval_batch in eval_dataloader:
                            if cfg.model_type == "lm":
                                eval_src = eval_batch["tgt_in"].to(device)
                                eval_tgt = eval_batch["tgt_out"].to(device)
                                eval_logits = raw_model(eval_src)
                            else:
                                eval_src    = eval_batch["src"].to(device)
                                eval_tgt_in = eval_batch["tgt_in"].to(device)
                                eval_tgt    = eval_batch["tgt_out"].to(device)
                                eval_logits = raw_model(eval_src, eval_tgt_in)
                            eval_mask = eval_batch.get("loss_mask")
                            if eval_mask is not None:
                                eval_mask = eval_mask.to(device)
                                eval_loss = torch.nn.functional.cross_entropy(
                                    eval_logits.reshape(-1, cfg.vocab_size),
                                    eval_tgt.reshape(-1),
                                    reduction="none",
                                )
                                eval_loss = (eval_loss * eval_mask.reshape(-1)).sum() / eval_mask.sum().clamp(min=1)
                            else:
                                eval_loss = torch.nn.functional.cross_entropy(
                                    eval_logits.reshape(-1, cfg.vocab_size),
                                    eval_tgt.reshape(-1),
                                    ignore_index=cfg.pad_idx,
                                )
                            eval_loss_total += eval_loss.item()
                            eval_steps += 1
                    eval_elapsed = time.monotonic() - eval_start_time
                    avg_eval_loss = eval_loss_total / max(eval_steps, 1)
                    logger.info(
                        f"  [Eval] Epoch {epoch} | OptStep {global_opt_step:>6} | "
                        f"Eval Loss: {avg_eval_loss:.4f} | "
                        f"eval time: {eval_elapsed:.1f}s"
                    )
                    raw_model.train()
                    # Reset log timer so the next train-log interval is measured
                    # from after the eval pass (avoids inflated ETA after eval).
                    last_log_time = time.monotonic()
                    last_log_step = global_opt_step

        # Flush any remaining accumulated gradients at epoch end
        if micro_batch_idx % accum_steps != 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
            if train_cfg.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(
                    raw_model.parameters(), max_norm=train_cfg.max_grad_norm
                )
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_opt_step += 1

        if is_main_rank:
            epoch_elapsed = time.monotonic() - epoch_start_time
            avg_loss = epoch_loss / max(opt_steps_per_epoch, 1)
            logger.info(
                f"  Epoch {epoch} done | avg loss: {avg_loss:.4f} | "
                f"epoch time: {epoch_elapsed:.1f}s"
            )

    if is_main_rank:
        total_elapsed = time.monotonic() - train_start_time
        logger.info(
            f"Training complete. | total time: "
            f"{int(total_elapsed // 3600):02d}h"
            f"{int(total_elapsed % 3600 // 60):02d}m"
            f"{int(total_elapsed % 60):02d}s"
        )

    # ── Final checkpoint at end of training ─────────────────────────────────
    if save_path is not None and is_main_rank:
        _save_checkpoint(
            raw_model, cfg, train_cfg, save_path,
            global_opt_step, train_cfg.max_checkpoints,
            saved_checkpoint_paths,
            callbacks=callbacks,
        )



# ── Module-level defaults ────────────────────────────────────────────────────

DEFAULT_VOCAB_SIZE        = 8000
DEFAULT_CHUNK_SIZE        = 256
DEFAULT_STRIDE            = 128
DEFAULT_TEST_RATIO        = 0.05
DEFAULT_TOKENIZE_WORKERS  = 8
DEFAULT_DATA_URL          = "https://cs.stanford.edu/people/karpathy/char-rnn/shakespeare_input.txt"

# Keys tried in order when extracting text from a pretrain JSONL record
JSONL_TEXT_KEYS = ("text", "content")

# ─────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def _read_txt_file(path: str) -> list[str]:
    """Return the full content of a plain-text file as a single string."""
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return [fh.read()]


def _read_jsonl_file(path: str) -> list[dict | str]:
    """Return raw records from a JSONL file as a list of dicts (or strings)."""
    records: list = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line_no, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                records.append(json.loads(raw))
            except json.JSONDecodeError:
                logger.warning(f"  [Warning] Skipping malformed JSON at {path}:{line_no}")
    return records


def _fetch_url(url: str, cache_path: "str | None" = None) -> list[str]:
    """Download a URL and return its text content as a single-element list.

    If *cache_path* is given, the downloaded content is saved there and reused
    on subsequent calls instead of re-downloading.
    """
    if cache_path and os.path.isfile(cache_path):
        logger.info(f"      Using cached file: {cache_path}")
        with open(cache_path, "r", encoding="utf-8") as cached_file:
            return [cached_file.read()]
    logger.info(f"      Downloading: {url}")
    with urllib.request.urlopen(url) as response:
        text = response.read().decode("utf-8", errors="replace")
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as cache_file:
            cache_file.write(text)
        logger.info(f"      Cached to: {cache_path}")
    return [text]


def load_data(
    data_path: str = DEFAULT_DATA_URL,
    url_cache_path: "str | None" = None,
) -> list:
    """Load raw data from *data_path*.

    Supported sources (auto-detected):
      - HTTP/HTTPS URL           : downloaded, returned as a single-element list of str
      - Single .txt file         : full file content as one string
      - Directory of .txt files  : each file as one string, all merged
      - Single .jsonl file       : list of raw dicts
      - Directory of .jsonl files: all files merged into a list of raw dicts

    Args:
        data_path     : file path, directory path, or HTTP/HTTPS URL.
        url_cache_path: local path to cache the downloaded content when
                        *data_path* is a URL.

    Returns:
        List of strings (txt / URL) or dicts (jsonl).
    """
    logger.info(f"[1/5] Loading data from: {data_path}")

    if data_path.startswith("http://") or data_path.startswith("https://"):
        records = _fetch_url(data_path, cache_path=url_cache_path)
    elif os.path.isfile(data_path):
        records = (
            _read_jsonl_file(data_path)
            if data_path.endswith(".jsonl")
            else _read_txt_file(data_path)
        )
    elif os.path.isdir(data_path):
        entries     = [os.path.join(data_path, f) for f in sorted(os.listdir(data_path))]
        txt_files   = [p for p in entries if p.endswith(".txt")   and os.path.isfile(p)]
        jsonl_files = [p for p in entries if p.endswith(".jsonl") and os.path.isfile(p)]
        if not txt_files and not jsonl_files:
            raise ValueError(f"No .txt or .jsonl files found in: {data_path}")
        records = []
        for path in txt_files:
            records.extend(_read_txt_file(path))
        for path in jsonl_files:
            records.extend(_read_jsonl_file(path))
    else:
        raise FileNotFoundError(f"Data path not found: {data_path!r}")

    logger.info(f"      Loaded {len(records):,} records")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Pretrain sample construction
# ─────────────────────────────────────────────────────────────────────────────

def _extract_pretrain_texts(records: list) -> list[str]:
    """Extract plain text strings from raw records (str or dict)."""
    texts: list[str] = []
    for record in records:
        if isinstance(record, str):
            texts.append(record)
        elif isinstance(record, dict):
            for key in JSONL_TEXT_KEYS:
                if key in record and isinstance(record[key], str):
                    text = record[key].strip()
                    if text:
                        texts.append(text)
                    break
    return texts


def build_pretrain_samples(
    token_id_lists: list[list[int]],
    concat: bool = True,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    stride: int = DEFAULT_STRIDE,
) -> list[dict]:
    """Convert lists of token ids into pretrain sample dicts.

    Args:
        token_id_lists : list of encoded token id sequences
        concat         : if True, concatenate all sequences and slice into
                         fixed-length chunks (best for dense pre-training).
                         If False, each sequence becomes its own sample.
        chunk_size     : length of each chunk in concat mode
        stride         : sliding-window stride in concat mode

    Returns:
        List of ``{"tgt_in": LongTensor, "tgt_out": LongTensor}`` dicts.
    """
    samples: list[dict] = []

    if concat:
        flat_ids: list[int] = []
        for ids in token_id_lists:
            flat_ids.extend(ids)
        for start in range(0, len(flat_ids) - chunk_size, stride):
            chunk = flat_ids[start : start + chunk_size + 1]
            samples.append({
                "tgt_in":  torch.tensor(chunk[:-1], dtype=torch.long),
                "tgt_out": torch.tensor(chunk[1:],  dtype=torch.long),
            })
    else:
        for ids in token_id_lists:
            if len(ids) < 2:
                continue
            samples.append({
                "tgt_in":  torch.tensor(ids[:-1], dtype=torch.long),
                "tgt_out": torch.tensor(ids[1:],  dtype=torch.long),
            })

    return samples


# ─────────────────────────────────────────────────────────────────────────────
# SFT sample construction
# ─────────────────────────────────────────────────────────────────────────────

# Speaker name → role mapping for sharegpt format
_SHAREGPT_HUMAN_ROLES     = {"human", "user"}
_SHAREGPT_ASSISTANT_ROLES = {"gpt", "assistant", "bot"}


def _encode_sft_alpaca(
    record: dict,
    tokenizer: Tokenizer,
    chat_formatter: ChatFormatter,
) -> "tuple[list[int], list[int]] | None":
    """Encode one alpaca record into (token_ids, loss_mask) using ChatFormatter.

    The instruction (+optional input) is rendered as a user turn via the
    formatter; the output is the assistant turn.  Only the assistant tokens
    carry loss_mask=1.

    Returns None if the record is malformed.
    """
    instruction = record.get("instruction", "").strip()
    input_text  = record.get("input", "").strip()
    output_text = record.get("output", "").strip()

    if not instruction or not output_text:
        return None

    # Combine instruction + input into a single user message
    user_text = instruction
    if input_text:
        user_text = f"{instruction}\n\n{input_text}"

    # Render user turn + assistant prefix via format_single_turn (no history)
    user_turn_text, assistant_start_text = chat_formatter.format_single_turn(user_text)
    # Render the full assistant turn and strip the opening prefix to get the body
    full_assistant_text = chat_formatter.assistant_template.format(assistant=output_text)
    assistant_body = full_assistant_text[len(assistant_start_text):]

    prompt_ids = tokenizer.encode(
        user_turn_text + assistant_start_text, add_bos=True, add_eos=False
    )
    output_ids = tokenizer.encode(assistant_body, add_bos=False, add_eos=False)

    token_ids = prompt_ids + output_ids
    loss_mask = [0] * len(prompt_ids) + [1] * len(output_ids)

    return token_ids, loss_mask


def _encode_sft_sharegpt(
    record: dict,
    tokenizer: Tokenizer,
    chat_formatter: ChatFormatter,
) -> "tuple[list[int], list[int]] | None":
    """Encode one sharegpt record into (token_ids, loss_mask) using ChatFormatter.

    Each human/assistant turn pair is rendered via the formatter templates so
    that the token boundaries match exactly what the model sees at inference.

    Expects ``{"conversations": [{"from": "human"/"gpt", "value": "..."}, ...]}``.
    Returns None if the record is malformed or has no assistant turn.
    """
    conversations = record.get("conversations", [])
    if not conversations:
        return None

    token_ids: list[int] = []
    loss_mask:  list[int] = []
    has_assistant_turn = False

    chat_formatter.reset()

    # Walk through turns; for each human+assistant pair, encode using templates
    pending_user: "str | None" = None

    for turn in conversations:
        speaker = turn.get("from", "").lower()
        text    = turn.get("value", "").strip()
        if not text:
            continue

        is_human     = speaker in _SHAREGPT_HUMAN_ROLES
        is_assistant = speaker in _SHAREGPT_ASSISTANT_ROLES

        if is_human:
            pending_user = text

        elif is_assistant and pending_user is not None:
            # format_single_turn renders only this user turn + assistant prefix,
            # without re-rendering prior history (which is already in token_ids).
            user_turn_text, assistant_start_text = chat_formatter.format_single_turn(pending_user)
            # The assistant response body follows the assistant_start prefix
            full_assistant_text = chat_formatter.assistant_template.format(assistant=text)
            assistant_body = full_assistant_text[len(assistant_start_text):]

            user_ids      = tokenizer.encode(
                user_turn_text + assistant_start_text,
                add_bos=(len(token_ids) == 0),
                add_eos=False,
            )
            assistant_ids = tokenizer.encode(assistant_body, add_bos=False, add_eos=False)

            token_ids.extend(user_ids)
            loss_mask.extend([0] * len(user_ids))
            token_ids.extend(assistant_ids)
            loss_mask.extend([1] * len(assistant_ids))

            pending_user = None
            has_assistant_turn = True

    if not has_assistant_turn:
        return None

    return token_ids, loss_mask


def build_sft_samples(
    records: list[dict],
    tokenizer: Tokenizer,
    chat_formatter: ChatFormatter,
    chat_format: str = "alpaca",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> list[dict]:
    """Convert SFT records into training sample dicts with loss masks.

    Each sample contains:
      - ``tgt_in``   : input token ids (all but last)
      - ``tgt_out``  : target token ids (all but first)
      - ``loss_mask``: 1 where loss should be computed (assistant tokens only)

    Dialogue is rendered through *chat_formatter* so that training and
    inference use identical prompt formatting.  Long sequences are truncated
    to *chunk_size + 1* tokens.

    Args:
        records       : list of raw dicts (alpaca or sharegpt format)
        tokenizer     : fitted Tokenizer
        chat_formatter: ChatFormatter instance used to render prompts
        chat_format   : "alpaca" or "sharegpt"
        chunk_size    : maximum sequence length (in tokens)

    Returns:
        List of sample dicts.
    """
    encode_fn = _encode_sft_alpaca if chat_format == "alpaca" else _encode_sft_sharegpt

    samples: list[dict] = []
    skipped = 0

    for record in records:
        result = encode_fn(record, tokenizer, chat_formatter)
        if result is None:
            skipped += 1
            continue

        token_ids, loss_mask = result

        # Truncate to chunk_size + 1 (need one extra for the shift)
        max_len = chunk_size + 1
        if len(token_ids) > max_len:
            token_ids = token_ids[:max_len]
            loss_mask = loss_mask[:max_len]

        if len(token_ids) < 2:
            skipped += 1
            continue

        samples.append({
            "tgt_in":    torch.tensor(token_ids[:-1], dtype=torch.long),
            "tgt_out":   torch.tensor(token_ids[1:],  dtype=torch.long),
            "loss_mask": torch.tensor(loss_mask[1:],  dtype=torch.long),
        })

    if skipped:
        logger.info(f"      Skipped {skipped:,} malformed / empty records")

    return samples


# ─────────────────────────────────────────────────────────────────────────────
# Shared tokenisation helper
# ─────────────────────────────────────────────────────────────────────────────

def _encode_one(args: tuple) -> list[int]:
    """Thread-pool worker: encode a single text for pretrain."""
    tokenizer, text = args
    return tokenizer.encode(text, add_bos=True, add_eos=True)


# ─────────────────────────────────────────────────────────────────────────────
# TrainPipeline
# ─────────────────────────────────────────────────────────────────────────────

class TrainPipeline:
    """End-to-end training pipeline supporting both pre-training and SFT.

    Encapsulates tokenizer training, data preparation, model construction,
    and the training loop.  Each step caches its results so that subsequent
    runs skip expensive computation automatically.

    Example (pretrain)::

        pipeline = TrainPipeline(output_dir="model/my_model")
        pipeline.run(task="pretrain", data_path="data/corpus/")

    Example (SFT)::

        pipeline = TrainPipeline(output_dir="model/sft_model")
        pipeline.run(
            task="sft",
            chat_format="alpaca",
            data_path="data/alpaca.jsonl",
            resume_from="model/pretrain_model",
        )
    """

    def __init__(self, output_dir: str = "model/train_model") -> None:
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    # ── Step 1: load data ────────────────────────────────────────────────────

    def load_data(self, data_path: str, url_cache_path: "str | None" = None) -> list:
        """Delegate to the module-level :func:`load_data`."""
        return load_data(data_path, url_cache_path=url_cache_path)

    # ── Step 2: tokenizer ────────────────────────────────────────────────────

    def build_tokenizer(
        self,
        texts: list[str],
        vocab_size: int = DEFAULT_VOCAB_SIZE,
        tokenizer_type: "Literal['basic', 'bpe']" = "basic",
        # basic tokenizer parameters
        min_frequency: "int | dict[int, int]" = 2,
        max_ngram_len: int = 4,
        length_score_alpha: float = 0.75,
        # bpe tokenizer parameters
        bpe_min_frequency: int = 2,
        # shared parameters
        pretok_pattern: "str | None" = None,
        show_progress: bool = True,
    ) -> Tokenizer:
        """Train a tokenizer on *texts*, or load from cache if already saved.

        Args:
            texts              : raw text samples used for vocabulary training
            vocab_size         : target vocabulary size
            tokenizer_type     : ``"basic"`` for the N-gram frequency-filtered
                                 tokenizer (default), or ``"bpe"`` for Byte Pair
                                 Encoding.
            min_frequency      : (basic only) minimum n-gram frequency to include
                                 in vocab; accepts an ``int`` (applied to all
                                 n-gram lengths) or a ``dict[int, int]`` mapping
                                 n-gram length → minimum count.
            max_ngram_len      : (basic only) maximum n-gram length to consider.
            length_score_alpha : (basic only) exponent for length-normalised
                                 ranking score; higher values favour longer tokens.
            bpe_min_frequency  : (bpe only) minimum pair merge frequency;
                                 merging stops when the best pair falls below
                                 this threshold.
            pretok_pattern     : optional regex pre-tokenizer pattern shared by
                                 both tokenizer types.  ``None`` uses the default
                                 GPT-2-style pattern.
            show_progress      : whether to print training progress.

        Returns:
            A ready-to-use :class:`Tokenizer` (or :class:`BPETokenizer`).
        """
        tokenizer_path = os.path.join(self.output_dir, "tokenizer.json")
        if os.path.exists(tokenizer_path):
            logger.info("[2/5] Loading tokenizer from cache …")
            tokenizer = Tokenizer.from_pretrained(self.output_dir)
            logger.info(f"      Tokenizer vocab size: {tokenizer.vocab_size:,}  (cached)")
            return tokenizer

        logger.info(f"[2/5] Training {tokenizer_type} tokenizer (vocab_size={vocab_size}) …")
        if tokenizer_type == "bpe":
            tokenizer = BPETokenizer.train(
                texts=texts,
                vocab_size=vocab_size,
                pretok_pattern=pretok_pattern,
                min_frequency=bpe_min_frequency,
                show_progress=show_progress,
            )
        else:
            tokenizer = Tokenizer.train(
                texts=texts,
                vocab_size=vocab_size,
                min_frequency=min_frequency,
                max_ngram_len=max_ngram_len,
                length_score_alpha=length_score_alpha,
                pretok_pattern=pretok_pattern,
                show_progress=show_progress,
            )
        tokenizer.save(self.output_dir)
        logger.info(f"      Tokenizer vocab size: {tokenizer.vocab_size:,}")
        return tokenizer

    # ── Step 3: split corpus ─────────────────────────────────────────────────

    def split_corpus(
        self,
        records: list,
        tokenizer: Tokenizer,
        task: str = "pretrain",
        chat_format: str = "alpaca",
        chat_formatter: "ChatFormatter | None" = None,
        concat: bool = True,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        stride: int = DEFAULT_STRIDE,
        test_ratio: float = DEFAULT_TEST_RATIO,
        num_workers: int = DEFAULT_TOKENIZE_WORKERS,
    ) -> tuple[list[dict], list[dict]]:
        """Tokenise *records* and split into train / test sample lists.

        Results are cached to ``<output_dir>/train.pt`` and
        ``<output_dir>/test.pt``.  Pretrain tokenisation is parallelised
        across *num_workers* threads; SFT encoding runs single-threaded
        (each record is small and encoding is fast).

        Args:
            records        : raw records returned by :func:`load_data`
            tokenizer      : fitted :class:`Tokenizer`
            task           : "pretrain" or "sft"
            chat_format    : "alpaca" or "sharegpt" (only used when task="sft")
            chat_formatter : ChatFormatter used to render prompts; a default
                             instance is created when None and task="sft"
            concat         : passed through to :func:`build_pretrain_samples`
            chunk_size     : token chunk / max sequence length
            stride         : sliding-window stride (pretrain concat mode only)
            test_ratio     : fraction of samples reserved for the test set
            num_workers    : parallel tokenisation threads (pretrain only)

        Returns:
            ``(train_samples, test_samples)``
        """
        train_cache = os.path.join(self.output_dir, "train.pt")
        test_cache  = os.path.join(self.output_dir, "test.pt")

        if os.path.exists(train_cache) and os.path.exists(test_cache):
            logger.info("[3/5] Loading train/test splits from cache …")
            train_samples = torch.load(train_cache, weights_only=True)
            test_samples  = torch.load(test_cache,  weights_only=True)
            logger.info(
                f"      Samples — train: {len(train_samples):,}  "
                f"test: {len(test_samples):,}  (cached)"
            )
            return train_samples, test_samples

        if task == "sft":
            if chat_formatter is None:
                chat_formatter = ChatFormatter()
            all_samples = self._build_sft_split(
                records, tokenizer,
                chat_format=chat_format,
                chat_formatter=chat_formatter,
                chunk_size=chunk_size,
            )
        else:
            all_samples = self._build_pretrain_split(
                records, tokenizer,
                concat=concat, chunk_size=chunk_size,
                stride=stride, num_workers=num_workers,
            )

        random.shuffle(all_samples)
        num_test      = max(1, int(len(all_samples) * test_ratio))
        test_samples  = all_samples[:num_test]
        train_samples = all_samples[num_test:]

        logger.info(
            f"      Samples — train: {len(train_samples):,}  "
            f"test: {len(test_samples):,}"
        )

        torch.save(train_samples, train_cache)
        torch.save(test_samples,  test_cache)
        logger.info(f"      Splits cached → {train_cache}, {test_cache}")

        return train_samples, test_samples

    def _build_pretrain_split(
        self,
        records: list,
        tokenizer: Tokenizer,
        concat: bool,
        chunk_size: int,
        stride: int,
        num_workers: int,
    ) -> list[dict]:
        texts = _extract_pretrain_texts(records)
        logger.info(
            f"[3/5] Tokenising {len(texts):,} texts "
            f"(workers={num_workers}, concat={concat}) …"
        )

        token_id_lists: list[list[int]] = [None] * len(texts)  # type: ignore[list-item]
        work_items = [(tokenizer, text) for text in texts]

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_index = {
                executor.submit(_encode_one, item): idx
                for idx, item in enumerate(work_items)
            }
            completed  = 0
            report_every = max(1, len(texts) // 20)
            for future in as_completed(future_to_index):
                token_id_lists[future_to_index[future]] = future.result()
                completed += 1
                if completed % report_every == 0:
                    logger.info(f"      Tokenised {completed:,} / {len(texts):,} …")

        total_tokens = sum(len(ids) for ids in token_id_lists)
        logger.info(f"      Total tokens: {total_tokens:,}")

        return build_pretrain_samples(
            token_id_lists, concat=concat, chunk_size=chunk_size, stride=stride
        )

    def _build_sft_split(
        self,
        records: list,
        tokenizer: Tokenizer,
        chat_format: str,
        chat_formatter: ChatFormatter,
        chunk_size: int,
    ) -> list[dict]:
        logger.info(f"[3/5] Building SFT samples (format={chat_format}) …")
        # Filter to dicts only; txt / URL sources are not valid for SFT
        dict_records = [r for r in records if isinstance(r, dict)]
        if len(dict_records) < len(records):
            logger.info(f"      Skipped {len(records) - len(dict_records):,} non-dict records")
        samples = build_sft_samples(
            dict_records, tokenizer,
            chat_formatter=chat_formatter,
            chat_format=chat_format,
            chunk_size=chunk_size,
        )
        logger.info(f"      Built {len(samples):,} SFT samples")
        return samples

    # ── Step 4: build model ──────────────────────────────────────────────────

    def build_model(
        self,
        tokenizer: Tokenizer,
        resume_from: "str | None" = None,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 12,
        d_ff: int = 2048,
        max_len: int = DEFAULT_CHUNK_SIZE,
        dropout: float = 0.1,
        dtype: str = "float32",
        num_kv_heads: int = 0,
        rope_base: int = 10000,
        pos_encoding: str = "rope",
        ffn_type: str = "glu",
        activation: str = "silu",
        norm_layer: str = "rmsnorm",
        norm_type: str = "pre",
        qk_norm: bool = False,
        v_norm: bool = False,
        tie_weights: bool = False,
        learnable_residual: bool = False,
        is_chat: bool = False,
    ) -> tuple[Transformer, ModelConfig]:
        """Instantiate a decoder-only LM, optionally resuming from a checkpoint.

        When *resume_from* is provided the model config and weights are loaded
        from that directory, allowing continued pre-training or SFT without
        restarting from scratch.

        Args:
            tokenizer          : fitted tokenizer (provides vocab_size and special ids)
            resume_from        : path to a saved model directory to resume from;
                                 when set, all architecture arguments below are ignored.
            d_model            : model hidden dimension
            num_heads          : number of attention heads
            num_layers         : number of decoder layers
            d_ff               : feed-forward hidden dimension
            max_len            : maximum sequence length (for RoPE cache)
            dropout            : dropout probability
            dtype              : weight dtype — "float32", "float16", or "bfloat16"
            num_kv_heads       : KV heads for GQA/MQA (0 = same as num_heads)
            rope_base          : RoPE frequency base
            pos_encoding       : "rope" or "sinusoidal"
            ffn_type           : "glu" or "standard"
            activation         : activation function name
            norm_layer         : "rmsnorm" or "layernorm"
            norm_type          : "pre" or "post"
            qk_norm            : apply per-head Q/K normalisation
            v_norm             : apply per-head V normalisation
            tie_weights        : tie embedding and output projection weights
            learnable_residual : add learnable scalar on each residual branch
            is_chat            : mark model as a chat/instruction-tuned model

        Returns:
            ``(model, cfg)`` tuple.
        """
        if resume_from is not None:
            logger.info(f"[4/5] Resuming model from: {resume_from}")
            model = Transformer.from_pretrained(resume_from)
            cfg   = model.cfg
            num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(
                f"      Parameters: {num_params:,}  "
                f"dtype: {cfg.dtype}  (resumed)"
            )
            return model, cfg

        cfg = ModelConfig(
            vocab_size         = tokenizer.vocab_size,
            pad_idx            = tokenizer.pad_id,
            bos_idx            = tokenizer.bos_id,
            eos_idx            = tokenizer.eos_id,
            model_type         = "lm",
            d_model            = d_model,
            num_heads          = num_heads,
            num_encoder_layers = 0,
            num_decoder_layers = num_layers,
            d_ff               = d_ff,
            max_len            = max_len + 2,
            dropout            = dropout,
            dtype              = dtype,
            num_kv_heads       = num_kv_heads,
            rope_base          = rope_base,
            pos_encoding       = pos_encoding,
            ffn_type           = ffn_type,
            activation         = activation,
            norm_layer         = norm_layer,
            norm_type          = norm_type,
            qk_norm            = qk_norm,
            v_norm             = v_norm,
            tie_weights        = tie_weights,
            learnable_residual = learnable_residual,
            is_chat            = is_chat,
        )
        model = Transformer(cfg)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"[4/5] Model built — parameters: {num_params:,}  dtype: {dtype}")
        return model, cfg

    # ── Convenience: data pipeline (steps 1-3) ──────────────────────────────

    def run_data_pipeline(
        self,
        data_path: str,
        task: str = "pretrain",
        chat_format: str = "alpaca",
        chat_formatter: "ChatFormatter | None" = None,
        vocab_size: int = DEFAULT_VOCAB_SIZE,
        min_frequency: int = 2,
        max_ngram_len: int = 4,
        concat: bool = True,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        stride: int = DEFAULT_STRIDE,
        test_ratio: float = DEFAULT_TEST_RATIO,
        num_workers: int = DEFAULT_TOKENIZE_WORKERS,
        url_cache_path: "str | None" = None,
    ) -> "tuple[Tokenizer, list[dict], list[dict]]":
        """Run steps 1-3: load data, train tokenizer, build train/test splits.

        Returns:
            ``(tokenizer, train_samples, test_samples)``
        """
        records = self.load_data(data_path, url_cache_path=url_cache_path)

        texts_for_tokenizer: list[str] = []
        if task == "pretrain":
            texts_for_tokenizer = _extract_pretrain_texts(records)
        else:
            # For SFT, collect all text fields for tokenizer training
            for record in records:
                if isinstance(record, dict):
                    for key in ("instruction", "input", "output", "value"):
                        if key in record and isinstance(record[key], str):
                            texts_for_tokenizer.append(record[key])

        tokenizer = self.build_tokenizer(
            texts_for_tokenizer,
            vocab_size    = vocab_size,
            min_frequency = min_frequency,
            max_ngram_len = max_ngram_len,
        )

        train_samples, test_samples = self.split_corpus(
            records,
            tokenizer,
            task           = task,
            chat_format    = chat_format,
            chat_formatter = chat_formatter,
            concat         = concat,
            chunk_size     = chunk_size,
            stride         = stride,
            test_ratio     = test_ratio,
            num_workers    = num_workers,
        )

        return tokenizer, train_samples, test_samples

    # ── Convenience: model pipeline (step 4) ────────────────────────────────

    def run_model_pipeline(
        self,
        tokenizer: Tokenizer,
        resume_from: "str | None" = None,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 12,
        d_ff: int = 2048,
        dropout: float = 0.1,
        dtype: str = "float32",
        num_kv_heads: int = 0,
        rope_base: int = 10000,
        pos_encoding: str = "rope",
        ffn_type: str = "glu",
        activation: str = "silu",
        norm_layer: str = "rmsnorm",
        norm_type: str = "pre",
        qk_norm: bool = False,
        v_norm: bool = False,
        tie_weights: bool = False,
        learnable_residual: bool = False,
        is_chat: bool = False,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> "tuple[Transformer, ModelConfig]":
        """Run step 4: build or resume the model.

        Returns:
            ``(model, cfg)``
        """
        return self.build_model(
            tokenizer          = tokenizer,
            resume_from        = resume_from,
            d_model            = d_model,
            num_heads          = num_heads,
            num_layers         = num_layers,
            d_ff               = d_ff,
            max_len            = chunk_size,
            dropout            = dropout,
            dtype              = dtype,
            num_kv_heads       = num_kv_heads,
            rope_base          = rope_base,
            pos_encoding       = pos_encoding,
            ffn_type           = ffn_type,
            activation         = activation,
            norm_layer         = norm_layer,
            norm_type          = norm_type,
            qk_norm            = qk_norm,
            v_norm             = v_norm,
            tie_weights        = tie_weights,
            learnable_residual = learnable_residual,
            is_chat            = is_chat,
        )

    # ── Step 5: training ─────────────────────────────────────────────────────

    def run_training(
        self,
        model: Transformer,
        train_samples: list[dict],
        test_samples: list[dict],
        tokenizer: Tokenizer,
        device: "str | torch.device" = "cpu",
        batch_size: int = 32,
        lr: float = 3e-4,
        num_epochs: int = 3,
        warmup_steps: int = 400,
        lr_scheduler: str = "rsqrt",
        mixed_precision: str = "no",
        save_every_steps: int = 500,
        max_checkpoints: int = 3,
        eval_every_steps: int = 1000,
        log_interval: int = 10,
    ) -> None:
        """Configure and run the training loop.

        Args:
            model            : Transformer model to train
            cfg              : ModelConfig used to build the model
            train_samples    : list of training sample dicts
            test_samples     : list of evaluation sample dicts
            tokenizer        : fitted tokenizer (saved alongside checkpoints)
            device           : torch device string or object
            batch_size       : samples per gradient update
            lr               : peak learning rate
            num_epochs       : number of full passes over the dataset
            warmup_steps     : linear LR warm-up steps
            mixed_precision  : "no" | "fp16" | "bf16"
            save_every_steps : checkpoint frequency (0 = only at end)
            max_checkpoints  : maximum recent checkpoints to keep
            eval_every_steps : eval frequency (0 = disabled)
            log_interval     : print loss every N steps
        """
        logger.info("[5/5] Training …")

        train_cfg = TrainConfig(
            batch_size                  = batch_size,
            lr                          = lr,
            warmup_steps                = warmup_steps,
            lr_scheduler                = lr_scheduler,
            max_grad_norm               = 1.0,
            num_epochs                  = num_epochs,
            gradient_accumulation_steps = 1,
            mixed_precision             = mixed_precision,
            gradient_checkpointing      = False,
            save_every_steps            = save_every_steps,
            max_checkpoints             = max_checkpoints,
            eval_every_steps            = eval_every_steps,
            log_interval                = log_interval,
            device                      = str(device),
        )

        pad_collate  = partial(collate_fn, pad_id=model.cfg.pad_idx)
        train_loader = DataLoader(
            TransformerDataset(train_samples),
            batch_size = train_cfg.batch_size,
            shuffle    = True,
            collate_fn = pad_collate,
        )
        eval_loader = DataLoader(
            TransformerDataset(test_samples),
            batch_size = train_cfg.batch_size,
            shuffle    = False,
            collate_fn = pad_collate,
        )

        train_lm(
            model           = model,
            train_cfg       = train_cfg,
            dataloader      = train_loader,
            save_path       = self.output_dir,
            eval_dataloader = eval_loader,
            callbacks       = [CopyTokenizerCallback(self.output_dir)],
        )

    # ── Convenience: full pipeline in one call ───────────────────────────────

    def run(
        self,
        data_path: str,
        device: "str | torch.device" = "cpu",
        resume_from: "str | None" = None,
        url_cache_path: "str | None" = None,
        # task
        task: str = "pretrain",
        chat_format: str = "alpaca",
        chat_formatter: "ChatFormatter | None" = None,
        # tokenizer
        vocab_size: int = DEFAULT_VOCAB_SIZE,
        min_frequency: int = 2,
        max_ngram_len: int = 4,
        # samples
        concat: bool = True,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        stride: int = DEFAULT_STRIDE,
        test_ratio: float = DEFAULT_TEST_RATIO,
        num_workers: int = DEFAULT_TOKENIZE_WORKERS,
        # model architecture (ignored when resume_from is set)
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 12,
        d_ff: int = 2048,
        dropout: float = 0.1,
        dtype: str = "float32",
        num_kv_heads: int = 0,
        rope_base: int = 10000,
        pos_encoding: str = "rope",
        ffn_type: str = "glu",
        activation: str = "silu",
        norm_layer: str = "rmsnorm",
        norm_type: str = "pre",
        qk_norm: bool = False,
        v_norm: bool = False,
        tie_weights: bool = False,
        learnable_residual: bool = False,
        is_chat: bool = False,
        # training
        batch_size: int = 32,
        lr: float = 3e-4,
        num_epochs: int = 3,
        warmup_steps: int = 400,
        lr_scheduler: str = "rsqrt",
        mixed_precision: str = "no",
        save_every_steps: int = 500,
        max_checkpoints: int = 3,
        eval_every_steps: int = 1000,
        log_interval: int = 10,
    ) -> None:
        """Run the full training pipeline end-to-end.

        Convenience wrapper that calls each step in order.  All arguments
        are forwarded to the corresponding step methods.

        Args:
            data_path      : data source (file / directory / URL)
            device         : torch device
            resume_from    : checkpoint directory to resume from
            url_cache_path : local cache path for URL downloads
            task           : "pretrain" or "sft"
            chat_format    : "alpaca" or "sharegpt" (only used when task="sft")
            ...            : all other args forwarded to the respective step
        """
        if task not in ("pretrain", "sft"):
            raise ValueError(f"task must be 'pretrain' or 'sft', got: {task!r}")
        if task == "sft" and chat_format not in ("alpaca", "sharegpt"):
            raise ValueError(f"chat_format must be 'alpaca' or 'sharegpt', got: {chat_format!r}")

        tokenizer, train_samples, test_samples = self.run_data_pipeline(
            data_path      = data_path,
            task           = task,
            chat_format    = chat_format,
            chat_formatter = chat_formatter,
            vocab_size     = vocab_size,
            min_frequency  = min_frequency,
            max_ngram_len  = max_ngram_len,
            concat         = concat,
            chunk_size     = chunk_size,
            stride         = stride,
            test_ratio     = test_ratio,
            num_workers    = num_workers,
            url_cache_path = url_cache_path,
        )

        model, cfg = self.run_model_pipeline(
            tokenizer          = tokenizer,
            resume_from        = resume_from,
            d_model            = d_model,
            num_heads          = num_heads,
            num_layers         = num_layers,
            d_ff               = d_ff,
            dropout            = dropout,
            dtype              = dtype,
            num_kv_heads       = num_kv_heads,
            rope_base          = rope_base,
            pos_encoding       = pos_encoding,
            ffn_type           = ffn_type,
            activation         = activation,
            norm_layer         = norm_layer,
            norm_type          = norm_type,
            qk_norm            = qk_norm,
            v_norm             = v_norm,
            tie_weights        = tie_weights,
            learnable_residual = learnable_residual,
            is_chat            = is_chat,
            chunk_size         = chunk_size,
        )

        self.run_training(
            model,
            train_samples,
            test_samples,
            tokenizer        = tokenizer,
            device           = device,
            batch_size       = batch_size,
            lr               = lr,
            num_epochs       = num_epochs,
            warmup_steps     = warmup_steps,
            lr_scheduler     = lr_scheduler,
            mixed_precision  = mixed_precision,
            save_every_steps = save_every_steps,
            max_checkpoints  = max_checkpoints,
            eval_every_steps = eval_every_steps,
            log_interval     = log_interval,
        )

        tokenizer.save(self.output_dir)
        logger.info(f"\nAll artefacts saved to: {self.output_dir}/")

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="General-purpose training script for transformer_minimal.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── I/O ──────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--data", required=True,
        help="Path to training data (file, directory, or URL).",
    )
    parser.add_argument(
        "--output", required=True,
        help="Directory to save checkpoints and tokenizer.",
    )
    parser.add_argument(
        "--resume", default=None,
        help="Checkpoint directory to resume training from.",
    )
    parser.add_argument(
        "--url-cache", default=None, dest="url_cache",
        help="Local path to cache downloaded URL data.",
    )

    # ── Task ─────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--task", default="pretrain", choices=["pretrain", "sft"],
        help="Training task: 'pretrain' or 'sft'.",
    )
    parser.add_argument(
        "--chat-format", default="alpaca", dest="chat_format",
        choices=["alpaca", "sharegpt"],
        help="Chat format for SFT task.",
    )

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    parser.add_argument("--vocab-size",    type=int,   default=DEFAULT_VOCAB_SIZE)
    parser.add_argument("--min-frequency", type=int,   default=2)
    parser.add_argument("--max-ngram-len", type=int,   default=4)

    # ── Data ─────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--concat", action="store_true", default=True,
        help="Concatenate all texts before chunking (pretrain).",
    )
    parser.add_argument(
        "--no-concat", dest="concat", action="store_false",
        help="Disable text concatenation.",
    )
    parser.add_argument("--chunk-size",       type=int,   default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--stride",           type=int,   default=DEFAULT_STRIDE)
    parser.add_argument("--test-ratio",       type=float, default=DEFAULT_TEST_RATIO)
    parser.add_argument("--tokenize-workers", type=int,   default=DEFAULT_TOKENIZE_WORKERS)

    # ── Model architecture ────────────────────────────────────────────────────
    parser.add_argument("--d-model",    type=int,   default=512)
    parser.add_argument("--num-heads",  type=int,   default=8)
    parser.add_argument("--num-layers", type=int,   default=12)
    parser.add_argument("--d-ff",       type=int,   default=2048)
    parser.add_argument("--dropout",    type=float, default=0.1)
    parser.add_argument(
        "--dtype", default="float32",
        choices=["float32", "float16", "bfloat16"],
    )
    parser.add_argument(
        "--num-kv-heads", type=int, default=0, dest="num_kv_heads",
        help="Number of KV heads for GQA (0 = same as num_heads).",
    )
    parser.add_argument("--rope-base", type=int, default=10000,
                        help="Base for RoPE positional encoding.")
    parser.add_argument(
        "--pos-encoding", default="rope", dest="pos_encoding",
        choices=["rope", "sinusoidal", "none"],
    )
    parser.add_argument(
        "--ffn-type", default="glu", dest="ffn_type",
        choices=["glu", "standard"],
    )
    parser.add_argument(
        "--activation", default="silu",
        choices=["silu", "gelu", "relu"],
    )
    parser.add_argument(
        "--norm-layer", default="rmsnorm", dest="norm_layer",
        choices=["rmsnorm", "layernorm"],
    )
    parser.add_argument(
        "--norm-type", default="pre", dest="norm_type",
        choices=["pre", "post"],
    )
    parser.add_argument("--qk-norm",           dest="qk_norm",     action="store_true", default=False,
                        help="Apply RMSNorm to Q and K before attention.")
    parser.add_argument("--v-norm",            action="store_true", default=False,
                        help="Apply RMSNorm to V before attention.")
    parser.add_argument("--tie-weights",       dest="tie_weights",  action="store_true", default=False,
                        help="Tie input embedding and output projection weights.")
    parser.add_argument("--learnable-residual", action="store_true", default=False,
                        help="Add learnable scalar to residual connections.")

    # ── Training ──────────────────────────────────────────────────────────────
    parser.add_argument("--batch-size",        type=int,   default=32)
    parser.add_argument("--lr",                type=float, default=3e-4)
    parser.add_argument("--num-epochs",        type=int,   default=3)
    parser.add_argument("--warmup-steps",      type=int,   default=400)
    parser.add_argument(
        "--lr-scheduler", default="rsqrt", dest="lr_scheduler",
        choices=["rsqrt", "cosine"],
        help="LR schedule: 'rsqrt' = inverse-square-root (default), "
             "'cosine' = linear warmup then cosine decay to 0.",
    )
    parser.add_argument(
        "--mixed-precision", default="no",
        choices=["no", "fp16", "bf16"],
    )
    parser.add_argument("--save-every-steps",  type=int,   default=500)
    parser.add_argument("--max-checkpoints",   type=int,   default=3)
    parser.add_argument("--eval-every-steps",  type=int,   default=1000)
    parser.add_argument("--log-interval",      type=int,   default=10)

    # ── Runtime ──────────────────────────────────────────────────────────────
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument(
        "--task-name", default=None, dest="task_name",
        help="Name used as the training log file stem (e.g. 'shakespeare_pretrain'). "
             "Defaults to a timestamp (YYYYMMDD_HHMMSS) when not provided.",
    )

    return parser.parse_args()

def main() -> None:
    args = parse_args()

    _init_file_logger(args.task_name)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    pipeline = TrainPipeline(output_dir=args.output)
    pipeline.run(
        data_path          = args.data,
        device             = args.device,
        resume_from        = args.resume,
        url_cache_path     = args.url_cache,
        task               = args.task,
        chat_format        = args.chat_format,
        vocab_size         = args.vocab_size,
        min_frequency      = args.min_frequency,
        max_ngram_len      = args.max_ngram_len,
        concat             = args.concat,
        chunk_size         = args.chunk_size,
        stride             = args.stride,
        test_ratio         = args.test_ratio,
        num_workers        = args.tokenize_workers,
        d_model            = args.d_model,
        num_heads          = args.num_heads,
        num_layers         = args.num_layers,
        d_ff               = args.d_ff,
        dropout            = args.dropout,
        dtype              = args.dtype,
        num_kv_heads       = args.num_kv_heads,
        rope_base          = args.rope_base,
        pos_encoding       = args.pos_encoding,
        ffn_type           = args.ffn_type,
        activation         = args.activation,
        norm_layer         = args.norm_layer,
        norm_type          = args.norm_type,
        qk_norm            = args.qk_norm,
        v_norm             = args.v_norm,
        tie_weights        = args.tie_weights,
        learnable_residual = args.learnable_residual,
        batch_size         = args.batch_size,
        lr                 = args.lr,
        num_epochs         = args.num_epochs,
        warmup_steps       = args.warmup_steps,
        lr_scheduler       = args.lr_scheduler,
        mixed_precision    = args.mixed_precision,
        save_every_steps   = args.save_every_steps,
        max_checkpoints    = args.max_checkpoints,
        eval_every_steps   = args.eval_every_steps,
        log_interval       = args.log_interval,
    )

if __name__ == "__main__":
    main()