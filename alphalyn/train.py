"""train.py – Standalone model training loop (decoupled from self-play generation).

Watches a data directory produced by generate.py and trains the network
whenever a new iteration's data from all workers is fully available.

How it works
------------
1. Track ``next_iter`` (starting from 1).  Wait until every worker's
   ``worker{id}.npz`` file exists under ``data_dir/iter_{next_iter}/``.
2. Collect the most recent ``replay_buffer`` completed iters and train for
   ``train_epochs`` epochs.
3. Save the updated checkpoint, then write ``{save_path}.DONE`` so that
   generate.py workers know new weights are ready for hot-reloading.
4. Increment ``next_iter`` and repeat indefinitely until interrupted (Ctrl-C).

Usage
-----
    # Watch self_play_data/, 2 workers per iter, train on the latest 5 iters
    python train.py --game connect4 --save-path bakc4/checkpoint.pt \\
        --data-dir self_play_data --num-workers 2 --replay-buffer 5

    # GPU training, custom poll interval
    python train.py --game othello --save-path checkpoint.pt \\
        --device cuda --num-workers 4 --poll-interval 30
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from model import PolicyValueNet, ResNetConfig
from train_utils import (
    make_game,
    _add_game_specific_args,
    iter_data_dir,
    iter_checkpoint_path,
    iter_checkpoint_done_path,
    worker_npz_path,
)
from augmentation import make_augment_fn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """Configuration for the standalone training loop.

    Attributes
    ----------
    game_name:
        Game name; one of othello / gomoku / connect4 / tictactoe.
    save_path:
        Path to save / load the model checkpoint (.pt).
        After each training round a ``{save_path}.DONE`` marker is also
        written so that generate.py workers know new weights are ready.
    data_dir:
        Root directory written by generate.py; contains iter_* subdirs.
    num_workers:
        Number of parallel generate.py workers per iteration.  Training for
        iter N does not start until all ``worker{0..num_workers-1}.npz``
        files are present under ``data_dir/iter_{N}/``.
    replay_buffer_batches:
        Number of most-recent completed iters to include in each training
        round.  Older iters are ignored.
    poll_interval:
        Seconds to sleep between data-directory scans when waiting for
        workers to finish.
    train_epochs:
        Number of gradient-update epochs per training round.
    batch_size:
        Mini-batch size for the DataLoader.
    lr:
        Initial learning rate for the Adam optimiser.
    weight_decay:
        L2 regularisation coefficient.
    device:
        Torch device string, e.g. ``"cpu"``, ``"cuda"``, or ``"mps"``.
    num_input_planes:
        Number of input feature planes; must match generate.py.
    num_filters:
        ResNet conv filter count; must match generate.py / checkpoint.
    num_residual_blocks:
        Number of residual blocks; must match generate.py / checkpoint.
    value_head_hidden_size:
        Hidden layer width of the value head.
    lr_schedule:
        LR schedule: ``"none"`` | ``"step"`` | ``"cosine"``.
    lr_decay_step:
        Step schedule: decay LR every this many training rounds.
    lr_decay_gamma:
        Step schedule: multiplicative decay factor.
    lr_min:
        Cosine schedule: minimum LR.
    game_namespace:
        argparse.Namespace forwarded to the game's Args constructor.
    """
    game_name: str = "othello"
    save_path: str = "checkpoint.pt"
    data_dir: str = "self_play_data"
    num_workers: int = 1
    replay_buffer_batches: int = 5
    poll_interval: float = 10.0
    train_epochs: int = 1
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    device: str = "cpu"
    num_input_planes: int = 3
    num_filters: int = 128
    num_residual_blocks: int = 10
    value_head_hidden_size: int = 256
    lr_schedule: str = "none"
    lr_decay_step: int = 10
    lr_decay_gamma: float = 0.5
    lr_min: float = 1e-5
    augment: bool = False
    game_namespace: argparse.Namespace = field(default_factory=argparse.Namespace)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SelfPlayDataset(Dataset):
    """PyTorch Dataset that loads .npz files from a list of batch directories."""

    def __init__(self, batch_dirs: List[str]) -> None:
        self.boards: List[np.ndarray] = []
        self.policies: List[np.ndarray] = []
        self.values: List[float] = []

        for batch_dir in batch_dirs:
            for npz_path in glob.glob(os.path.join(batch_dir, "*.npz")):
                data = np.load(npz_path)
                self.boards.extend(data["boards"])
                self.policies.extend(data["policies"])
                self.values.extend(data["values"].tolist())

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        board = torch.from_numpy(self.boards[index])
        policy = torch.from_numpy(self.policies[index])
        value = torch.tensor(self.values[index], dtype=torch.float32)
        return board, policy, value


# ---------------------------------------------------------------------------
# Iter discovery
# ---------------------------------------------------------------------------

def _iter_is_ready(data_dir: str, iter_id: int, num_workers: int) -> bool:
    """Return True when all num_workers npz files exist under iter_{iter_id}/."""
    iter_path = iter_data_dir(data_dir, iter_id)
    if not os.path.isdir(iter_path):
        return False
    for worker_id in range(num_workers):
        if not os.path.exists(worker_npz_path(data_dir, iter_id, worker_id)):
            return False
    return True


def _collect_replay_dirs(data_dir: str, up_to_iter: int, num_iters: int) -> List[str]:
    """Return the most recent num_iters iter dirs up to and including up_to_iter."""
    first = max(1, up_to_iter - num_iters + 1)
    return [iter_data_dir(data_dir, i) for i in range(first, up_to_iter + 1)]


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _train_one_epoch(
    net: PolicyValueNet,
    device: torch.device,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    augment_fn: Optional[callable] = None,
) -> Tuple[float, float]:
    """Run one training epoch; return (avg_policy_loss, avg_value_loss)."""
    net.train()
    total_policy_loss = 0.0
    total_value_loss = 0.0
    num_batches = 0

    for boards, policies, values in loader:
        boards = boards.to(device)
        policies = policies.to(device)
        values = values.to(device)

        if augment_fn is not None:
            boards, policies, values = augment_fn(boards, policies, values)

        log_policy, predicted_value = net(boards)

        policy_loss = -(policies * log_policy).sum(dim=1).mean()
        value_loss = nn.functional.mse_loss(predicted_value, values)
        loss = policy_loss + value_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
        num_batches += 1

    return (
        total_policy_loss / max(num_batches, 1),
        total_value_loss / max(num_batches, 1),
    )


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(config: TrainingConfig) -> None:
    """Watch data_dir and train the network whenever a new iter's data is complete.

    The loop runs indefinitely until interrupted (Ctrl-C).  Each training round:
    1. Wait until all ``num_workers`` worker .npz files exist under
       ``data_dir/iter_{next_iter}/``.
    2. Load the most recent ``replay_buffer_batches`` iters into a Dataset.
    3. Run ``train_epochs`` epochs of gradient updates.
    4. Save the updated checkpoint and write ``{save_path}.DONE`` so that
       generate.py workers know new weights are ready to hot-reload.
    5. Increment next_iter and repeat.
    """
    game = make_game(config.game_name, config.game_namespace)
    net_config = ResNetConfig(
        board_height=game.height,
        board_width=game.width,
        num_input_planes=config.num_input_planes,
        num_actions=game.num_actions,
        num_filters=config.num_filters,
        num_residual_blocks=config.num_residual_blocks,
        value_head_hidden_size=config.value_head_hidden_size,
    )
    net = PolicyValueNet(net_config)

    save_dir = os.path.dirname(config.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # Load from the latest existing iter checkpoint, falling back to save_path.
    save_stem, save_ext = os.path.splitext(config.save_path)
    loaded = False
    existing_iter_ckpts = sorted(glob.glob(f"{save_stem}_iter*{save_ext}"))
    if existing_iter_ckpts:
        latest_ckpt = existing_iter_ckpts[-1]
        logger.info("Resuming from latest iter checkpoint '%s'", latest_ckpt)
        net.load_state_dict(torch.load(latest_ckpt, map_location="cpu"))
        loaded = True
    elif os.path.exists(config.save_path):
        logger.info("Loaded checkpoint from '%s'", config.save_path)
        net.load_state_dict(torch.load(config.save_path, map_location="cpu"))
        loaded = True

    if not loaded:
        logger.info("No checkpoint found — starting from random weights")
        torch.save(net.state_dict(), config.save_path)

    device = torch.device(config.device)
    net.to(device)

    optimizer = optim.Adam(net.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    scheduler: Optional[optim.lr_scheduler.LRScheduler] = None
    if config.lr_schedule == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=config.lr_decay_step, gamma=config.lr_decay_gamma
        )
        logger.info("LR schedule: step (decay=%.2f every %d rounds)", config.lr_decay_gamma, config.lr_decay_step)
    elif config.lr_schedule == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10000, eta_min=config.lr_min)
        logger.info("LR schedule: cosine (lr_min=%.2e)", config.lr_min)

    logger.info(
        "Training loop started — watching '%s' for iter_* dirs "
        "(num_workers=%d, poll_interval=%.1fs, replay=%d iters)",
        config.data_dir, config.num_workers, config.poll_interval, config.replay_buffer_batches,
    )

    augment_fn = None
    if config.augment:
        augment_fn = make_augment_fn(game)
        if augment_fn is not None:
            logger.info("Data augmentation enabled for game '%s'", game.name)
        else:
            logger.warning(
                "Augmentation requested but no symmetry group defined for '%s' — disabled",
                game.name,
            )

    next_iter = 1

    while True:
        iter_start = time.time()

        # Wait until all workers have written their .npz for this iter.
        wait_start = time.time()
        while not _iter_is_ready(config.data_dir, next_iter, config.num_workers):
            logger.debug(
                "Waiting for iter %d (%d workers)…", next_iter, config.num_workers
            )
            time.sleep(config.poll_interval)
            continue
        wait_elapsed = time.time() - wait_start

        logger.info("iter %d is ready (waited %.1fs) — starting training round", next_iter, wait_elapsed)

        replay_dirs = _collect_replay_dirs(
            config.data_dir, next_iter, config.replay_buffer_batches
        )
        dataset = SelfPlayDataset(replay_dirs)
        if len(dataset) == 0:
            logger.warning("Dataset is empty for iter %d — skipping", next_iter)
            next_iter += 1
            continue

        loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=(config.device != "cpu"),
        )

        train_start = time.time()
        for epoch in range(1, config.train_epochs + 1):
            policy_loss, value_loss = _train_one_epoch(net, device, loader, optimizer, augment_fn)
            logger.info(
                "iter %d | epoch %d/%d | policy_loss=%.4f  value_loss=%.4f",
                next_iter, epoch, config.train_epochs, policy_loss, value_loss,
            )
        train_elapsed = time.time() - train_start

        # Save per-iter checkpoint and its .DONE marker for generate.py workers.
        save_start = time.time()
        iter_ckpt = iter_checkpoint_path(config.save_path, next_iter)
        iter_done = iter_checkpoint_done_path(config.save_path, next_iter)
        torch.save(net.state_dict(), iter_ckpt)
        # Also overwrite the generic save_path so other tools can find "latest".
        torch.save(net.state_dict(), config.save_path)
        open(iter_done, "w").close()
        save_elapsed = time.time() - save_start

        iter_elapsed = time.time() - iter_start
        logger.info(
            "iter %d done | total=%.1fs  wait=%.1fs  train=%.1fs  save=%.1fs | "
            "samples=%d  policy_loss=%.4f  value_loss=%.4f  →  %s",
            next_iter, iter_elapsed, wait_elapsed, train_elapsed, save_elapsed,
            len(dataset), policy_loss, value_loss, iter_ckpt,
        )

        if scheduler is not None:
            scheduler.step()

        next_iter += 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> TrainingConfig:
    """Parse command-line arguments and return a TrainingConfig."""
    parser = argparse.ArgumentParser(
        description="Standalone training loop: watches data_dir and trains on new batches."
    )
    parser.add_argument("--game", default="othello",
                        choices=["tictactoe", "othello", "gomoku", "connect4"])

    _argv_game = "othello"
    for idx, token in enumerate(sys.argv[:-1]):
        if token in ("--game", "-game"):
            _argv_game = sys.argv[idx + 1]
            break
    _add_game_specific_args(parser, _argv_game)

    parser.add_argument("--save-path", default="checkpoint.pt",
                        help="Checkpoint path to save/load (.pt)")
    parser.add_argument("--data-dir", default="self_play_data",
                        help="Root directory written by generate.py (contains iter_*/ subdirs)")
    parser.add_argument("--num-workers", type=int, default=1,
                        help="Number of parallel generate.py workers per iter (default: 1)")
    parser.add_argument("--replay-buffer", type=int, default=5,
                        help="Number of most-recent completed iters to train on (default: 5)")
    parser.add_argument("--poll-interval", type=float, default=10.0,
                        help="Seconds to sleep while waiting for workers to finish (default: 10.0)")
    parser.add_argument("--train-epochs", type=int, default=1,
                        help="Training epochs per round (default: 1)")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Mini-batch size (default: 256)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (default: 1e-3)")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="L2 regularisation (default: 1e-4)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"],
                        help="Training device (default: cpu)")
    parser.add_argument("--filters", type=int, default=128,
                        help="ResNet filter count; must match checkpoint (default: 128)")
    parser.add_argument("--blocks", type=int, default=10,
                        help="Number of residual blocks; must match checkpoint (default: 10)")
    parser.add_argument("--value-hidden", type=int, default=256,
                        help="Value head hidden size; must match checkpoint (default: 256)")
    parser.add_argument("--lr-schedule", default="none", choices=["none", "step", "cosine"],
                        help="LR schedule (default: none)")
    parser.add_argument("--lr-decay-step", type=int, default=10,
                        help="Step schedule: decay every N training rounds (default: 10)")
    parser.add_argument("--lr-decay-gamma", type=float, default=0.5,
                        help="Step schedule: multiplicative decay factor (default: 0.5)")
    parser.add_argument("--lr-min", type=float, default=1e-5,
                        help="Cosine schedule: minimum LR (default: 1e-5)")
    parser.add_argument("--augment", action="store_true", default=False,
                        help="Enable symmetry-based data augmentation (game-dependent)")

    args = parser.parse_args()

    return TrainingConfig(
        game_name=args.game,
        save_path=args.save_path,
        data_dir=args.data_dir,
        num_workers=args.num_workers,
        replay_buffer_batches=args.replay_buffer,
        poll_interval=args.poll_interval,
        train_epochs=args.train_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
        num_filters=args.filters,
        num_residual_blocks=args.blocks,
        value_head_hidden_size=args.value_hidden,
        lr_schedule=args.lr_schedule,
        lr_decay_step=args.lr_decay_step,
        lr_decay_gamma=args.lr_decay_gamma,
        lr_min=args.lr_min,
        augment=args.augment,
        game_namespace=args,
    )


if __name__ == "__main__":
    config = _parse_args()
    try:
        train(config)
    except KeyboardInterrupt:
        logger.info("Training interrupted, exiting.")
