"""
shakespeare_demo.py
-------------------
Shakespeare LM demo built on top of PretrainPipeline from pretrain.py.

Usage:
    python shakespeare_demo.py [--device cpu]
"""

import argparse
import random

import torch

from train_pipeline import TrainPipeline

# ── Constants ────────────────────────────────────────────────────────────────

DATA_URL       = "https://cs.stanford.edu/people/karpathy/char-rnn/shakespeare_input.txt"
MODEL_DIR      = "model/shakespeare"
URL_CACHE_PATH = "data/shakespeare/shakespeare_input.txt"

# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Shakespeare LM demo")
    parser.add_argument("--device", default="cpu",
        help="torch device string (default: cpu)")
    args = parser.parse_args()

    random.seed(42)
    torch.manual_seed(42)

    pipeline = TrainPipeline(output_dir=MODEL_DIR)
    pipeline.run(
        data_path        = DATA_URL,
        url_cache_path   = URL_CACHE_PATH,
        device           = args.device,
        # tokenizer
        vocab_size       = 4096,
        min_frequency    = 20,
        max_ngram_len    = 10,
        # samples
        concat           = True,
        chunk_size       = 256,
        stride           = 128,
        test_ratio       = 0.05,
        # model architecture
        d_model          = 256,
        num_heads        = 4,
        num_layers       = 6,
        d_ff             = 512,
        dropout          = 0.05,
        num_kv_heads     = 2,
        pos_encoding     = "rope",
        ffn_type         = "glu",
        activation       = "silu",
        norm_layer       = "rmsnorm",
        norm_type        = "pre",
        qk_norm          = True,
        v_norm           = True,
        learnable_residual = True,
        tie_weights      = False,
        # training
        batch_size       = 64,
        lr               = 1e-2,
        num_epochs       = 10,
        warmup_steps     = 200,
        mixed_precision  = "no",
        save_every_steps = 500,
        max_checkpoints  = 3,
        eval_every_steps = 200,
        log_interval     = 10,
    )


if __name__ == "__main__":
    main()