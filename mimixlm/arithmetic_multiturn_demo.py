"""
arithmetic_multiturn_demo.py
----------------------------
Multi-turn arithmetic conversation demo using TrainPipeline from train.py.

Vocabulary: all printable ASCII characters + common word tokens.
Tokenizer:  built directly from the vocab list (no corpus training needed).
Data:       multi-turn arithmetic conversations, generated and cached to disk.
Training:   decoder-only LM via TrainPipeline.build_model + run_training.

Usage:
    python arithmetic_multiturn_demo.py [--ops + -] [--num-range 0 49]
                                        [--turns 3] [--output model/arith_mt]
"""

import argparse
import os
import random

import torch

from train_pipeline import TrainPipeline, TrainConfig, make_dataloader
from transformer_minimal import ModelConfig, Transformer
from tokenizer_minimal import Tokenizer
from chat_cli import ChatFormatter

# ─────────────────────────────────────────────────────────────────────────────
# Vocabulary
# ─────────────────────────────────────────────────────────────────────────────

# All printable ASCII characters (codes 32–126)
_ASCII_PRINTABLE = [chr(code) for code in range(32, 127)]

VOCAB: list[str] = _ASCII_PRINTABLE + [
    "What", "Then", "And",
    "what", "then", "and",
    " is", 
    " plus", " minus", " times",
]

# Role tags used by ChatFormatter — registered as atomic special tokens
ROLE_TAGS = ["<|user|>\n", "<|assistant|>\n", "<|end|>\n"]

# ─────────────────────────────────────────────────────────────────────────────
# Arithmetic helpers
# ─────────────────────────────────────────────────────────────────────────────

def _evaluate(a: int, op: str, b: int) -> int:
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    raise ValueError(f"Unknown operator: {op}")


def _build_train_test_split(
    num_range: tuple[int, int],
    ops: list[str],
    test_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[list[tuple[int, str, int]], list[tuple[int, str, int]]]:
    """Enumerate all (a, op, b) triples and split into disjoint train/test sets."""
    lo, hi = num_range
    all_triples = [
        (a, op, b)
        for op in ops
        for a in range(lo, hi + 1)
        for b in range(lo, hi + 1)
    ]
    rng = random.Random(seed)
    rng.shuffle(all_triples)
    split = max(1, int(len(all_triples) * test_fraction))
    return all_triples[split:], all_triples[:split]


def _build_conversations(
    triples: list[tuple[int, str, int]],
    num_range: tuple[int, int],
    turns_per_conv: int = 3,
    seed: int = 7,
) -> list[list[tuple[str, str]]]:
    """
    Build multi-turn arithmetic conversations from a list of (a, op, b) triples.

    Turn 1 : "What is {a} {op_word} {b}?"  →  answer
    Turn 2+: alternates between two styles:
        - "And {c} {op_word} {d}?"  — independent new expression
        - "Then {op_word} {c}?"     — continues from previous answer (chain)
    """
    op_word_map = {"+": "plus", "-": "minus", "*": "times"}
    lo, hi = num_range

    rng = random.Random(seed)
    rng.shuffle(triples)

    conversations: list[list[tuple[str, str]]] = []
    for first_triple in triples:
        a0, op0, b0 = first_triple

        for op_word0 in [op0, op_word_map[op0]]:
            turns: list[tuple[str, str]] = []
            prev_answer = _evaluate(a0, op0, b0)
            question = f"What is {a0} {op_word0} {b0}?"
            answer = str(prev_answer)
            if rng.random() > 0.5:
                question = question.lower()
            turns.append((question, str(answer)))

            for _ in range(turns_per_conv - 1):
                op      = rng.choice(triples)[1]
                op_word = op_word_map[op] if rng.random() > 0.5 else op

                if rng.random() < 0.5:
                    # Chain: build on previous answer
                    c        = rng.randint(lo, hi)
                    answer   = _evaluate(prev_answer, op, c)
                    question = f"Then {op_word} {c}?"
                else:
                    # Independent new expression
                    c        = rng.randint(lo, hi)
                    d        = rng.randint(lo, hi)
                    answer   = _evaluate(c, op, d)
                    question = f"And {c} {op_word} {d}?"

                prev_answer = answer
                if rng.random() > 0.5:
                    question = question.lower()

                turns.append((question, str(answer)))

            conversations.append(turns)

    return conversations


# ─────────────────────────────────────────────────────────────────────────────
# Sample builder  (mirrors _multi_turn_sample in test_transformer_minimal.py)
# ─────────────────────────────────────────────────────────────────────────────

def _build_sample(
    turns: list[tuple[str, str]],
    tokenizer: Tokenizer,
) -> dict:
    """
    Encode a multi-turn conversation into a training sample dict.

    Uses ChatFormatter.format_single_turn to render each user turn without
    re-rendering prior history (avoiding token duplication).

    Returns:
        {"tgt_in": LongTensor, "tgt_out": LongTensor, "loss_mask": LongTensor}
        loss_mask = 1 for assistant tokens, 0 for user/role-tag tokens.
    """
    formatter = ChatFormatter()

    all_ids:   list[int]  = []
    loss_mask: list[int]  = []
    first_turn = True

    for user_text, asst_text in turns:
        user_turn_text, assistant_start_text = formatter.format_single_turn(user_text)
        full_assistant_text = formatter.assistant_template.format(assistant=asst_text)
        assistant_body = full_assistant_text[len(assistant_start_text):]

        user_ids = tokenizer.encode(
            user_turn_text + assistant_start_text,
            add_bos=first_turn,
            add_eos=False,
        )
        asst_ids = tokenizer.encode(assistant_body, add_bos=False, add_eos=False)

        all_ids   += user_ids + asst_ids
        loss_mask += [0] * len(user_ids) + [1] * len(asst_ids)
        first_turn = False

    # Print a few sample conversations for sanity check
    '''
    for turn_idx, (user_text, asst_text) in enumerate(turns):
        print(f"    Turn {turn_idx + 1}  U: {user_text!r}")
        print(f"           A: {asst_text!r}")
        print(tokenizer._pretok_regex.findall(user_text))
        print(tokenizer._pretok_regex.findall(asst_text))
    print(all_ids)
    print([tokenizer.id_to_token(x) for x in all_ids])
    input()
    '''

    full_ids = torch.tensor(all_ids, dtype=torch.long)
    return {
        "tgt_in":    full_ids[:-1],
        "tgt_out":   full_ids[1:],
        "loss_mask": torch.tensor(loss_mask[1:], dtype=torch.long)
    }


# ─────────────────────────────────────────────────────────────────────────────
# Data generation with caching
# ─────────────────────────────────────────────────────────────────────────────

def build_or_load_samples(
    data_dir: str,
    tokenizer: Tokenizer,
    num_range: tuple[int, int],
    ops: list[str],
    turns_per_conv: int,
    test_fraction: float = 0.1,
) -> tuple[list[dict], list[dict]]:
    """Generate multi-turn samples and cache to disk; reload on subsequent runs."""
    os.makedirs(data_dir, exist_ok=True)
    train_cache = os.path.join(data_dir, "train.pt")
    test_cache  = os.path.join(data_dir, "test.pt")

    if os.path.exists(train_cache) and os.path.exists(test_cache):
        print("[data] Loading cached samples …")
        train_samples = torch.load(train_cache, weights_only=True)
        test_samples  = torch.load(test_cache,  weights_only=True)
        print(
            f"      train: {len(train_samples):,}  "
            f"test: {len(test_samples):,}  (cached)"
        )
        return train_samples, test_samples

    print("[data] Generating conversations …")
    train_triples, test_triples = _build_train_test_split(
        num_range, ops, test_fraction=test_fraction
    )
    print(f"      triples — train: {len(train_triples):,}  test: {len(test_triples):,}")

    train_convs = _build_conversations(
        train_triples, num_range=num_range, turns_per_conv=turns_per_conv, seed=7
    )
    test_convs  = _build_conversations(
        test_triples,  num_range=num_range, turns_per_conv=turns_per_conv, seed=99
    )
    print(
        f"      conversations — train: {len(train_convs):,}  "
        f"test: {len(test_convs):,}  ({turns_per_conv} turns each)"
    )

    train_samples = [_build_sample(conv, tokenizer) for conv in train_convs]
    test_samples  = [_build_sample(conv, tokenizer) for conv in test_convs]

    torch.save(train_samples, train_cache)
    torch.save(test_samples,  test_cache)
    print(f"      Cached → {train_cache}, {test_cache}")

    return train_samples, test_samples


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-turn arithmetic LM demo")
    parser.add_argument("--ops",       nargs="+", default=["+", "-"],
                        choices=["+", "-", "*"])
    parser.add_argument("--num-range", nargs=2, type=int, default=[0, 100],
                        metavar=("LO", "HI"))
    parser.add_argument("--turns",     type=int, default=3,
                        help="Turns per conversation (default: 3)")
    parser.add_argument("--output",    default="model/arith_multiturn",
                        help="Directory to save model checkpoints")
    parser.add_argument("--device",    default="cpu")
    parser.add_argument("--epochs",    type=int, default=30)
    parser.add_argument("--batch-size",type=int, default=256)
    parser.add_argument("--lr",        type=float, default=1e-2)
    args = parser.parse_args()

    random.seed(42)
    torch.manual_seed(42)

    num_range: tuple[int, int] = (args.num_range[0], args.num_range[1])

    # ── Tokenizer (no training needed) ───────────────────────────────────────
    tokenizer = Tokenizer(VOCAB, eos_token="<|end|>\n")
    for tag in ROLE_TAGS:
        tokenizer.add_special_token(tag)
    tokenizer.save(args.output)
    print(f"[tokenizer] vocab_size={tokenizer.vocab_size}")

    # ── Data (generate + cache, or load from cache) ───────────────────────────
    data_dir = args.output.replace("model/", "data/", 1)
    train_samples, test_samples = build_or_load_samples(
        data_dir      = data_dir,
        tokenizer     = tokenizer,
        num_range     = num_range,
        ops           = args.ops,
        turns_per_conv= args.turns,
    )

    # ── Model via TrainPipeline ───────────────────────────────────────────────
    pipeline = TrainPipeline(output_dir=args.output)
    model, cfg = pipeline.build_model(
        tokenizer,
        d_model    = 128,
        num_heads  = 4,
        num_layers = 4,
        d_ff       = 256,
        max_len    = 256,
        dropout    = 0.03,
        pos_encoding = "rope",
        ffn_type     = "glu",
        activation   = "silu",
        norm_layer   = "rmsnorm",
        norm_type    = "pre",
        is_chat      = True
    )

    # ── Training via TrainPipeline ────────────────────────────────────────────
    pipeline.run_training(
        model,
        train_samples,
        test_samples,
        tokenizer        = tokenizer,
        device           = args.device,
        batch_size       = args.batch_size,
        lr               = args.lr,
        num_epochs       = args.epochs,
        warmup_steps     = 200,
        mixed_precision  = "no",
        save_every_steps = 500,
        max_checkpoints  = 3,
        eval_every_steps = 200,
        log_interval     = 50,
    )

    print(f"\nAll artefacts saved to: {args.output}/")


if __name__ == "__main__":
    main()