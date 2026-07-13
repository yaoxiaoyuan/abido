"""generate.py – Single-process self-play data generation script (continuous mode).

Continuously generates self-play data.  After each iteration's games a ``.npz``
file and an empty per-worker ``DONE`` marker are written under
``{data_dir}/iter_{N}/``.  Every ``--reload-interval`` games the script checks
whether ``checkpoint_iter{M}.pt.DONE`` exists (written by train.py after saving
iter-M weights); if found, it hot-loads ``checkpoint_iter{M}.pt`` and advances
the expected reload index.  The DONE file is NOT deleted.

Differences from train_pipeline_cpu.py
---------------------------------------
- Single main process only – no multiprocessing workers, safe to use GPU
- Device is freely configurable via ``--device`` (not forced to CPU)
- Has its own ``GenerationConfig``, independent of ``TrainConfig``
- Outer infinite loop runs until manually interrupted (Ctrl-C)
- Checkpoint reload keyed on per-iter DONE files, not mtime

Usage
-----
    # Continuous generation, check for new model every 4 games
    python generate.py --game connect4 --save-path bakc4/checkpoint.pt --reload-interval 4

    # GPU inference, 8 games per iter, worker-id=1
    python generate.py --game connect4 --save-path bakc4/checkpoint.pt \\
        --device cuda --games-per-iter 8 --worker-id 1 --reload-interval 8
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import sys
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from mcts import MCTSConfig
from model import PolicyValueNet, ResNetConfig
from ai_player import AIPlayer, AIPlayerConfig
from train_utils import (
    make_game,
    _add_game_specific_args,
    _play_one_game,
    iter_data_dir,
    iter_checkpoint_path,
    iter_checkpoint_done_path,
    worker_npz_path,
    worker_done_path,
)

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
class GenerationConfig:
    """Configuration for continuous self-play data generation.

    Attributes
    ----------
    game_name:
        Game name; one of othello / gomoku / connect4 / tictactoe.
    save_path:
        Path to the model weights file (.pt). Random weights are used if the
        file does not exist.
    data_dir:
        Root output directory; each iter is written to
        ``{data_dir}/iter_{iter_id}/``.
    worker_id:
        Worker index for this process; used to distinguish output filenames
        when multiple instances run in parallel.
    games_per_iter:
        Number of games to play per iteration.
    reload_interval:
        Check for a new checkpoint .DONE marker every this many games.
    device:
        Torch device string, e.g. ``"cpu"``, ``"cuda"``, or ``"mps"``.
    mcts:
        MCTS configuration (num_simulations, c_puct, Dirichlet noise, etc.).
    ai_player:
        AIPlayer configuration (temperature, greedy, batch_size, etc.).
    temperature:
        Temperature used for moves before ``temperature_threshold`` is reached.
    temperature_threshold:
        Moves before this count use temperature=1.0 (exploratory); after use
        greedy (temperature=0.0).
    num_input_planes:
        Number of input feature planes; must match the checkpoint.
    num_filters:
        ResNet conv filter count; must match the checkpoint.
    num_residual_blocks:
        Number of residual blocks; must match the checkpoint.
    value_head_hidden_size:
        Hidden layer size in the value head; must match the checkpoint.
    game_namespace:
        argparse.Namespace forwarded to the game's Args constructor.
    """
    game_name: str = "othello"
    save_path: str = "checkpoint.pt"
    data_dir: str = "self_play_data"
    worker_id: int = 0
    games_per_iter: int = 4
    reload_interval: int = 4
    device: str = "cpu"
    ai_player: AIPlayerConfig = field(default_factory=AIPlayerConfig)
    temperature: float = 1.0
    temperature_threshold: int = 30
    num_input_planes: int = 3
    num_filters: int = 128
    num_residual_blocks: int = 10
    value_head_hidden_size: int = 256
    game_namespace: argparse.Namespace = field(default_factory=argparse.Namespace)


# ---------------------------------------------------------------------------
# Model loader helper
# ---------------------------------------------------------------------------

def _load_weights(net: PolicyValueNet, save_path: str) -> float:
    """Load weights from save_path into net and return the file's mtime.

    Returns 0.0 if the file does not exist.
    """
    if not os.path.exists(save_path):
        return 0.0
    state_dict = torch.load(save_path, map_location="cpu")
    net.load_state_dict(state_dict)
    return os.path.getmtime(save_path)


# ---------------------------------------------------------------------------
# Core: continuous single-process generation
# ---------------------------------------------------------------------------

def generate(config: GenerationConfig) -> None:
    """Continuously generate self-play data in the main process until interrupted.

    Each iteration plays ``config.games_per_iter`` games, saves them as a
    single ``.npz`` file, and writes an empty worker ``DONE`` marker.
    Every ``config.reload_interval`` games the script checks whether
    ``iter_checkpoint_done_path(save_path, next_reload_iter)`` exists (written
    by train.py); if found, it hot-loads the corresponding checkpoint and
    advances ``next_reload_iter``.  The DONE file is NOT deleted.

    Output layout:
        {data_dir}/iter_{iter_id}/worker{worker_id}.npz   (worker_npz_path)
        {data_dir}/iter_{iter_id}/worker{worker_id}.DONE  (worker_done_path)

    Checkpoint reload protocol:
        train.py writes  {stem}_iter{N}.DONE  after saving iter-N weights.
        generate.py detects that file, loads {stem}_iter{N}.pt, then
        increments next_reload_iter to wait for iter N+1.
    """
    os.makedirs(config.data_dir, exist_ok=True)

    # Build game and network
    game = make_game(config.game_name, config.game_namespace)
    net_config = ResNetConfig(
        board_height=game.height,
        board_width=game.width,
        num_input_planes=config.num_input_planes,
        num_actions=game.num_actions,
        num_filters=config.num_filters,
        num_residual_blocks=config.num_residual_blocks,
    )
    net = PolicyValueNet(net_config)

    # Load the highest existing iter checkpoint, or fall back to save_path.
    save_stem, save_ext = os.path.splitext(config.save_path)
    existing = sorted(glob.glob(f"{save_stem}_iter*{save_ext}"))
    if existing:
        initial_ckpt = existing[-1]
        _load_weights(net, initial_ckpt)
        logger.info("Loaded initial weights from latest iter checkpoint '%s'", initial_ckpt)
    elif _load_weights(net, config.save_path) > 0:
        logger.info("Loaded initial weights from '%s'", config.save_path)
    else:
        logger.info("No checkpoint found, using random weights")

    net.eval()

    ai = AIPlayer(game, net, config.ai_player)

    logger.info(
        "Worker %d starting continuous generation on device '%s' "
        "(games_per_iter=%d, reload_interval=%d)",
        config.worker_id, config.device, config.games_per_iter, config.reload_interval,
    )

    iter_id = 0
    games_since_last_reload_check = 0
    # The next iter checkpoint index we expect train.py to produce.
    next_reload_iter = 1

    while True:
        iter_id += 1
        iter_start = time.time()
        current_iter_dir = iter_data_dir(config.data_dir, iter_id)
        os.makedirs(current_iter_dir, exist_ok=True)

        all_boards: list[np.ndarray] = []
        all_policies: list[np.ndarray] = []
        all_values: list[float] = []
        all_q_values: list[float] = []
        all_final_boards: list[np.ndarray] = []
        game_times: list[float] = []

        for game_idx in range(config.games_per_iter):
            # Every reload_interval games, check if the next iter checkpoint is ready.
            if games_since_last_reload_check >= config.reload_interval:
                games_since_last_reload_check = 0
                done_path = iter_checkpoint_done_path(config.save_path, next_reload_iter)
                if os.path.exists(done_path):
                    ckpt_path = iter_checkpoint_path(config.save_path, next_reload_iter)
                    net.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
                    net.eval()
                    logger.info(
                        "Worker %d: reloaded weights from '%s'",
                        config.worker_id, ckpt_path,
                    )
                    next_reload_iter += 1

            game_start = time.time()
            game.reset()
            boards, policies, values, q_values, final_board = _play_one_game(
                game=game,
                ai=ai,
                temperature_threshold=config.temperature_threshold,
                temperature=config.temperature
            )

            game_elapsed = time.time() - game_start

            all_boards.extend(boards)
            all_policies.extend(policies)
            all_values.extend(values)
            all_q_values.extend(q_values)
            all_final_boards = all_final_boards + [final_board] * len(boards)

            game_times.append(game_elapsed)
            games_since_last_reload_check += 1

            logger.info(
                "Worker %d | iter %d | game %d/%d done — %d moves, %.1fs, total samples: %d",
                config.worker_id, iter_id, game_idx + 1, config.games_per_iter,
                len(boards), game_elapsed, len(all_values),
            )

        # Save this iter's data to disk
        npz_path = worker_npz_path(config.data_dir, iter_id, config.worker_id)
        np.savez_compressed(
            npz_path,
            boards=np.array(all_boards, dtype=np.float32),
            policies=np.array(all_policies, dtype=np.float32),
            values=np.array(all_values, dtype=np.float32),
            q_values=np.array(all_q_values, dtype=np.float32),
            final_boards=np.array(all_final_boards, dtype=np.float32),
        )

        # Write per-worker DONE marker
        done_path = worker_done_path(config.data_dir, iter_id, config.worker_id)
        open(done_path, "w").close()

        iter_elapsed = time.time() - iter_start
        avg_game_time = sum(game_times) / len(game_times) if game_times else 0.0
        logger.info(
            "Worker %d | iter %d done | total=%.1fs  games=%d  avg_game=%.1fs  samples=%d  → %s",
            config.worker_id, iter_id, iter_elapsed,
            config.games_per_iter, avg_game_time, len(all_values), npz_path,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> GenerationConfig:
    """Parse command-line arguments and return a GenerationConfig."""
    parser = argparse.ArgumentParser(
        description="Single-process continuous self-play generation; writes a DONE marker after each batch."
    )
    parser.add_argument("--game", default="othello",
                        choices=["tictactoe", "othello", "gomoku", "connect4"])

    # Pre-read --game so that game-specific args are registered before parse_args().
    _argv_game = "othello"
    for idx, token in enumerate(sys.argv[:-1]):
        if token in ("--game", "-game"):
            _argv_game = sys.argv[idx + 1]
            break
    _add_game_specific_args(parser, _argv_game)

    parser.add_argument("--save-path", default="checkpoint.pt",
                        help="Model weights path (.pt); random weights used if missing")
    parser.add_argument("--data-dir", default="self_play_data",
                        help="Root output directory; each iter is written to {data-dir}/iter_{id}/")
    parser.add_argument("--worker-id", type=int, default=0,
                        help="Worker index used in output filenames (default: 0)")
    parser.add_argument("--games-per-iter", type=int, default=4,
                        help="Number of games to play per iteration (default: 4)")
    parser.add_argument("--reload-interval", type=int, default=4,
                        help="Check for a new checkpoint .DONE marker every this many games (default: 4)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"],
                        help="Inference device (default: cpu)")
    parser.add_argument("--sims", type=int, default=400,
                        help="MCTS simulations per move (default: 400)")
    parser.add_argument("--c-puct", type=float, default=1.5,
                        help="PUCT exploration constant (default: 1.5)")
    parser.add_argument("--dirichlet-alpha", type=float, default=0.3,
                        help="Dirichlet noise alpha; set 0.0 to disable (default: 0.3)")
    parser.add_argument("--dirichlet-epsilon", type=float, default=0.25,
                        help="Dirichlet noise weight (default: 0.25)")
    parser.add_argument("--mcts-batch", type=int, default=4,
                        help="MCTS inference batch size (default: 4)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature for moves before the temperature threshold (default: 1.0)")
    parser.add_argument("--temp-threshold", type=int, default=30,
                        help="Moves before this use temperature=1.0; after use greedy (default: 30)")
    parser.add_argument("--playout-cap-random", type=float, default=0.0,
                        help="Probability of using reduced playout cap (KataGo-style); 0.0 disables (default: 0.0)")
    parser.add_argument("--playout-cap-ratio", type=float, default=0.5,
                        help="Fraction of sims when playout cap triggers (default: 0.5)")
    parser.add_argument("--fpu", type=float, default=0.0,
                        help="First Play Urgency value; meaning depends on --fpu-strategy: 'reduction' subtracts from parent Q, 'fixed' uses as flat Q for unvisited nodes (default: 0.0)")
    parser.add_argument("--fpu-strategy", default="fixed", choices=["reduction", "fixed"],
                        help="FPU strategy: 'reduction' inherits parent Q, 'fixed' uses flat Q=fpu (default: reduction)")
    parser.add_argument("--c-puct-strategy", default="fixed", choices=["fixed", "dynamic"],
                        help="PUCT strategy: 'fixed' constant c_puct, 'dynamic' grows with log((N+c_base)/c_base) (default: fixed)")
    parser.add_argument("--c-base", type=float, default=19652.0,
                        help="Base constant for dynamic PUCT formula (default: 19652.0)")
    parser.add_argument("--filters", type=int, default=128,
                        help="ResNet filter count; must match checkpoint (default: 128)")
    parser.add_argument("--blocks", type=int, default=10,
                        help="Number of residual blocks; must match checkpoint (default: 10)")
    parser.add_argument("--value-hidden", type=int, default=256,
                        help="Value head hidden size; must match checkpoint (default: 256)")

    args = parser.parse_args()

    mcts_config = MCTSConfig(
        num_simulations=args.sims,
        c_puct=args.c_puct,
        dirichlet_alpha=args.dirichlet_alpha,
        dirichlet_epsilon=args.dirichlet_epsilon,
        device=args.device,
        playout_cap_random=args.playout_cap_random,
        playout_cap_ratio=args.playout_cap_ratio,
        fpu_value=args.fpu,
        fpu_strategy=args.fpu_strategy,
        c_puct_strategy=args.c_puct_strategy,
        c_base=args.c_base
    )

    ai_config = AIPlayerConfig(
        mcts=mcts_config,
        batch_size=args.mcts_batch,
    )

    return GenerationConfig(
        game_name=args.game,
        save_path=args.save_path,
        data_dir=args.data_dir,
        worker_id=args.worker_id,
        games_per_iter=args.games_per_iter,
        reload_interval=args.reload_interval,
        device=args.device,
        ai_player=ai_config,
        temperature=args.temperature,
        temperature_threshold=args.temp_threshold,
        num_filters=args.filters,
        num_residual_blocks=args.blocks,
        value_head_hidden_size=args.value_hidden,
        game_namespace=args,
    )


if __name__ == "__main__":
    config = _parse_args()
    try:
        generate(config)
    except KeyboardInterrupt:
        logger.info("Worker %d: interrupted, exiting.", config.worker_id)
