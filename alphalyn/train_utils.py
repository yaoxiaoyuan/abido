"""train_utils.py – Shared utility functions used by train.py and generate.py.

Contains:
- ``make_game``               : instantiate a BoardGame by name
- ``_add_game_specific_args`` : register game-specific CLI arguments
- ``_play_one_game``          : run one complete self-play game and collect training samples
- Naming-convention helpers   : canonical paths for iter dirs, worker files, checkpoints
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from board_game import BoardGame
from model import PolicyValueNet, encode_board
from ai_player import AIPlayer


# ---------------------------------------------------------------------------
# Naming-convention helpers
# ---------------------------------------------------------------------------
# All canonical path construction is centralised here so that train.py and
# generate.py never hard-code directory/file name patterns independently.

def iter_data_dir(data_dir: str, iter_id: int) -> str:
    """Return the data directory for a given iteration: ``{data_dir}/iter_{N}/``."""
    return os.path.join(data_dir, f"iter_{iter_id}")


def worker_npz_path(data_dir: str, iter_id: int, worker_id: int) -> str:
    """Return the .npz file path for a specific worker and iteration."""
    return os.path.join(iter_data_dir(data_dir, iter_id), f"worker{worker_id}.npz")


def worker_done_path(data_dir: str, iter_id: int, worker_id: int) -> str:
    """Return the per-worker DONE marker path for a specific worker and iteration."""
    return os.path.join(iter_data_dir(data_dir, iter_id), f"worker{worker_id}.DONE")


def iter_checkpoint_path(save_path: str, iter_id: int) -> str:
    """Return the per-iter checkpoint path.

    Example: ``save_path="ckpt/c4.pt"`` → ``"ckpt/c4_iter3.pt"``
    """
    stem, ext = os.path.splitext(save_path)
    return f"{stem}_iter{iter_id}{ext}"


def iter_checkpoint_done_path(save_path: str, iter_id: int) -> str:
    """Return the per-iter checkpoint DONE marker path.

    Format: ``{stem}_iter{N}.DONE``  (no ``.pt`` suffix on the DONE file).
    Example: ``save_path="ckpt/c4.pt"`` → ``"ckpt/c4_iter3.DONE"``
    """
    stem, _ext = os.path.splitext(save_path)
    return f"{stem}_iter{iter_id}.DONE"


# ---------------------------------------------------------------------------
# Game factory
# ---------------------------------------------------------------------------

def make_game(game_name: str, namespace: argparse.Namespace) -> BoardGame:
    """Instantiate a ``BoardGame`` by name, forwarding *namespace* to the Args constructor."""
    if game_name == "othello":
        from game_othello import OthelloGame, OthelloArgs
        return OthelloGame(OthelloArgs(namespace))
    if game_name == "gomoku":
        from game_gomoku import GomokuGame, GomokuArgs
        return GomokuGame(GomokuArgs(namespace))
    if game_name == "connect4":
        from game_connect4 import Connect4Game, Connect4Args
        return Connect4Game(Connect4Args(namespace))
    if game_name == "tictactoe":
        from game_tictactoe import TicTacToeGame, TicTacToeArgs
        return TicTacToeGame(TicTacToeArgs(namespace))
    if game_name == "hex":
        from game_hex import HexGame, HexArgs
        return HexGame(HexArgs(namespace))
    raise ValueError(f"Unknown game: {game_name!r}. Choose from othello, gomoku, connect4, tictactoe, hex.")


def _add_game_specific_args(parser: argparse.ArgumentParser, game_name: str) -> None:
    """Register game-specific arguments for *game_name* on *parser*.

    Each game's ``XxxArgs.add_common_args`` already calls the base
    ``GameArgs.add_common_args``; here we only need the game-specific extras
    (e.g. ``--gomoku-size`` for Gomoku).
    """
    if game_name == "gomoku":
        from game_gomoku import GomokuArgs
        GomokuArgs.add_common_args(parser)
    # Other games currently have no extra args beyond the common set.


# ---------------------------------------------------------------------------
# Single-game self-play
# ---------------------------------------------------------------------------

def _play_one_game(
    game: BoardGame,
    ai: AIPlayer,
    temperature_threshold: int,
    temperature: float = 1.0,
    second_ai: Optional[AIPlayer] = None,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[float]]:
    """Play one complete self-play game and collect training samples.

    Parameters
    ----------
    game:
        The board game instance to play on.
    ai:
        The AI player (with internal MCTS) that selects moves during self-play.
    temperature_threshold:
        Move count threshold for temperature annealing.  Moves before this
        threshold use ``temperature``; moves after use temperature 0.0
        (greedy / deterministic selection).
    temperature:
        Temperature used for moves before ``temperature_threshold`` is reached.
        Once ``move_count >= temperature_threshold``, temperature drops to 0.0
        (greedy selection).
    second_ai:
        Optional second AI player used for the opponent's turns.  When ``None``
        (default) the same ``ai`` plays both sides.
    """
    ai.reset()
    if second_ai:
        second_ai.reset()

    move_history: List[Tuple[np.ndarray, np.ndarray, int, float]] = []
    # Each entry: (board_encoding, policy_vector, player_who_moved, q_values)

    move_count = 0
    num_actions = game.num_actions

    while not game.state.is_game_over:
        current_player = game.state.turn
        
        current_ai = ai
        if second_ai is not None and current_player == game.PLAYER_SECOND:
            current_ai = second_ai

        # Use exploratory temperature for early moves, greedy afterwards.
        if move_count > temperature_threshold:
            temperature = 0.0

        action_probs = current_ai._mcts.get_action_probs(
            temperature=temperature,
            batch_size=current_ai.config.batch_size,
        )

        # Build full policy vector (zeros for illegal actions)
        policy_vector = np.zeros(num_actions, dtype=np.float32)
        for action, prob in action_probs:
            idx = game.action_to_index(action)
            policy_vector[idx] = prob

        # Encode board before the move
        board_encoding = encode_board(
            game.state.board,
            current_player,
        ).squeeze(0).numpy()  # (num_input_planes, H, W)

        # Select and apply action
        action = current_ai._mcts.select_action(action_probs, greedy=(temperature == 0.0))
        q_value = current_ai._mcts._root.mean_value
        game.move(current_player, action)

        # AI observe the action to keep tree in sync
        ai.observe_action(action)
        if second_ai:
            second_ai.observe_action(action)

        move_history.append((board_encoding, policy_vector, current_player, q_value))
        move_count += 1

    # Determine outcome for value targets
    winner = game.state.winner

    boards: List[np.ndarray] = []
    policies: List[np.ndarray] = []
    values: List[float] = []
    q_values: List[float] = []
    final_board = game.state.board.copy()
    for board_encoding, policy_vector, player_who_moved, q_value in move_history:
        if winner == game.RESULT_TIE:
            value = 0.0
        elif winner == player_who_moved:
            value = 1.0
        else:
            value = -1.0
        boards.append(board_encoding)
        policies.append(policy_vector)

        q_values.append(q_value)
        values.append(value)

    return boards, policies, values, q_values, final_board
