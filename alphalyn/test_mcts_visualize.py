"""test_mcts_visualize.py – Visualise MCTS evaluation on a specific board position.

This script sets up a hand-crafted board position (default: Othello mid-game),
runs MCTS for a configurable number of simulations, then renders two outputs:

1. **mcts_tree.svg** – the search tree (via graphviz), showing visit counts,
   Q-values, and prior probabilities for the top nodes.
2. **mcts_policy.png** – a heatmap of the action probabilities produced by
   MCTS overlaid on the board grid (via matplotlib).

Usage
-----
    # Random-weight model (quick sanity check)
    python test_mcts_visualize.py

    # With a trained checkpoint
    python test_mcts_visualize.py --save-path models/othello/best.pt --sims 800

    # Gomoku on a different position
    python test_mcts_visualize.py --game gomoku --sims 200

Dependencies
------------
    pip install matplotlib graphviz
    # Plus the Graphviz system binaries: https://graphviz.org/download/
"""

import argparse
import sys
import os
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend – works in headless environments
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Make sure the alpha/ directory is on the path when invoked directly.
sys.path.insert(0, os.path.dirname(__file__))

from mcts import MCTS, MCTSConfig, visualize_tree
from model import PolicyValueNet, ResNetConfig
import torch


# ---------------------------------------------------------------------------
# Hand-crafted positions for each supported game
# ---------------------------------------------------------------------------

def _make_othello_position(
    board: Optional[np.ndarray] = None,
    turn: Optional[int] = None,
):
    """Return an OthelloGame set to *board* / *turn*, or a default mid-game position."""
    from game_othello import OthelloGame, OthelloArgs
    import argparse

    namespace = argparse.Namespace(
        gui=False, grid_h=80, grid_w=80, mode="hum",
        model_path="", model_path_first="", model_path_second="",
        num_simulations=400, ai_plays_first=False,
        num_filters=64, num_residual_blocks=5, value_head_hidden_size=128,
        device="cpu",
    )
    args = OthelloArgs(namespace)
    game = OthelloGame(args)
    game.reset()

    if board is not None:
        game.state.board = board.copy()
    else:
        # Default: mid-game position after ~12 moves with real choices for both sides.
        default_board = np.zeros((8, 8), dtype=int)
        for r, c in [(3, 3), (4, 4), (3, 5), (2, 4), (4, 2), (5, 3)]:
            default_board[r][c] = 1  # black
        for r, c in [(3, 4), (4, 3), (2, 3), (5, 4), (3, 2), (4, 5)]:
            default_board[r][c] = 2  # white
        game.state.board = default_board

    game.state.turn = game.PLAYER_FIRST if turn is None else turn
    return game


def _make_tictactoe_position(
    board: Optional[np.ndarray] = None,
    turn: Optional[int] = None,
):
    """Return a TicTacToeGame set to *board* / *turn*, or a default position."""
    from game_tictactoe import TicTacToeGame, TicTacToeArgs
    import argparse

    namespace = argparse.Namespace(
        gui=False, grid_h=80, grid_w=80, mode="hum",
        model_path="", model_path_first="", model_path_second="",
        num_simulations=400, ai_plays_first=False,
        num_filters=64, num_residual_blocks=5, value_head_hidden_size=128,
        device="cpu",
    )
    args = TicTacToeArgs(namespace)
    game = TicTacToeGame(args)
    game.reset()

    if board is not None:
        game.state.board = board.copy()
    else:
        # Default: X has centre + bottom-right; O has two corners; X to move.
        default_board = np.zeros((3, 3), dtype=int)
        default_board[1][1] = 1  # X centre
        default_board[0][0] = 2  # O top-left
        default_board[2][2] = 1  # X bottom-right
        default_board[0][2] = 2  # O top-right
        game.state.board = default_board

    game.state.turn = game.PLAYER_FIRST if turn is None else turn
    return game


def _make_gomoku_position(
    board: Optional[np.ndarray] = None,
    turn: Optional[int] = None,
):
    """Return a GomokuGame set to *board* / *turn*, or a default position."""
    from game_gomoku import GomokuGame, GomokuArgs
    import argparse

    namespace = argparse.Namespace(
        gui=False, grid_h=40, grid_w=40, mode="hum",
        model_path="", model_path_first="", model_path_second="",
        num_simulations=400, ai_plays_first=False,
        num_filters=64, num_residual_blocks=5, value_head_hidden_size=128,
        device="cpu", gomoku_size=9,
    )
    args = GomokuArgs(namespace)
    game = GomokuGame(args)
    game.reset()

    if board is not None:
        game.state.board = board.copy()
    else:
        # Default: black has a 3-in-a-row threat; white needs to respond.
        cx, cy = game.height // 2, game.width // 2
        for i in range(3):
            game.state.board[cx][cy + i] = 1  # black horizontal threat
        game.state.board[cx - 1][cy + 1] = 2  # white stone

    game.state.turn = game.PLAYER_SECOND if turn is None else turn
    return game


def _make_connect4_position(
    board: Optional[np.ndarray] = None,
    turn: Optional[int] = None,
):
    """Return a Connect4Game set to *board* / *turn*, or a default position."""
    from game_connect4 import Connect4Game, Connect4Args
    import argparse

    namespace = argparse.Namespace(
        gui=False, grid_h=80, grid_w=80, mode="hum",
        model_path="", model_path_first="", model_path_second="",
        num_simulations=400, ai_plays_first=False,
        num_filters=64, num_residual_blocks=5, value_head_hidden_size=128,
        device="cpu",
    )
    args = Connect4Args(namespace)
    game = Connect4Game(args)
    game.reset()

    if board is not None:
        game.state.board = board.copy()
    else:
        # Default: red has a 3-in-a-row at the bottom; yellow must block.
        h = game.height - 1
        for col in range(3):
            game.state.board[h][col] = 1  # red bottom row
        game.state.board[h][4] = 2        # yellow blocker

    game.state.turn = game.PLAYER_SECOND if turn is None else turn
    return game

GAME_BUILDERS = {
    "othello":   _make_othello_position,
    "tictactoe": _make_tictactoe_position,
    "gomoku":    _make_gomoku_position,
    "connect4":  _make_connect4_position,
}


# ---------------------------------------------------------------------------
# Policy heatmap renderer
# ---------------------------------------------------------------------------

def plot_policy_heatmap(
    game,
    action_probs,
    output_path: str = "mcts_policy.png",
    title: str = "MCTS Action Probabilities",
) -> None:
    """Draw the board with a probability heatmap and save to *output_path*.

    Each legal action is coloured by its MCTS probability.  The board pieces
    are drawn on top so the position context is clear.

    Parameters
    ----------
    game:
        A ``BoardGame`` instance with ``state.board`` already set.
    action_probs:
        List of ``(MoveAction, probability)`` pairs from ``MCTS.get_action_probs``.
    output_path:
        Destination ``.png`` file path.
    title:
        Figure title string.
    """
    height, width = game.height, game.width
    board = game.state.board

    # Build a 2-D probability grid (pass actions are excluded from the heatmap).
    prob_grid = np.zeros((height, width), dtype=float)
    for action, prob in action_probs:
        if action.extra == "pass":
            continue
        row, col = action.row, action.col
        if not (0 <= col < width):
            continue
        if row == -1:
            # Column-drop games (e.g. Connect4): place probability at the
            # lowest empty row in this column so it looks like where the
            # piece would actually land.
            empty_rows = np.where(board[:, col] == 0)[0]
            row = int(empty_rows[-1]) if len(empty_rows) > 0 else height - 1
        if 0 <= row < height:
            prob_grid[row][col] = prob

    fig, ax = plt.subplots(figsize=(max(6, width), max(6, height)))
    fig.patch.set_facecolor("#2d3340")
    ax.set_facecolor("#2d3340")

    # Heatmap layer – only non-zero cells are visible.
    masked = np.ma.masked_where(prob_grid == 0, prob_grid)
    im = ax.imshow(masked, cmap="YlOrRd", vmin=0, vmax=prob_grid.max() or 1,
                   origin="upper", extent=[-0.5, width - 0.5, height - 0.5, -0.5],
                   aspect="equal", interpolation="nearest")

    # Grid lines.
    ax.set_xticks(np.arange(-0.5, width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, height, 1), minor=True)
    ax.grid(which="minor", color="#555", linewidth=0.8)
    ax.tick_params(which="minor", length=0)
    ax.set_xticks(range(width))
    ax.set_xticklabels(range(width), color="white", fontsize=9)
    ax.set_yticks(range(height))
    ax.set_yticklabels(range(height), color="white", fontsize=9)
    ax.tick_params(colors="white")

    # Draw board pieces.
    for row in range(height):
        for col in range(width):
            cell = int(board[row][col])
            if cell == 0:
                continue
            # Use the game's player colours if available, otherwise fall back to B/W.
            try:
                p1_color, p2_color = game.player_colors
                raw = p1_color if cell == 1 else p2_color
                facecolor = tuple(c / 255 for c in raw)
                edgecolor = "white" if cell == 1 else "#333"
            except Exception:
                facecolor = "black" if cell == 1 else "white"
                edgecolor = "white" if cell == 1 else "black"
            circle = mpatches.Circle(
                (col, row), radius=0.38,
                facecolor=facecolor, edgecolor=edgecolor, linewidth=1.5,
            )
            ax.add_patch(circle)

    # Annotate probability values on non-zero cells.
    for row in range(height):
        for col in range(width):
            prob = prob_grid[row][col]
            if prob > 0.005:
                ax.text(col, row, f"{prob:.2f}", ha="center", va="center",
                        fontsize=7, color="black", fontweight="bold")

    # Colorbar.
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("MCTS probability", color="white", fontsize=10)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    # Pass-action annotation (if any).
    pass_prob = sum(p for a, p in action_probs if a.extra == "pass")
    if pass_prob > 0.001:
        ax.set_xlabel(f"Pass probability: {pass_prob:.3f}", color="white", fontsize=10)

    ax.set_title(title, color="white", fontsize=13, pad=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[policy heatmap] saved → {os.path.abspath(output_path)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """"""
    #game_name = "connect4"
    #model_path = "model/connect4/checkpoint_iter125.pt"
    game_name = "othello"
    #model_path = "model/othello/checkpoint_iter86.pt"
    model_path = "model/othello_v2_s5/checkpoint_iter161.pt"
    sims = 5000
    max_depth = 2
    max_children = 10
    tree_out = "mcts_tree.svg"
    heatmap_out = "mcts_policy.png"
    device = "mps"
    c_puct = 1.5
    filters = 256
    blocks = 15
    value_hidden = 256
    custom_board = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 2, 1, 0, 0, 0],
        [0, 0, 0, 1, 2, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0]
        ], dtype=np.long)
    custom_turn = 0
    game = GAME_BUILDERS[game_name](board=custom_board, turn=custom_turn)

    # ---- Build network ----
    net_config = ResNetConfig(
        board_height=game.height,
        board_width=game.width,
        num_actions=game.num_actions,
        num_filters=filters,
        num_residual_blocks=blocks,
        value_head_hidden_size=value_hidden,
    )
    net = PolicyValueNet(net_config)
    state_dict = torch.load(model_path, map_location=device)
    net.load_state_dict(state_dict)
    net.eval()

    # ---- Run MCTS ----
    mcts_config = MCTSConfig(
        num_simulations=sims,
        c_puct=c_puct,
        dirichlet_alpha=0.0,   # disable noise for deterministic evaluation
        dirichlet_epsilon=0.0,
        device=device,
    )
    mcts = MCTS(game, net, mcts_config)

    print(f"[mcts] Running {sims} simulations…")
    action_probs = mcts.get_action_probs(temperature=1.0, batch_size=4)

    # Print top-5 moves.
    sorted_probs = sorted(action_probs, key=lambda x: x[1], reverse=True)
    print("\n[mcts] Top moves:")
    for action, prob in sorted_probs[:5]:
        label = f"pass" if action.extra == "pass" else f"({action.row},{action.col})"
        print(f"  {label:12s}  prob={prob:.4f}")

    # ---- Visualise tree ----
    print(f"\n[tree] Rendering search tree → '{tree_out}'…")
    try:
        saved_tree = visualize_tree(
            mcts,
            output_path=tree_out,
            max_depth=max_depth,
            max_children=max_children,
        )
        print(f"[tree] saved → {saved_tree}")
    except ImportError as exc:
        print(f"[tree] Skipped: {exc}")
    except Exception as exc:
        print(f"[tree] Error: {exc}")

    # ---- Policy heatmap ----
    print(f"\n[heatmap] Rendering policy heatmap → '{heatmap_out}'…")
    player_label = "Player 1 (first)" if game.state.turn == game.PLAYER_FIRST else "Player 2 (second)"
    plot_policy_heatmap(
        game,
        action_probs,
        output_path=heatmap_out,
        title=f"{game_name} – MCTS({sims} sims) – {player_label} to move",
    )


if __name__ == "__main__":
    main()
