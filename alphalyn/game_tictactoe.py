"""game_tictactoe.py – Tic-Tac-Toe game implementation.

Tic-Tac-Toe is a two-player game played on a 3×3 grid.  Player 1 places X
and Player 2 places O, alternating turns.  The first player to align three
of their pieces in a row, column, or diagonal wins.  If all nine cells are
filled with no winner the game is a draw.

Classes
-------
TicTacToeArgs
    Configuration container with Tic-Tac-Toe-specific defaults (3×3 board,
    120×120-pixel cells).
TicTacToeGame
    Concrete ``BoardGame`` subclass implementing all game rules:
    - ``move()``              – place a piece on an empty cell.
    - ``get_legal_actions()`` – returns all empty cells.
    - ``check_winner()``      – numpy row/column/diagonal sum checks.
    - ``action_to_index`` / ``index_to_action`` / ``num_actions`` – policy
      vector encoding: flat index ``row * width + col`` (length = 9).
    - GUI click and terminal ``"row col"`` input parsing.

Usage
-----
    # Terminal mode
    python game_tictactoe.py

    # pygame GUI mode
    python game_tictactoe.py --gui
"""

from typing import List, Optional, Tuple
import argparse
import numpy as np

from board_game import BoardGame, GameArgs, GameState, MoveAction, run_demo

# Board dimensions for standard Tic-Tac-Toe
TICTACTOE_SIZE: int = 3
TICTACTOE_WIN_LENGTH: int = 3

# Default grid cell size in pixels
DEFAULT_GRID_H: int = 120
DEFAULT_GRID_W: int = 120

# Board cell values – separate from PLAYER_FIRST/PLAYER_SECOND (0/1)
# so that empty cells (0) are never confused with a player's piece.
BOARD_EMPTY: int = 0
BOARD_PLAYER_FIRST: int = 1   # X
BOARD_PLAYER_SECOND: int = 2  # O

# Stone colours indexed by board cell value.
# Index 0 (empty) is unused but kept so the list is directly indexable.
STONE_COLORS: Tuple[
    Tuple[int, int, int],
    Tuple[int, int, int],
    Tuple[int, int, int],
] = (
    (0, 0, 0),       # BOARD_EMPTY   – unused placeholder
    (220, 80, 80),   # BOARD_PLAYER_FIRST  – red (X)
    (80, 130, 220),  # BOARD_PLAYER_SECOND – blue (O)
)


class TicTacToeArgs(GameArgs):
    """Argument container for Tic-Tac-Toe with game-specific defaults."""

    def __init__(self, namespace: argparse.Namespace) -> None:
        super().__init__(namespace, width=TICTACTOE_SIZE, height=TICTACTOE_SIZE, title="Tic-Tac-Toe")


class TicTacToeGame(BoardGame):
    """Tic-Tac-Toe (3×3) implementation built on top of ``BoardGame``.

    Rules
    -----
    - 3×3 board; players alternate placing X (Player 1) and O (Player 2).
    - First player to align three pieces horizontally, vertically, or
      diagonally wins.
    - If the board is full with no winner the game is a tie.
    - No pass is ever needed.
    """

    # Light wood/tan colour for the TicTacToe board background.
    GUI_BOARD_BG_COLOR = (210, 190, 155)

    @property
    def name(self) -> str:
        """Canonical short name of this game (e.g. ``"othello"``, ``"connect4"``)."""
        return "tictactoe"

    @property
    def player_colors(self):
        return STONE_COLORS[1], STONE_COLORS[2]

    def __init__(self, args: GameArgs) -> None:
        super().__init__(args)
        self._grid_h: int = DEFAULT_GRID_H
        self._grid_w: int = DEFAULT_GRID_W
        self.reset()

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    def get_board_size(self, pos_x: int, pos_y: int, grid_h: int, grid_w: int) -> Tuple[int, int]:
        """Return pixel board size for the stone-in-cell layout."""
        return self.get_grid_board_size(self._grid_h, self._grid_w, stone_on_intersection=False)

    def draw_board(self) -> None:
        """Draw the board grid and all placed pieces."""
        self.draw_grids_and_stones(
            pos_x=self._gui_board_offset_x if self.gui_mode else 0,
            pos_y=self._gui_board_offset_y if self.gui_mode else 0,
            grid_h=self._grid_h,
            grid_w=self._grid_w,
            stone_colors=STONE_COLORS,
            stone_on_intersection=False,
        )

    def move(self, player: int, action: MoveAction) -> bool:
        """Place a piece at (action.row, action.col).

        Returns
        -------
        bool
            ``True`` if the move was legal and applied, ``False`` otherwise.
        """
        row, col = action.row, action.col
        if not self._is_legal(row, col):
            return False

        board_value = BOARD_PLAYER_FIRST if player == self.PLAYER_FIRST else BOARD_PLAYER_SECOND
        self.state.board[row][col] = board_value
        self.last_action = action

        winner = self.check_winner()
        if winner is not None:
            self.switch_turn()   # turn → opponent, so _terminal_value sees loser's perspective
            self.end_game(winner)
        elif not self._has_any_empty_cell():
            self.switch_turn()
            self.end_game(self.RESULT_TIE)
        else:
            self.switch_turn()

        return True

    def get_legal_actions(self, player: int) -> List[MoveAction]:
        """Return a ``MoveAction`` for every empty cell on the board."""
        empty_positions = np.argwhere(self.state.board == BOARD_EMPTY)
        return [
            MoveAction(row=int(pos[0]), col=int(pos[1]), player=player)
            for pos in empty_positions
        ]

    def check_winner(self) -> Optional[int]:
        """Scan all rows, columns, and diagonals for a three-in-a-row.

        Returns
        -------
        int or None
            ``PLAYER_FIRST``, ``PLAYER_SECOND``, or ``None`` if no winner yet.
        """
        for board_value, player in (
            (BOARD_PLAYER_FIRST, self.PLAYER_FIRST),
            (BOARD_PLAYER_SECOND, self.PLAYER_SECOND),
        ):
            plane = (self.state.board == board_value).astype(np.int8)

            # Rows and columns
            if np.any(plane.sum(axis=1) == TICTACTOE_WIN_LENGTH):
                return player
            if np.any(plane.sum(axis=0) == TICTACTOE_WIN_LENGTH):
                return player

            # Main diagonal (↘) and anti-diagonal (↙)
            if plane.trace() == TICTACTOE_WIN_LENGTH:
                return player
            if np.fliplr(plane).trace() == TICTACTOE_WIN_LENGTH:
                return player

        return None

    def action_to_index(self, action: MoveAction) -> int:
        """Flatten (row, col) to a single policy index in [0, 8]."""
        return action.row * self.width + action.col

    def index_to_action(self, index: int, player: int) -> MoveAction:
        """Convert a flat policy index back to a ``MoveAction``."""
        row, col = divmod(index, self.width)
        return MoveAction(row=row, col=col, player=player)

    @property
    def num_actions(self) -> int:
        """Tic-Tac-Toe has one action per cell (height × width = 9)."""
        return self.height * self.width

    def should_show_pass_button(self) -> bool:
        """Tic-Tac-Toe never requires a pass – always returns ``False``."""
        return False

    def reset_state(self, state: GameState) -> GameState:
        """Reset *state* to the initial Tic-Tac-Toe configuration."""
        state.board = np.zeros((self.height, self.width), dtype=int)
        state.turn = self.PLAYER_FIRST
        state.winner = None
        state.total_turns = 0
        state.is_game_over = False
        return state

    def draw_last_move_marker(self, action: MoveAction) -> None:
        """Draw a contrasting dot on the last-placed piece."""
        self.draw_grid_marker(
            action, self._grid_h, self._grid_w,
            first_color=(255, 255, 255), second_color=(20, 20, 20),
            stone_on_intersection=False,
        )

    def parse_terminal_input(self, user_input: str) -> Optional[MoveAction]:
        """Parse ``"row col"`` terminal input into a ``MoveAction``.

        Expects two space-separated integers, e.g. ``"1 2"`` for row 1, col 2.
        Returns ``None`` and prints a hint if the input is invalid or the cell
        is already occupied.
        """
        legal_actions = self.get_legal_actions(self.state.turn)
        legal_coords = [(a.row, a.col) for a in legal_actions]
        print(f"Legal cells (row col): {legal_coords}")

        parts = user_input.split()
        if len(parts) != 2:
            print("Enter row and column separated by a space, e.g. '1 2'.")
            return None

        try:
            row, col = int(parts[0]), int(parts[1])
        except ValueError:
            print("Row and column must be integers.")
            return None

        if (row, col) not in legal_coords:
            print(f"Cell ({row}, {col}) is not available. Choose from {legal_coords}.")
            return None

        return MoveAction(row=row, col=col, player=self.state.turn)

    def handle_board_click(
        self, pixel_x: int, pixel_y: int, grid_h: int, grid_w: int
    ) -> Optional[MoveAction]:
        """Translate a GUI left-click into a ``MoveAction`` for Tic-Tac-Toe.

        Parameters
        ----------
        pixel_x:
            X pixel coordinate of the click within the window.
        pixel_y:
            Y pixel coordinate of the click within the window.
        grid_h:
            Height of one grid cell in pixels.
        grid_w:
            Width of one grid cell in pixels.

        Returns
        -------
        MoveAction or None
            A valid ``MoveAction`` if the clicked cell is empty, ``None`` otherwise.
        """
        local_x = pixel_x - self._gui_board_offset_x
        local_y = pixel_y - self._gui_board_offset_y
        col = local_x // self._grid_w
        row = local_y // self._grid_h

        if not self._is_legal(row, col):
            return None

        return MoveAction(row=row, col=col, player=self.state.turn)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _is_legal(self, row: int, col: int) -> bool:
        """Return ``True`` if (row, col) is in-bounds and the cell is empty."""
        if row < 0 or row >= self.height or col < 0 or col >= self.width:
            return False
        return int(self.state.board[row][col]) == BOARD_EMPTY

    def _has_any_empty_cell(self) -> bool:
        """Return ``True`` if at least one cell is still empty."""
        return bool(np.any(self.state.board == BOARD_EMPTY))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _parser = argparse.ArgumentParser(description="Tic-Tac-Toe")
    GameArgs.add_common_args(_parser)
    run_demo(TicTacToeArgs(_parser.parse_args()), TicTacToeGame)
