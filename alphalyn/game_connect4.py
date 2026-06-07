"""game_connect4.py – Connect Four game implementation.

Connect Four is a two-player strategy game played on a 6×7 vertical grid.
Players take turns dropping coloured pieces into one of the seven columns;
each piece falls to the lowest available row in that column.  The first
player to form a horizontal, vertical, or diagonal line of four consecutive
pieces wins.  The game is a draw if the board fills completely with no winner.

Classes
-------
Connect4Args
    Configuration container with Connect Four-specific defaults (6 rows,
    7 columns, 80×80-pixel cells, blue board background).
Connect4Game
    Concrete ``BoardGame`` subclass implementing all game rules:
    - ``move()``         – drop a piece into a column (gravity applies).
    - ``get_legal_actions()`` – returns all non-full columns.
    - ``check_winner()`` – numpy sliding-window scan for four-in-a-row.
    - ``action_to_index`` / ``index_to_action`` / ``num_actions`` – policy
      vector encoding: one entry per column (length = ``width = 7``).
    - GUI click and terminal input parsing.

Usage
-----
    # Terminal mode
    python game_connect4.py

    # pygame GUI mode
    python game_connect4.py --gui
"""

from typing import List, Optional, Tuple
import argparse
import numpy as np

from board_game import BoardGame, GameArgs, GameState, MoveAction, run_demo


# Board dimensions for standard Connect Four
CONNECT4_ROWS: int = 6
CONNECT4_COLS: int = 7
CONNECT4_WIN_LENGTH: int = 4


# Default grid cell size in pixels
DEFAULT_GRID_H: int = 80
DEFAULT_GRID_W: int = 80


class Connect4Args(GameArgs):
    """Argument container for Connect Four, with game-specific defaults."""

    def __init__(self, namespace: argparse.Namespace) -> None:
        super().__init__(namespace, width=CONNECT4_COLS, height=CONNECT4_ROWS, title="Connect Four")

# Board cell values – kept separate from PLAYER_FIRST/PLAYER_SECOND (which are 0/1)
# so that empty cells (0) are never confused with a player's piece.
BOARD_EMPTY: int = 0
BOARD_PLAYER_FIRST: int = 1
BOARD_PLAYER_SECOND: int = 2

# Stone colours indexed by board cell value: index 1 = Player 1, index 2 = Player 2.
# Index 0 (empty) is unused but kept so the list is directly indexable by cell value.
STONE_COLORS: Tuple[
    Tuple[int, int, int],
    Tuple[int, int, int],
    Tuple[int, int, int],
] = (
    (0, 0, 0),       # BOARD_EMPTY   – unused placeholder
    (220, 50, 50),   # BOARD_PLAYER_FIRST  – red
    (230, 200, 30),  # BOARD_PLAYER_SECOND – yellow
)

# Board background colour (GUI)
BOARD_COLOR: Tuple[int, int, int] = (30, 80, 180)  # blue


class Connect4Game(BoardGame):
    """Connect Four implementation built on top of ``BoardGame``.

    Rules
    -----
    - 6 × 7 board; pieces fall to the lowest empty row in the chosen column.
    - First player to connect four pieces horizontally, vertically, or
      diagonally wins.
    - If the board is full with no winner the game is a tie.
    - Players never need to pass (``should_show_pass_button`` always returns
      ``False``).
    """

    # Override board background to the classic blue Connect Four colour.
    GUI_BOARD_BG_COLOR = BOARD_COLOR  # (30, 80, 180)

    @property
    def name(self) -> str:
        """Canonical short name of this game (e.g. ``"othello"``, ``"connect4"``)."""
        return "connect4"

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
        return self.get_grid_board_size(grid_h, grid_w, stone_on_intersection=False)

    def draw_board(self) -> None:
        """Draw the board background and all pieces.

        The blue board background is applied automatically by the base class
        via ``GUI_BOARD_BG_COLOR``; no manual fill is needed here.
        """
        self.draw_grids_and_stones(
            pos_x=self._gui_board_offset_x if self.gui_mode else 0,
            pos_y=self._gui_board_offset_y if self.gui_mode else 0,
            grid_h=self._grid_h,
            grid_w=self._grid_w,
            stone_colors=STONE_COLORS,
            stone_on_intersection=False,
        )

    def move(self, player: int, action: MoveAction) -> bool:
        """Drop a piece into the column specified by ``action.col``.

        The piece falls to the lowest empty row in that column.

        Returns
        -------
        bool
            ``True`` if the move was legal and applied, ``False`` otherwise.
        """
        col = action.col
        if col < 0 or col >= self.width:
            return False

        # Find the lowest empty row in this column
        target_row = self._find_drop_row(col)
        if target_row is None:
            return False  # column is full

        # Map player constant → board cell value to avoid collision with BOARD_EMPTY (0)
        board_value = BOARD_PLAYER_FIRST if player == self.PLAYER_FIRST else BOARD_PLAYER_SECOND
        self.state.board[target_row][col] = board_value
        self.last_action = MoveAction(row=target_row, col=col, player=player)

        winner = self.check_winner()
        if winner is not None:
            self.switch_turn()   # turn → opponent, so _terminal_value sees loser's perspective
            self.end_game(winner)
        elif not self._has_any_legal_action():
            self.switch_turn()
            self.end_game(self.RESULT_TIE)
        else:
            self.switch_turn()

        return True

    def get_legal_actions(self, player: int) -> List[MoveAction]:
        """Return one ``MoveAction`` per non-full column."""
        # A column is legal iff its top cell (row 0) is still empty (== 0).
        # np.where returns column indices where that condition holds.
        legal_cols = np.where(self.state.board[0, :] == 0)[0]
        return [MoveAction(row=-1, col=int(col), player=player) for col in legal_cols]

    def check_winner(self) -> Optional[int]:
        """Scan the board for a four-in-a-row using numpy sliding-window sums.

        Returns
        -------
        int or None
            ``PLAYER_FIRST``, ``PLAYER_SECOND``, or ``None`` if no winner yet.
            (Tie is handled separately in ``move()``.)
        """
        # Build binary planes using board cell values (1/2), not player constants (0/1).
        # This avoids false positives from empty cells (BOARD_EMPTY = 0).
        for board_value, player in (
            (BOARD_PLAYER_FIRST, self.PLAYER_FIRST),
            (BOARD_PLAYER_SECOND, self.PLAYER_SECOND),
        ):
            plane = (self.state.board == board_value).astype(np.int8)

            # Horizontal: sum over 4 consecutive columns
            if np.any(plane[:, :-3] + plane[:, 1:-2] + plane[:, 2:-1] + plane[:, 3:] == CONNECT4_WIN_LENGTH):
                return player

            # Vertical: sum over 4 consecutive rows
            if np.any(plane[:-3, :] + plane[1:-2, :] + plane[2:-1, :] + plane[3:, :] == CONNECT4_WIN_LENGTH):
                return player

            # Diagonal ↘
            if np.any(
                plane[:-3, :-3] + plane[1:-2, 1:-2] + plane[2:-1, 2:-1] + plane[3:, 3:]
                == CONNECT4_WIN_LENGTH
            ):
                return player

            # Diagonal ↙
            if np.any(
                plane[:-3, 3:] + plane[1:-2, 2:-1] + plane[2:-1, 1:-2] + plane[3:, :-3]
                == CONNECT4_WIN_LENGTH
            ):
                return player

        return None

    def action_to_index(self, action: MoveAction) -> int:
        """Map a Connect Four action to a policy index.

        Connect Four actions only vary by column (pieces fall to the lowest
        empty row automatically), so the policy vector has length ``width``
        with no pass slot needed.
        """
        return action.col

    def index_to_action(self, index: int, player: int) -> MoveAction:
        """Convert a column index back to a ``MoveAction`` for Connect Four."""
        return MoveAction(row=-1, col=index, player=player)

    @property
    def num_actions(self) -> int:
        """Connect Four has one action per column (no pass)."""
        return self.width

    def should_show_pass_button(self) -> bool:
        """Connect Four never requires a pass – always returns ``False``."""
        return False

    def reset_state(self, state: GameState) -> GameState:
        """Reset *state* to the initial Connect Four configuration."""
        state.board = np.zeros((self.height, self.width), dtype=int)
        state.turn = self.PLAYER_FIRST
        state.winner = None
        state.total_turns = 0
        state.is_game_over = False
        return state

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _find_drop_row(self, col: int) -> Optional[int]:
        """Return the lowest empty row index in *col*, or ``None`` if full.

        Scans the column from bottom to top using numpy to find the first
        zero cell without a Python-level loop.
        """
        col_data = self.state.board[:, col]
        # Flip so index 0 is the bottom row; find the first zero from the bottom.
        flipped = col_data[::-1]
        empty_indices = np.where(flipped == 0)[0]
        if empty_indices.size == 0:
            return None
        # Convert flipped index back to original row index
        return int(self.height - 1 - empty_indices[0])

    def _has_any_legal_action(self) -> bool:
        """Return ``True`` if at least one column still has an empty cell."""
        # The top row (row 0) being zero means that column is not full.
        return bool(np.any(self.state.board[0, :] == 0))

    def _check_direction(
        self,
        board: np.ndarray,
        start_row: int,
        start_col: int,
        delta_row: int,
        delta_col: int,
        player: int,
    ) -> bool:
        """Return ``True`` if there are ``CONNECT4_WIN_LENGTH`` consecutive
        *player* pieces starting at (start_row, start_col) in the given direction."""
        for step in range(1, CONNECT4_WIN_LENGTH):
            next_row = start_row + delta_row * step
            next_col = start_col + delta_col * step
            if next_row < 0 or next_row >= self.height:
                return False
            if next_col < 0 or next_col >= self.width:
                return False
            if int(board[next_row][next_col]) != player:
                return False
        return True


# ---------------------------------------------------------------------------
# Minimal terminal demo
# ---------------------------------------------------------------------------

    def parse_terminal_input(self, user_input: str) -> Optional[MoveAction]:
        """Parse a terminal input string into a ``MoveAction`` for Connect Four.

        Expects the user to type a column number (0-indexed).  Returns ``None``
        and prints a hint if the input is invalid or the column is full.

        Parameters
        ----------
        user_input:
            Raw string from ``input()``, already stripped.

        Returns
        -------
        MoveAction or None
            A valid ``MoveAction`` if the input is legal, ``None`` otherwise.
        """
        legal_cols = [action.col for action in self.get_legal_actions(self.state.turn)]
        print(f"Legal columns: {legal_cols}")

        try:
            chosen_col = int(user_input)
        except ValueError:
            print("Please enter a column number.")
            return None

        if chosen_col not in legal_cols:
            print(f"Invalid column. Choose from {legal_cols}.")
            return None

        return MoveAction(row=-1, col=chosen_col, player=self.state.turn)

    def draw_last_move_marker(self, action: MoveAction) -> None:
        """Draw a contrasting dot on the last-dropped piece."""
        self.draw_grid_marker(
            action, self._grid_h, self._grid_w,
            first_color=(255, 255, 255), second_color=(20, 20, 20),
            stone_on_intersection=False,
        )

    def handle_board_click(
        self, pixel_x: int, pixel_y: int, grid_h: int, grid_w: int
    ) -> Optional[MoveAction]:
        """Translate a GUI left-click into a ``MoveAction`` for Connect Four.

        In Connect Four the player chooses a column by clicking anywhere in
        that column.  The piece falls to the lowest empty row automatically.

        Parameters
        ----------
        pixel_x:
            X pixel coordinate of the click within the window.
        pixel_y:
            Y pixel coordinate of the click (unused – column is determined by x).
        grid_h:
            Height of one grid cell in pixels (unused here).
        grid_w:
            Width of one grid cell in pixels.

        Returns
        -------
        MoveAction or None
            A valid ``MoveAction`` if the clicked column is legal, ``None`` otherwise.
        """
        local_x = pixel_x - self._gui_board_offset_x
        col = local_x // grid_w
        if col < 0 or col >= self.width:
            return None
        if self._find_drop_row(col) is None:
            return None  # column is full
        return MoveAction(row=-1, col=col, player=self.state.turn)


if __name__ == "__main__":
    _parser = argparse.ArgumentParser(description="Connect Four")
    GameArgs.add_common_args(_parser)
    run_demo(Connect4Args(_parser.parse_args()), Connect4Game)
