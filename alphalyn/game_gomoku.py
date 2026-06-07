"""game_gomoku.py – Gomoku (Five in a Row) game implementation.

Gomoku is a two-player strategy game played on a 15×15 grid.  Black (Player 1)
moves first.  Players alternate placing stones on empty intersections; the
first player to form an unbroken line of five or more stones in any direction
(horizontal, vertical, or diagonal) wins.  If the board fills completely with
no winner the game is a draw.  No pass is ever required.

Classes
-------
GomokuArgs
    Configuration container with Gomoku-specific defaults (15×15 board,
    40×40-pixel cells, wooden-brown background, Go-style intersection layout).
GomokuGame
    Concrete ``BoardGame`` subclass implementing all Gomoku rules:
    - ``move()``              – place a stone on an empty intersection.
    - ``get_legal_actions()`` – returns all empty intersections.
    - ``check_winner()``      – numpy sliding-window scan for five-in-a-row
                               in all four directions.
    - ``action_to_index`` / ``index_to_action`` / ``num_actions`` – policy
      vector encoding: flat index ``row * width + col`` (length = 225).
    - GUI nearest-intersection click and terminal ``"row,col"`` input parsing.

Usage
-----
    # Terminal mode
    python game_gomoku.py

    # pygame GUI mode
    python game_gomoku.py --gui
"""

from typing import List, Optional, Tuple
import argparse
import numpy as np

from board_game import BoardGame, GameArgs, GameState, MoveAction, run_demo

# Board dimensions for standard Gomoku
GOMOKU_SIZE: int = 15
GOMOKU_WIN_LENGTH: int = 5
GOMOKU_PADDING: int = 30

# Default grid cell size in pixels
DEFAULT_GRID_H: int = 36
DEFAULT_GRID_W: int = 36

# Board cell values (0 = empty, 1 = black / first, 2 = white / second)
BOARD_EMPTY: int = 0
BOARD_PLAYER_FIRST: int = 1   # black
BOARD_PLAYER_SECOND: int = 2  # white

# Stone colours indexed by board cell value
STONE_COLORS: Tuple = (
    (0, 0, 0),          # BOARD_EMPTY – unused placeholder
    (20, 20, 20),       # BOARD_PLAYER_FIRST  – black
    (240, 240, 240),    # BOARD_PLAYER_SECOND – white
)

BOARD_BG_COLOR: Tuple[int, int, int] = (220, 179, 92)  # wooden colour


class GomokuArgs(GameArgs):
    """Argument container for Gomoku, with game-specific defaults."""

    @staticmethod
    def add_common_args(parser: argparse.ArgumentParser) -> None:
        """Register all Gomoku arguments: common args + Gomoku-specific options."""
        GameArgs.add_common_args(parser)
        parser.add_argument("--gomoku-size", type=int, default=GOMOKU_SIZE,
                            help=f"Board size (default: {GOMOKU_SIZE})")

    def __init__(self, namespace: argparse.Namespace) -> None:
        size = getattr(namespace, "gomoku_size", GOMOKU_SIZE)
        super().__init__(namespace, width=size, height=size, title="Gomoku")


class GomokuGame(BoardGame):
    """Gomoku (Five in a Row) implementation built on top of ``BoardGame``.

    Rules
    -----
    - 15 × 15 board; stones placed on grid-line intersections.
    - Black (PLAYER_FIRST) moves first.
    - First player to connect exactly five (or more) stones in a row
      horizontally, vertically, or diagonally wins.
    - If the board is full with no winner the game is a tie.
    - Players never need to pass.
    """

    # Wooden-yellow Gomoku board colour.
    GUI_BOARD_BG_COLOR = BOARD_BG_COLOR  # (220, 179, 92)

    @property
    def name(self) -> str:
        """Canonical short name of this game (e.g. ``"othello"``, ``"connect4"``)."""
        return "gomoku"

    @property
    def player_colors(self):
        return STONE_COLORS[1], STONE_COLORS[2]

    def __init__(self, args: GomokuArgs) -> None:
        super().__init__(args)
        self._padding: int = GOMOKU_PADDING
        self._grid_h: int = DEFAULT_GRID_H
        self._grid_w: int = DEFAULT_GRID_W
        self.reset()

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    def get_board_size(self, pos_x: int, pos_y: int, grid_h: int, grid_w: int) -> Tuple[int, int]:
        return self.get_grid_board_size(self._grid_h, self._grid_w, stone_on_intersection=True, padding=self._padding)

    def draw_board(self) -> None:
        """Draw the wooden board background and all stones.

        The board background colour is applied by the base class via
        ``GUI_BOARD_BG_COLOR``; no manual fill is needed here.
        """
        self.draw_grids_and_stones(
            pos_x=self._gui_board_offset_x if self.gui_mode else 0,
            pos_y=self._gui_board_offset_y if self.gui_mode else 0,
            grid_h=self._grid_h,
            grid_w=self._grid_w,
            stone_colors=STONE_COLORS,
            stone_on_intersection=True,
            padding=self._padding,
        )

    def move(self, player: int, action: MoveAction) -> bool:
        """Place a stone at (action.row, action.col).

        Returns ``False`` if the cell is already occupied or out of bounds.
        """
        row, col = action.row, action.col
        if not (0 <= row < self.height and 0 <= col < self.width):
            return False
        if self.state.board[row][col] != BOARD_EMPTY:
            return False

        board_value = BOARD_PLAYER_FIRST if player == self.PLAYER_FIRST else BOARD_PLAYER_SECOND
        self.state.board[row][col] = board_value
        self.last_action = action

        winner = self.check_winner()
        if winner is not None:
            self.switch_turn()   # turn → opponent, so _terminal_value sees loser's perspective
            self.end_game(winner)
        elif not np.any(self.state.board == BOARD_EMPTY):
            self.switch_turn()
            self.end_game(self.RESULT_TIE)
        else:
            self.switch_turn()

        return True

    def get_legal_actions(self, player: int) -> List[MoveAction]:
        """Return all empty intersections as legal actions."""
        empty_positions = np.argwhere(self.state.board == BOARD_EMPTY)
        return [
            MoveAction(row=int(row), col=int(col), player=player)
            for row, col in empty_positions
        ]

    def action_to_index(self, action: MoveAction) -> int:
        """Map a Gomoku action to a flat policy index (``row * width + col``)."""
        return action.row * self.width + action.col

    def index_to_action(self, index: int, player: int) -> MoveAction:
        """Convert a flat policy index back to a ``MoveAction`` for Gomoku."""
        row = index // self.width
        col = index % self.width
        return MoveAction(row=row, col=col, player=player)

    @property
    def num_actions(self) -> int:
        """Gomoku has one action per board cell (no pass)."""
        return self.height * self.width

    def should_show_pass_button(self) -> bool:
        """Gomoku never requires a pass – always returns ``False``."""
        return False

    def check_winner(self) -> Optional[int]:
        """Scan for five-in-a-row using numpy sliding-window sums."""
        for board_value, player in (
            (BOARD_PLAYER_FIRST, self.PLAYER_FIRST),
            (BOARD_PLAYER_SECOND, self.PLAYER_SECOND),
        ):
            plane = (self.state.board == board_value).astype(np.int8)
            win = GOMOKU_WIN_LENGTH

            # Horizontal
            if np.any(sum(plane[:, i:self.width - win + i + 1] for i in range(win)) == win):
                return player
            # Vertical
            if np.any(sum(plane[i:self.height - win + i + 1, :] for i in range(win)) == win):
                return player
            # Diagonal ↘
            if np.any(sum(plane[i:self.height - win + i + 1, i:self.width - win + i + 1] for i in range(win)) == win):
                return player
            # Diagonal ↙
            if np.any(sum(plane[i:self.height - win + i + 1, win - i - 1:self.width - i] for i in range(win)) == win):
                return player

        return None

    def reset_state(self, state: GameState) -> GameState:
        """Reset to an empty Gomoku board."""
        state.board = np.zeros((self.height, self.width), dtype=int)
        state.turn = self.PLAYER_FIRST
        state.winner = None
        state.total_turns = 0
        state.is_game_over = False
        return state

    def draw_last_move_marker(self, action: MoveAction) -> None:
        """Draw a contrasting dot on the last-placed stone (intersection layout)."""
        self.draw_grid_marker(
            action, self._grid_h, self._grid_w,
            first_color=(255, 255, 255), second_color=(20, 20, 20),
            stone_on_intersection=True, padding=self._padding,
        )

    def handle_board_click(
        self, pixel_x: int, pixel_y: int, grid_h: int, grid_w: int
    ) -> Optional[MoveAction]:
        """Convert a GUI click to the nearest intersection ``MoveAction``."""
        local_x = pixel_x - self._gui_board_offset_x
        local_y = pixel_y - self._gui_board_offset_y
        col = round((local_x - self._padding) / self._grid_w)
        row = round((local_y - self._padding) / self._grid_h)
        if not (0 <= row < self.height and 0 <= col < self.width):
            return None
        if self.state.board[row][col] != BOARD_EMPTY:
            return None
        return MoveAction(row=row, col=col, player=self.state.turn)

    def parse_terminal_input(self, user_input: str) -> Optional[MoveAction]:
        """Parse ``row,col`` terminal input into a ``MoveAction``."""
        try:
            parts = user_input.replace(" ", "").split(",")
            row, col = int(parts[0]), int(parts[1])
        except (ValueError, IndexError):
            print("Please enter row,col (e.g. 7,7).")
            return None

        if not (0 <= row < self.height and 0 <= col < self.width):
            print(f"Out of bounds. Row and col must be in [0, {self.height - 1}].")
            return None
        if self.state.board[row][col] != BOARD_EMPTY:
            print(f"Cell ({row},{col}) is already occupied.")
            return None

        return MoveAction(row=row, col=col, player=self.state.turn)


if __name__ == "__main__":
    _parser = argparse.ArgumentParser(description="Gomoku")
    GomokuArgs.add_common_args(_parser)
    run_demo(GomokuArgs(_parser.parse_args()), GomokuGame)
