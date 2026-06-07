"""game_othello.py – Othello (Reversi) game implementation.

Othello is a two-player strategy game played on an 8×8 board with stones
that are black on one side and white on the other.  The game starts with
four stones in the centre (2 black, 2 white in a diagonal arrangement).

A placement is legal only if it flanks at least one opponent stone in any of
the eight directions; all flanked stones are then flipped to the placing
player's colour.  If a player has no legal move they must pass (encoded as a
special action with ``extra="pass"``).  The game ends when neither player can
move; the player with more stones wins, equal counts is a draw.

Classes
-------
OthelloArgs
    Configuration container with Othello-specific defaults (8×8 board,
    70×70-pixel cells, forest-green background).
OthelloGame
    Concrete ``BoardGame`` subclass implementing all Othello rules:
    - ``move()``              – place a stone and flip flanked opponents;
                               handles the pass action.
    - ``get_legal_actions()`` – returns all legal placements for a player.
    - ``check_winner()``      – declares winner when neither side can move.
    - ``action_to_index`` / ``index_to_action`` / ``num_actions`` – policy
      vector encoding: ``row * width + col`` for placements, last index for
      pass (length = ``height * width + 1 = 65``).
    - GUI intersection-click and terminal ``"row,col"`` input parsing.

Usage
-----
    # Terminal mode
    python game_othello.py

    # pygame GUI mode
    python game_othello.py --gui
"""

from typing import List, Optional, Tuple
import argparse
import numpy as np

from board_game import BoardGame, GameArgs, GameState, MoveAction, run_demo

# Board dimensions for standard Othello
OTHELLO_SIZE: int = 8
OTHELLO_PADDING: int = 30

# Default grid cell size in pixels
DEFAULT_GRID_H: int = 70
DEFAULT_GRID_W: int = 70

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

BOARD_BG_COLOR: Tuple[int, int, int] = (34, 139, 34)  # forest green

# All 8 directions for flip scanning
DIRECTIONS: List[Tuple[int, int]] = [
    (-1, -1), (-1, 0), (-1, 1),
    (0,  -1),           (0,  1),
    (1,  -1),  (1, 0),  (1,  1),
]


class OthelloArgs(GameArgs):
    """Argument container for Othello, with game-specific defaults."""

    def __init__(self, namespace: argparse.Namespace) -> None:
        super().__init__(namespace, width=OTHELLO_SIZE, height=OTHELLO_SIZE, title="Othello")


class OthelloGame(BoardGame):
    """Othello (Reversi) implementation built on top of ``BoardGame``.

    Rules
    -----
    - 8 × 8 board; stones placed on grid-line intersections.
    - Initial position: four stones in the centre (2 black, 2 white).
    - A move is legal only if it flanks at least one opponent stone in any of
      the 8 directions; all flanked stones are flipped.
    - If the current player has no legal moves they must pass.
    - The game ends when neither player can move.
    - The player with more stones wins; equal counts is a tie.
    """

    # Forest-green Othello board colour.
    GUI_BOARD_BG_COLOR = BOARD_BG_COLOR  # (34, 139, 34)

    @property
    def name(self) -> str:
        """Canonical short name of this game (e.g. ``"othello"``, ``"connect4"``)."""
        return "othello"

    @property
    def player_colors(self):
        return STONE_COLORS[1], STONE_COLORS[2]

    def __init__(self, args: OthelloArgs) -> None:
        super().__init__(args)
        self._grid_h: int = args.grid_h
        self._grid_w: int = args.grid_w
        self.reset()

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    def get_board_size(self, pos_x: int, pos_y: int, grid_h: int, grid_w: int) -> Tuple[int, int]:
        return self.get_grid_board_size(grid_h, grid_w, stone_on_intersection=False)

    def draw_board(self) -> None:
        """Draw the green board background and all stones (stone-in-cell layout)."""
        self.draw_grids_and_stones(
            pos_x=self._gui_board_offset_x if self.gui_mode else 0,
            pos_y=self._gui_board_offset_y if self.gui_mode else 0,
            grid_h=self._grid_h,
            grid_w=self._grid_w,
            stone_colors=STONE_COLORS,
            stone_on_intersection=False,
        )

    def move(self, player: int, action: MoveAction) -> bool:
        """Place a stone and flip all flanked opponent stones.

        A pass action is identified by ``action.extra == "pass"`` (and
        ``action.row == action.col == -1``).  Pass is only legal when the
        player has no legal placements; in that case the turn is handed to
        the opponent without modifying the board.

        Returns ``False`` if the move is illegal.
        """
        if action.extra == "pass":
            self.last_action = action
            self.switch_turn()
            return True

        row, col = action.row, action.col

        board_value = BOARD_PLAYER_FIRST if player == self.PLAYER_FIRST else BOARD_PLAYER_SECOND
        self.state.board[row][col] = board_value
        self._flip_stones(player, row, col)
        self.last_action = action

        winner = self.check_winner()
        if winner is not None:
            self.switch_turn()   # turn → opponent, so _terminal_value sees loser's perspective
            self.end_game(winner)
        else:
            self.switch_turn()

        return True

    def get_legal_actions(self, player: int) -> List[MoveAction]:
        """Return all legal actions for *player*.

        When no placement is available the only legal action is a pass
        (``extra="pass"``), which MCTS needs to expand the node correctly.
        """
        legal_actions = []
        for row in range(self.height):
            for col in range(self.width):
                if self._is_legal(player, row, col):
                    legal_actions.append(MoveAction(row=row, col=col, player=player))
        if not legal_actions:
            # Pass is only valid when the opponent still has real placements.
            # If neither side can move, the game is over (handled by check_winner).
            opponent = self.PLAYER_SECOND if player == self.PLAYER_FIRST else self.PLAYER_FIRST
            opponent_has_real_moves = any(
                self._is_legal(opponent, row, col)
                for row in range(self.height)
                for col in range(self.width)
            )
            if opponent_has_real_moves:
                legal_actions.append(MoveAction(row=-1, col=-1, player=player, extra="pass"))
        return legal_actions

    def action_to_index(self, action: MoveAction) -> int:
        """Map an Othello action to a flat policy index.

        Regular placements use ``row * width + col``.
        The pass action (``extra == "pass"``) maps to the last index
        ``height * width``.
        """
        if action.extra == "pass":
            return self.height * self.width
        return action.row * self.width + action.col

    def index_to_action(self, index: int, player: int) -> MoveAction:
        """Convert a flat policy index back to a ``MoveAction`` for Othello."""
        pass_index = self.height * self.width
        if index == pass_index:
            return MoveAction(row=-1, col=-1, player=player, extra="pass")
        row = index // self.width
        col = index % self.width
        return MoveAction(row=row, col=col, player=player)

    @property
    def num_actions(self) -> int:
        """Othello has one action per board cell plus one pass action."""
        return self.height * self.width + 1

    def should_show_pass_button(self) -> bool:
        """Show Pass when the current player has no real placements (only pass available)."""
        if self.state.is_game_over:
            return False
        actions = self.get_legal_actions(self.state.turn)
        return all(a.extra == "pass" for a in actions)

    def check_winner(self) -> Optional[int]:
        """Return the winner when neither player can move, or ``None`` if ongoing."""
        current_player = self.state.turn
        opponent = self.PLAYER_SECOND if current_player == self.PLAYER_FIRST else self.PLAYER_FIRST

        # get_legal_actions returns an empty list only when both sides have no
        # real placements (the opponent check is already embedded in get_legal_actions).
        # A non-empty list means either real placements or a valid pass – game ongoing.
        if self.get_legal_actions(current_player):
            return None  # game still ongoing

        # Count stones
        first_count = int(np.sum(self.state.board == BOARD_PLAYER_FIRST))
        second_count = int(np.sum(self.state.board == BOARD_PLAYER_SECOND))

        if first_count > second_count:
            return self.PLAYER_FIRST
        if second_count > first_count:
            return self.PLAYER_SECOND
        return self.RESULT_TIE

    def reset_state(self, state: GameState) -> GameState:
        """Reset to the standard Othello starting position."""
        state.board = np.zeros((self.height, self.width), dtype=int)
        # Place the four centre stones
        mid = self.height // 2
        state.board[mid - 1][mid - 1] = BOARD_PLAYER_SECOND  # white
        state.board[mid - 1][mid]     = BOARD_PLAYER_FIRST   # black
        state.board[mid][mid - 1]     = BOARD_PLAYER_FIRST   # black
        state.board[mid][mid]         = BOARD_PLAYER_SECOND  # white
        state.turn = self.PLAYER_FIRST
        state.winner = None
        state.total_turns = 0
        state.is_game_over = False
        return state

    def draw_last_move_marker(self, action: MoveAction) -> None:
        """Draw a contrasting dot on the last-placed stone."""
        self.draw_grid_marker(
            action, self._grid_h, self._grid_w,
            first_color=(255, 255, 255), second_color=(20, 20, 20),
            stone_on_intersection=False,
        )

    def handle_board_click(
        self, pixel_x: int, pixel_y: int, grid_h: int, grid_w: int
    ) -> Optional[MoveAction]:
        """Convert a GUI click to a board cell ``MoveAction`` (stone-in-cell layout)."""
        local_x = pixel_x - self._gui_board_offset_x
        local_y = pixel_y - self._gui_board_offset_y
        col = local_x // grid_w
        row = local_y // grid_h
        if not (0 <= row < self.height and 0 <= col < self.width):
            return None
        if not self._is_legal(self.state.turn, row, col):
            return None
        return MoveAction(row=row, col=col, player=self.state.turn)

    def parse_terminal_input(self, user_input: str) -> Optional[MoveAction]:
        """Parse ``row,col`` terminal input into a ``MoveAction``."""
        legal_actions = self.get_legal_actions(self.state.turn)
        legal_coords = [(a.row, a.col) for a in legal_actions]
        print(f"Legal moves (row,col): {legal_coords}")

        try:
            parts = user_input.replace(" ", "").split(",")
            row, col = int(parts[0]), int(parts[1])
        except (ValueError, IndexError):
            print("Please enter row,col (e.g. 3,4).")
            return None

        if (row, col) not in legal_coords:
            print(f"Illegal move. Choose from {legal_coords}.")
            return None

        return MoveAction(row=row, col=col, player=self.state.turn)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _opponent_board_value(self, player: int) -> int:  # noqa: E301
        return BOARD_PLAYER_SECOND if player == self.PLAYER_FIRST else BOARD_PLAYER_FIRST

    def _player_board_value(self, player: int) -> int:
        return BOARD_PLAYER_FIRST if player == self.PLAYER_FIRST else BOARD_PLAYER_SECOND

    def _flanked_in_direction(
        self, player: int, row: int, col: int, delta_row: int, delta_col: int
    ) -> List[Tuple[int, int]]:
        """Return list of opponent positions flanked in one direction, or [] if none."""
        opponent_value = self._opponent_board_value(player)
        player_value = self._player_board_value(player)
        flanked = []
        next_row, next_col = row + delta_row, col + delta_col

        while 0 <= next_row < self.height and 0 <= next_col < self.width:
            cell = int(self.state.board[next_row][next_col])
            if cell == opponent_value:
                flanked.append((next_row, next_col))
            elif cell == player_value:
                return flanked  # valid flank found
            else:
                break  # empty cell – no flank
            next_row += delta_row
            next_col += delta_col

        return []  # reached edge or empty without closing

    def _is_legal(self, player: int, row: int, col: int) -> bool:
        """Return True if placing at (row, col) is a legal move for *player*.

        Pass actions use ``row == col == -1`` and are never legal placements;
        they must be validated separately in ``move``.
        """
        if row == -1 or col == -1:
            return False
        if self.state.board[row][col] != BOARD_EMPTY:
            return False
        return any(
            len(self._flanked_in_direction(player, row, col, dr, dc)) > 0
            for dr, dc in DIRECTIONS
        )

    def _flip_stones(self, player: int, row: int, col: int) -> None:
        """Flip all opponent stones flanked by the stone just placed at (row, col)."""
        player_value = self._player_board_value(player)
        for delta_row, delta_col in DIRECTIONS:
            for flip_row, flip_col in self._flanked_in_direction(player, row, col, delta_row, delta_col):
                self.state.board[flip_row][flip_col] = player_value


if __name__ == "__main__":
    _parser = argparse.ArgumentParser(description="Othello")
    GameArgs.add_common_args(_parser)
    run_demo(OthelloArgs(_parser.parse_args()), OthelloGame)
