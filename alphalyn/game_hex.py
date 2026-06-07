"""game_hex.py – Hex board game implementation.

Hex is a two-player connection game played on a rhombus-shaped board of
hexagonal cells (default 11×11).  Players alternate placing stones on empty
cells.  Player 1 (Red) wins by connecting the top edge to the bottom edge;
Player 2 (Blue) wins by connecting the left edge to the right edge.

Hex has no draws – exactly one player must win (proven by John Nash's
strategy-stealing argument).  There is no pass action.

The board uses offset coordinates (odd-row shift right) for rendering
hexagonal cells as pointy-top hexagons.

Classes
-------
HexArgs
    Configuration container with Hex-specific defaults (11×11 board,
    40-pixel hex radius).
HexGame
    Concrete ``BoardGame`` subclass implementing all Hex rules:
    - ``move()``              – place a stone on an empty cell.
    - ``get_legal_actions()`` – returns all empty cells.
    - ``check_winner()``      – DFS connectivity check for both players.
    - ``action_to_index`` / ``index_to_action`` / ``num_actions`` – policy
      vector encoding: ``row * width + col`` (length = ``height * width``).
    - Hexagonal GUI click detection and terminal ``"row,col"`` input parsing.

Usage
-----
    # Terminal mode
    python game_hex.py

    # pygame GUI mode
    python game_hex.py --gui
"""

from typing import List, Optional, Tuple
import argparse
import math
import numpy as np

from board_game import BoardGame, GameArgs, GameState, MoveAction, run_demo

# Board dimensions for standard Hex
HEX_SIZE: int = 11

# Default hex cell radius in pixels (distance from centre to vertex)
DEFAULT_HEX_RADIUS: int = 28

# Board cell values (0 = empty, 1 = red / first, 2 = blue / second)
BOARD_EMPTY: int = 0
BOARD_PLAYER_FIRST: int = 1   # red – connects top to bottom
BOARD_PLAYER_SECOND: int = 2  # blue – connects left to right

# Stone/cell fill colours indexed by board cell value
STONE_COLORS: Tuple = (
    (0, 0, 0),          # BOARD_EMPTY – unused placeholder
    (200, 50, 50),      # BOARD_PLAYER_FIRST  – red
    (50, 80, 200),      # BOARD_PLAYER_SECOND – blue
)

# Edge highlight colours (translucent borders indicating connection goals)
EDGE_COLOR_RED: Tuple[int, int, int] = (180, 60, 60)
EDGE_COLOR_BLUE: Tuple[int, int, int] = (60, 80, 180)

BOARD_BG_COLOR: Tuple[int, int, int] = (45, 52, 65)
CELL_EMPTY_COLOR: Tuple[int, int, int] = (220, 210, 190)
CELL_BORDER_COLOR: Tuple[int, int, int] = (80, 70, 50)

# All 6 hex neighbour directions (axial/offset-independent row,col deltas)
HEX_DIRECTIONS: List[Tuple[int, int]] = [
    (-1, 0), (-1, 1), (0, -1),
    (0, 1), (1, -1), (1, 0),
]


def _hex_center(row: int, col: int, hex_radius: int, offset_x: int, offset_y: int) -> Tuple[float, float]:
    """Compute pixel centre of a pointy-top hex at (row, col) using offset coords.

    Odd rows are shifted right by half the horizontal spacing.  This gives
    the classic Hex rhombus shape where each subsequent row is indented.

    For Hex specifically, we shift each row to the right by row * horiz/2
    to create the parallelogram/rhombus board shape.
    """
    horiz = hex_radius * math.sqrt(3)  # horizontal distance between centres
    vert = hex_radius * 1.5            # vertical distance between centres

    # Rhombus layout: each row shifts right by half horiz
    center_x = offset_x + col * horiz + row * (horiz * 0.5)
    center_y = offset_y + row * vert
    return center_x, center_y


def _hex_corners(center_x: float, center_y: float, radius: int) -> List[Tuple[float, float]]:
    """Return the 6 vertex coordinates of a pointy-top hexagon."""
    corners = []
    for i in range(6):
        angle_deg = 60 * i - 30  # pointy-top: first vertex at -30°
        angle_rad = math.radians(angle_deg)
        corners.append((
            center_x + radius * math.cos(angle_rad),
            center_y + radius * math.sin(angle_rad),
        ))
    return corners


def _pixel_to_hex(pixel_x: float, pixel_y: float, hex_radius: int,
                  offset_x: int, offset_y: int, height: int, width: int) -> Optional[Tuple[int, int]]:
    """Convert pixel coordinates to the nearest hex cell (row, col).

    Uses brute-force distance check against all cell centres – simple and
    robust for boards up to ~19×19.
    """
    best_dist = float("inf")
    best_cell = None
    for row in range(height):
        for col in range(width):
            cx, cy = _hex_center(row, col, hex_radius, offset_x, offset_y)
            dist = (pixel_x - cx) ** 2 + (pixel_y - cy) ** 2
            if dist < best_dist:
                best_dist = dist
                best_cell = (row, col)
    # Only accept if click is within the hex radius
    if best_cell is not None and best_dist <= (hex_radius * 0.9) ** 2:
        return best_cell
    return None


class HexArgs(GameArgs):
    """Argument container for Hex, with game-specific defaults."""

    def __init__(self, namespace: argparse.Namespace) -> None:
        super().__init__(namespace, width=HEX_SIZE, height=HEX_SIZE, title="Hex")
        # Override grid sizes to hex radius (used for board size calculation)
        self.hex_radius: int = getattr(namespace, "grid_h", DEFAULT_HEX_RADIUS)
        if self.hex_radius > 50:
            # If user passed default 80 (from GameArgs), use our own default
            self.hex_radius = DEFAULT_HEX_RADIUS
        self.grid_h = self.hex_radius
        self.grid_w = self.hex_radius


class HexGame(BoardGame):
    """Hex board game implementation built on top of ``BoardGame``.

    Rules
    -----
    - 11×11 rhombus board of hexagonal cells.
    - Player 1 (Red) connects the top edge to the bottom edge.
    - Player 2 (Blue) connects the left edge to the right edge.
    - Players alternate placing one stone per turn on any empty cell.
    - No draws are possible; exactly one player must win.
    - No pass action exists.
    """

    GUI_BOARD_BG_COLOR = BOARD_BG_COLOR

    @property
    def name(self) -> str:
        """Canonical short name of this game (e.g. ``"othello"``, ``"connect4"``)."""
        return "hex"

    @property
    def player_colors(self):
        return STONE_COLORS[1], STONE_COLORS[2]

    def __init__(self, args: HexArgs) -> None:
        super().__init__(args)
        self._hex_radius: int = args.hex_radius
        self._board_offset_x: int = 0
        self._board_offset_y: int = 0
        self.reset()

    # ------------------------------------------------------------------
    # Board size and rendering
    # ------------------------------------------------------------------

    def _compute_board_pixel_size(self) -> Tuple[int, int]:
        """Compute the bounding box of the hex rhombus board in pixels."""
        radius = self._hex_radius
        horiz = radius * math.sqrt(3)
        vert = radius * 1.5

        # Board spans: cols * horiz + (height-1) * horiz/2 for the row shift
        board_w = int((self.width - 1) * horiz + (self.height - 1) * (horiz * 0.5) + horiz) + 1
        board_h = int((self.height - 1) * vert + radius * 2) + 1
        return board_w, board_h

    def get_board_size(self, pos_x: int, pos_y: int, grid_h: int, grid_w: int) -> Tuple[int, int]:
        return self._compute_board_pixel_size()

    def draw_board(self) -> None:
        """Draw the hex board with coloured cells and edge indicators."""
        if self.gui_mode:
            self._draw_board_gui()
        else:
            self._draw_board_terminal()

    def _draw_board_gui(self) -> None:
        """Render the hexagonal board using pygame."""
        import pygame

        radius = self._hex_radius
        offset_x = self._gui_board_offset_x + int(radius * math.sqrt(3) * 0.5)
        offset_y = self._gui_board_offset_y + radius
        self._board_offset_x = offset_x
        self._board_offset_y = offset_y

        # Draw edge indicators (coloured borders showing connection goals)
        self._draw_edge_indicators(offset_x, offset_y, radius)

        # Draw each hex cell
        for row in range(self.height):
            for col in range(self.width):
                cx, cy = _hex_center(row, col, radius, offset_x, offset_y)
                corners = _hex_corners(cx, cy, radius)

                cell_value = int(self.state.board[row][col])
                if cell_value == BOARD_EMPTY:
                    fill_color = CELL_EMPTY_COLOR
                else:
                    fill_color = STONE_COLORS[cell_value]

                pygame.draw.polygon(self._surface, fill_color, corners)
                pygame.draw.polygon(self._surface, CELL_BORDER_COLOR, corners, 2)

    def _draw_edge_indicators(self, offset_x: int, offset_y: int, radius: int) -> None:
        """Draw coloured markers on the board edges to indicate connection goals."""
        import pygame

        marker_radius = max(4, radius // 6)

        # Top and bottom edges → Red (Player 1 connects top-bottom)
        for col in range(self.width):
            # Top edge
            cx, cy = _hex_center(0, col, radius, offset_x, offset_y)
            pygame.draw.circle(self._surface, EDGE_COLOR_RED, (int(cx), int(cy) - radius), marker_radius)
            # Bottom edge
            cx, cy = _hex_center(self.height - 1, col, radius, offset_x, offset_y)
            pygame.draw.circle(self._surface, EDGE_COLOR_RED, (int(cx), int(cy) + radius), marker_radius)

        # Left and right edges → Blue (Player 2 connects left-right)
        for row in range(self.height):
            # Left edge
            cx, cy = _hex_center(row, 0, radius, offset_x, offset_y)
            dx = radius * math.sqrt(3) * 0.5
            pygame.draw.circle(self._surface, EDGE_COLOR_BLUE, (int(cx - dx), int(cy)), marker_radius)
            # Right edge
            cx, cy = _hex_center(row, self.width - 1, radius, offset_x, offset_y)
            pygame.draw.circle(self._surface, EDGE_COLOR_BLUE, (int(cx + dx), int(cy)), marker_radius)

    def _draw_board_terminal(self) -> None:
        """Render the hex board as ASCII with rhombus indentation."""
        symbols = {BOARD_EMPTY: ".", BOARD_PLAYER_FIRST: "R", BOARD_PLAYER_SECOND: "B"}

        # Column header
        header = "   " + " ".join(f"{c:2d}" for c in range(self.width))
        print(header)

        for row in range(self.height):
            indent = " " * (row * 2)
            row_str = f"{indent}{row:2d}\\  "
            cells = []
            for col in range(self.width):
                cell_value = int(self.state.board[row][col])
                cells.append(symbols[cell_value])
            row_str += "  ".join(cells)
            row_str += f"  \\{row}"
            print(row_str)

        # Bottom header
        bottom_indent = " " * (self.height * 2)
        bottom = bottom_indent + "   " + " ".join(f"{c:2d}" for c in range(self.width))
        print(bottom)
        print("  Red (R): connect TOP ↔ BOTTOM")
        print("  Blue (B): connect LEFT ↔ RIGHT")

    # ------------------------------------------------------------------
    # Game logic
    # ------------------------------------------------------------------

    def move(self, player: int, action: MoveAction) -> bool:
        """Place a stone on the board at (row, col)."""
        row, col = action.row, action.col

        if self.state.board[row][col] != BOARD_EMPTY:
            return False

        board_value = BOARD_PLAYER_FIRST if player == self.PLAYER_FIRST else BOARD_PLAYER_SECOND
        self.state.board[row][col] = board_value
        self.last_action = action

        winner = self.check_winner()
        if winner is not None:
            self.switch_turn()
            self.end_game(winner)
        else:
            self.switch_turn()

        return True

    def get_legal_actions(self, player: int) -> List[MoveAction]:
        """Return all empty cells as legal actions."""
        legal_actions = []
        for row in range(self.height):
            for col in range(self.width):
                if self.state.board[row][col] == BOARD_EMPTY:
                    legal_actions.append(MoveAction(row=row, col=col, player=player))
        return legal_actions

    def check_winner(self) -> Optional[int]:
        """Check if either player has completed their connection.

        Player 1 (Red): connected path from any top-row cell to any bottom-row cell.
        Player 2 (Blue): connected path from any left-column cell to any right-column cell.
        """
        if self._check_connection(BOARD_PLAYER_FIRST):
            return self.PLAYER_FIRST
        if self._check_connection(BOARD_PLAYER_SECOND):
            return self.PLAYER_SECOND
        return None

    def _check_connection(self, board_value: int) -> bool:
        """DFS check whether board_value has a connected path across the board."""
        visited = set()
        stack = []

        if board_value == BOARD_PLAYER_FIRST:
            # Red connects top (row=0) to bottom (row=height-1)
            for col in range(self.width):
                if self.state.board[0][col] == board_value:
                    stack.append((0, col))
                    visited.add((0, col))
            target_row = self.height - 1
        else:
            # Blue connects left (col=0) to right (col=width-1)
            for row in range(self.height):
                if self.state.board[row][0] == board_value:
                    stack.append((row, 0))
                    visited.add((row, 0))
            target_col = self.width - 1

        while stack:
            row, col = stack.pop()

            # Check if we reached the target edge
            if board_value == BOARD_PLAYER_FIRST and row == target_row:
                return True
            if board_value == BOARD_PLAYER_SECOND and col == target_col:
                return True

            for delta_row, delta_col in HEX_DIRECTIONS:
                next_row = row + delta_row
                next_col = col + delta_col
                if (0 <= next_row < self.height
                        and 0 <= next_col < self.width
                        and (next_row, next_col) not in visited
                        and self.state.board[next_row][next_col] == board_value):
                    visited.add((next_row, next_col))
                    stack.append((next_row, next_col))

        return False

    def should_show_pass_button(self) -> bool:
        """Hex never requires a pass."""
        return False

    # ------------------------------------------------------------------
    # Policy vector encoding
    # ------------------------------------------------------------------

    def action_to_index(self, action: MoveAction) -> int:
        """Map a Hex action to a flat policy index: row * width + col."""
        return action.row * self.width + action.col

    def index_to_action(self, index: int, player: int) -> MoveAction:
        """Convert a flat policy index back to a MoveAction."""
        row = index // self.width
        col = index % self.width
        return MoveAction(row=row, col=col, player=player)

    @property
    def num_actions(self) -> int:
        """Hex has one action per board cell (no pass)."""
        return self.height * self.width

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset_state(self, state: GameState) -> GameState:
        """Reset to empty board with Player 1 (Red) to move first."""
        state.board = np.zeros((self.height, self.width), dtype=int)
        state.turn = self.PLAYER_FIRST
        state.winner = None
        state.total_turns = 0
        state.is_game_over = False
        return state

    # ------------------------------------------------------------------
    # GUI interaction
    # ------------------------------------------------------------------

    def draw_last_move_marker(self, action: MoveAction) -> None:
        """Draw a highlighted border around the last-placed hex cell."""
        import pygame
        row, col = action.row, action.col
        radius = self._hex_radius
        cx, cy = _hex_center(row, col, radius, self._board_offset_x, self._board_offset_y)
        corners = _hex_corners(cx, cy, radius)
        highlight_color = (255, 255, 0)  # bright yellow outline
        pygame.draw.polygon(self._surface, highlight_color, corners, 3)

    def handle_board_click(
        self, pixel_x: int, pixel_y: int, grid_h: int, grid_w: int
    ) -> Optional[MoveAction]:
        """Convert a GUI click to a hex cell MoveAction using distance-based hit testing."""
        cell = _pixel_to_hex(
            pixel_x, pixel_y,
            self._hex_radius,
            self._board_offset_x, self._board_offset_y,
            self.height, self.width,
        )
        if cell is None:
            return None
        row, col = cell
        if self.state.board[row][col] != BOARD_EMPTY:
            return None
        return MoveAction(row=row, col=col, player=self.state.turn)

    # ------------------------------------------------------------------
    # Terminal interaction
    # ------------------------------------------------------------------

    def parse_terminal_input(self, user_input: str) -> Optional[MoveAction]:
        """Parse ``row,col`` terminal input into a MoveAction."""
        try:
            parts = user_input.replace(" ", "").split(",")
            row, col = int(parts[0]), int(parts[1])
        except (ValueError, IndexError):
            print("Please enter row,col (e.g. 3,4).")
            return None

        if not (0 <= row < self.height and 0 <= col < self.width):
            print(f"Out of bounds. Row: 0-{self.height-1}, Col: 0-{self.width-1}.")
            return None

        if self.state.board[row][col] != BOARD_EMPTY:
            print("Cell already occupied.")
            return None

        return MoveAction(row=row, col=col, player=self.state.turn)


if __name__ == "__main__":
    _parser = argparse.ArgumentParser(description="Hex")
    GameArgs.add_common_args(_parser)
    run_demo(HexArgs(_parser.parse_args()), HexGame)
