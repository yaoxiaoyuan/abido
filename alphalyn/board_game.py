"""board_game.py – Abstract base classes and shared utilities for board games.

This module defines the core abstractions shared by all game implementations
in this project:

Classes
-------
MoveAction
    A data class representing a single player action (target row/col, acting
    player, and an optional ``extra`` payload for game-specific metadata such
    as a pass flag).
GameArgs
    A generic argument container used to configure any ``BoardGame`` instance
    (board dimensions, GUI mode, grid cell size, window title).
GameState
    A snapshot of the complete board state at one point in time: the numpy
    board array, whose turn it is, the winner (if any), and whether the game
    has ended.  Supports ``clone()`` for safe MCTS tree search.
BoardGame
    The abstract base class every concrete game must subclass.  Declares the
    interface for move execution, legal-action enumeration, winner detection,
    GUI/terminal rendering, and the AlphaZero policy-vector encoding
    (``action_to_index`` / ``index_to_action`` / ``num_actions``).

Helpers
-------
run_demo / run_gui_demo / run_terminal_demo
    Generic game-loop runners that work with any ``BoardGame`` subclass,
    providing both a pygame GUI mode and a text-only terminal mode.

Usage
-----
    # Terminal demo (no pygame required)
    from board_game import run_demo
    from game_othello import OthelloGame, OthelloArgs
    run_demo(OthelloArgs(gui_mode=False), OthelloGame)

    # GUI demo
    run_demo(OthelloArgs(gui_mode=True), OthelloGame)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Any
import argparse
import copy

import numpy as np

try:
    import pygame
    GUI_MODE_AVAILABLE: bool = True
except ImportError:
    GUI_MODE_AVAILABLE: bool = False


@dataclass
class MoveAction:
    """Represents a single move action in a board game.

    Attributes
    ----------
    row:
        Target row index (0-indexed from top).
    col:
        Target column index (0-indexed from left).
    player:
        The player making the move (``PLAYER_FIRST`` or ``PLAYER_SECOND``).
    extra:
        Optional game-specific payload (e.g. piece type, promotion flag).
    """

    row: int
    col: int
    player: int
    extra: Any = None


class GameArgs:
    """Generic argument container for any board game.

    Instances are built from an ``argparse.Namespace`` returned by a parser
    that has been extended with :meth:`add_common_args`.  Game-specific
    subclasses call ``super().__init__(namespace)`` and then overwrite the
    fixed board dimensions / title with their own constants.

    Attributes
    ----------
    width, height:
        Board dimensions in cells.
    gui_mode:
        ``True`` → pygame GUI; ``False`` → terminal.
    grid_h, grid_w:
        Pixel height / width of one grid cell (GUI only).
    title:
        Window caption (GUI only).
    mode:
        Play mode: ``"hum"`` | ``"ai"`` | ``"hum-ai"``.
    model_path:
        Shared / fallback checkpoint path for AI modes.
    model_path_first, model_path_second:
        Per-side checkpoints for ai-vs-ai (fall back to *model_path*).
    num_simulations:
        MCTS rollouts per move.
    ai_plays_first:
        hum-ai only – ``True`` → AI is ``PLAYER_FIRST``.
    num_filters, num_residual_blocks, value_head_hidden_size:
        ResNet architecture hyper-parameters forwarded to ``ResNetConfig``.
    """

    @staticmethod
    def add_common_args(parser: argparse.ArgumentParser) -> None:
        """Register all common board-game arguments on *parser*.

        Call this from each game's ``if __name__ == "__main__"`` block so that
        every game binary exposes the full set of options.
        """
        parser.add_argument("--gui", action="store_true",
                            help="Launch in pygame GUI mode")
        parser.add_argument("--grid-h", type=int, default=80,
                            help="Pixel height of one grid cell (default: 80)")
        parser.add_argument("--grid-w", type=int, default=80,
                            help="Pixel width of one grid cell (default: 80)")
        parser.add_argument("--mode", type=str, default="hum",
                            choices=["hum", "ai", "hum-ai"],
                            help="Play mode: hum (default) | ai | hum-ai")
        parser.add_argument("--model-path", type=str, default="",
                            help="Checkpoint path for single-AI or shared fallback")
        parser.add_argument("--model-path-first", type=str, default="",
                            help="ai-vs-ai: checkpoint for PLAYER_FIRST")
        parser.add_argument("--model-path-second", type=str, default="",
                            help="ai-vs-ai: checkpoint for PLAYER_SECOND")
        parser.add_argument("--sims", type=int, default=400,
                            help="MCTS simulations per move (default: 400)")
        parser.add_argument("--ai-plays-first", action="store_true",
                            help="hum-ai: AI takes PLAYER_FIRST instead of PLAYER_SECOND")
        parser.add_argument("--mcts-batch", type=int, default=4,
                            help="MCTS inference batch size (default: 4)")
        parser.add_argument("--filters", type=int, default=128,
                            help="ResNet filter count; must match checkpoint (default: 128)")
        parser.add_argument("--blocks", type=int, default=10,
                            help="Number of residual blocks; must match checkpoint (default: 10)")
        parser.add_argument("--value-hidden", type=int, default=256,
                            help="Value head hidden size; must match checkpoint (default: 256)")
        parser.add_argument("--device", type=str, default="cpu",
                            choices=["cpu", "cuda", "mps"],
                            help="Device for model inference (default: cpu)")

    def __init__(self, namespace: argparse.Namespace,
                 width: int = 0, height: int = 0, title: str = "Board Game") -> None:
        """Initialise from a parsed ``argparse.Namespace``.

        Parameters
        ----------
        namespace:
            Result of ``parser.parse_args()`` after calling
            :meth:`add_common_args` on the parser.
        width, height:
            Board dimensions; subclasses pass their own constants here.
        title:
            Window caption; subclasses pass their own game name here.
        """
        # Fixed board geometry supplied by the subclass.
        self.width:  int = width
        self.height: int = height
        self.title:  str = title

        # argparse stores hyphenated flags as underscore attributes.
        self.gui_mode:              bool = getattr(namespace, "gui",                    False)
        self.grid_h:                int  = getattr(namespace, "grid_h",                 80)
        self.grid_w:                int  = getattr(namespace, "grid_w",                 80)
        self.mode:                  str  = getattr(namespace, "mode",                   "hum")
        self.model_path:            str  = getattr(namespace, "model_path",             "")
        self.model_path_first:      str  = getattr(namespace, "model_path_first",       "")
        self.model_path_second:     str  = getattr(namespace, "model_path_second",      "")
        self.sims:                  int  = getattr(namespace, "sims",                   400)
        self.ai_plays_first:        bool = getattr(namespace, "ai_plays_first",         False)
        self.mcts_batch:            int  = getattr(namespace, "mcts_batch",             64)
        self.num_filters:           int  = getattr(namespace, "filters",                 128)
        self.num_residual_blocks:   int  = getattr(namespace, "blocks",                  10)
        self.value_head_hidden_size: int = getattr(namespace, "value_hidden",            256)
        self.device:                 str  = getattr(namespace, "device",                "cpu")


@dataclass
class GameState:
    """Holds the complete state of a board game at a given moment."""

    board: Optional[np.ndarray] = None  # shape (height, width), dtype int
    turn: Optional[int] = None
    # 0 for first player wins, 1 for second player wins, -1 for tie
    winner: Optional[int] = None
    total_turns: int = 0
    is_game_over: bool = False

    def clone(self) -> "GameState":
        """Return a deep copy of this state, useful for MCTS / AI search."""
        return copy.deepcopy(self)


class BoardGame(ABC):
    """Abstract base class for all board games.

    Subclasses must implement the abstract methods to define game-specific
    rendering, board layout, and move logic.

    Coordinate convention
    ---------------------
    - ``row`` / ``i`` : 0-indexed row from the top
    - ``col`` / ``j`` : 0-indexed column from the left
    - ``pos_x``       : pixel x-coordinate (horizontal)
    - ``pos_y``       : pixel y-coordinate (vertical)
    """

    # Player constants for readability
    PLAYER_FIRST: int = 0
    PLAYER_SECOND: int = 1
    RESULT_TIE: int = -1

    # ------------------------------------------------------------------
    # GUI layout constants (pixels) – subclasses may override
    # ------------------------------------------------------------------
    GUI_INFO_BAR_HEIGHT: int = 80       # height of the info panel (enough for dot + two text rows)
    GUI_BUTTON_HEIGHT: int = 36         # height of each button
    GUI_BUTTON_WIDTH: int = 150         # width of each button
    GUI_FONT_SIZE: int = 20             # font size for info / button labels
    GUI_BG_COLOR: Tuple[int, int, int] = (45, 52, 65)           # deep blue-grey window background
    GUI_INFO_BG_COLOR: Tuple[int, int, int] = (62, 72, 90)      # medium blue-grey info panel
    GUI_TEXT_COLOR: Tuple[int, int, int] = (235, 235, 235)      # near-white text
    GUI_BUTTON_COLOR: Tuple[int, int, int] = (70, 130, 180)
    GUI_BUTTON_HOVER_COLOR: Tuple[int, int, int] = (100, 160, 210)
    GUI_BUTTON_TEXT_COLOR: Tuple[int, int, int] = (255, 255, 255)
    GUI_OVERLAY_COLOR: Tuple[int, int, int, int] = (0, 0, 0, 160)  # RGBA
    # Board background colour for stone-in-cell layout (subclasses should override).
    GUI_BOARD_BG_COLOR: Tuple[int, int, int] = (60, 60, 60)
    # Gap between the board area and the window edges (pixels).
    GUI_BOARD_PADDING: int = 16

    def __init__(self, args: Any) -> None:
        self.width: int = args.width
        self.height: int = args.height
        self.gui_mode: bool = args.gui_mode
        self.state: GameState = GameState()
        self.last_action: Optional[MoveAction] = None  # tracks the most recent move for highlighting

        # pygame resources – only initialised in GUI mode
        self._surface: Any = None   # pygame.Surface (main window)
        self._font: Any = None      # pygame.font.Font
        self._show_reset_panel: bool = False  # whether the reset overlay is visible
        self._gui_board_w: int = 0  # cached board pixel width, set by run_gui_demo
        self._gui_board_offset_x: int = 0  # board left edge in window pixels
        self._gui_board_offset_y: int = 0  # board top edge in window pixels

        # Mode info for draw_info() label rendering.
        self._game_mode: str = getattr(args, "mode", "hum")
        self._is_first_player_ai: bool = getattr(args, "ai_plays_first", False)

    # ------------------------------------------------------------------
    # Player colour and label helpers
    # ------------------------------------------------------------------

    @property
    def player_colors(self) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
        """Return ``(player1_color, player2_color)`` as RGB tuples.

        The base implementation returns neutral greys.  Every concrete game
        subclass should override this to return the same colours used in
        ``draw_grids_and_stones()`` (i.e. ``STONE_COLORS[1]`` and
        ``STONE_COLORS[2]``).
        """
        return (180, 180, 180), (80, 80, 80)

    def get_player_label(self, player: int) -> str:
        """Return a display label for *player* that is aware of the play mode.

        - ``"hum"`` / ``"ai"`` modes → ``"Player 1"`` / ``"Player 2"``.
        - ``"hum-ai"`` mode → ``"AI"`` or ``"Human"`` depending on which side
          each player-id maps to.

        Parameters
        ----------
        player:
            ``PLAYER_FIRST`` or ``PLAYER_SECOND``.
        """
        if self._game_mode == "hum-ai":
            is_ai = (
                (player == self.PLAYER_FIRST  and     self._is_first_player_ai) or
                (player == self.PLAYER_SECOND and not self._is_first_player_ai)
            )
            return "AI" if is_ai else "Human"
        return "Player 1" if player == self.PLAYER_FIRST else "Player 2"

    # ------------------------------------------------------------------
    # Abstract interface – subclasses must implement these
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Canonical short name of this game (e.g. ``"othello"``, ``"connect4"``)."""

    @abstractmethod
    def get_board_size(self, pos_x: int, pos_y: int, grid_h: int, grid_w: int) -> Tuple[int, int]:
        """Return (board_pixel_width, board_pixel_height) for GUI rendering."""

    def get_window_size(self, pos_x: int, pos_y: int, grid_h: int, grid_w: int) -> Tuple[int, int]:
        """Return (window_pixel_width, window_pixel_height) for GUI rendering.

        Layout
        ------
        The window is divided into two columns:

        - **Left**: the board area, sized by ``get_board_size()``.
        - **Right**: a vertical stack of UI components from top to bottom:
            1. Info bar  (``GUI_INFO_BAR_HEIGHT``)
            2. Reset button  (always visible)
            3. Pass button   (only when ``should_show_pass_button()`` is True)
            4. Restart button (only when game is over)

        The window height equals ``max(board_height + info_bar, right_panel_height)``
        so neither side is clipped.
        """
        board_w, board_h = self.get_board_size(pos_x, pos_y, grid_h, grid_w)

        bp = self.GUI_BOARD_PADDING  # gap between board and window edges
        panel_margin = 16
        right_panel_width = self.GUI_BUTTON_WIDTH + panel_margin * 2

        # Board area: board + padding on all four sides
        total_board_width  = board_w + bp * 2
        total_board_height = board_h + bp * 2

        # Right panel: info block + 2 buttons (reset + pass worst case)
        num_buttons = 2
        right_panel_height = (
            panel_margin
            + self.GUI_INFO_BAR_HEIGHT
            + panel_margin * 2
            + num_buttons * (self.GUI_BUTTON_HEIGHT + panel_margin)
        )

        window_width  = total_board_width + right_panel_width
        window_height = max(total_board_height, right_panel_height)
        return window_width, window_height

    @abstractmethod
    def draw_board(self) -> None:
        """Draw the bare board (lines, background, etc.) without any pieces."""

    @abstractmethod
    def move(self, player: int, action: MoveAction) -> bool:
        """Apply *action* for *player* and update ``self.state``.

        Parameters
        ----------
        player:
            The acting player (``PLAYER_FIRST`` or ``PLAYER_SECOND``).
        action:
            A ``MoveAction`` describing the target cell and any extra payload.

        Returns
        -------
        bool
            ``True`` if the move was legal and applied, ``False`` otherwise.
        """

    @abstractmethod
    def get_legal_actions(self, player: int) -> List[Any]:
        """Return all legal actions available to *player* in the current state."""

    @abstractmethod
    def should_show_pass_button(self) -> bool:
        """Return whether the current player has no legal moves and must pass.

        Used by ``get_window_size()`` to decide whether to allocate space for
        the Pass button, and by ``draw_pass_button()`` to decide whether to
        render / print it.
        """

    @abstractmethod
    def check_winner(self) -> Optional[int]:
        """Check the current board for a winner.

        Returns
        -------
        int or None
            ``PLAYER_FIRST`` (0), ``PLAYER_SECOND`` (1), ``RESULT_TIE`` (-1),
            or ``None`` if the game is still ongoing.
        """

    @abstractmethod
    def handle_board_click(self, pixel_x: int, pixel_y: int, grid_h: int, grid_w: int) -> Optional[MoveAction]:
        """Translate a GUI left-click into a ``MoveAction`` (GUI mode only).

        Called by ``run_gui_demo`` whenever the player clicks inside the board
        area.  Should return a legal ``MoveAction`` or ``None`` to ignore the
        click (e.g. the click is outside any valid cell / intersection).

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
        """

    @abstractmethod
    def parse_terminal_input(self, user_input: str) -> Optional[MoveAction]:
        """Parse a raw terminal input string into a ``MoveAction`` (terminal mode only).

        Called by ``run_terminal_demo`` for each line the player types.
        Should return a legal ``MoveAction`` or ``None`` if the input is
        invalid (the loop will prompt again without advancing the game).

        Parameters
        ----------
        user_input:
            Raw string from ``input()``, already stripped of leading/trailing
            whitespace.
        """

    @abstractmethod
    def action_to_index(self, action: "MoveAction") -> int:
        """Convert a ``MoveAction`` to a flat integer index for the policy vector.

        The mapping is game-specific:

        - Games where every cell is a valid target (Gomoku, Othello) typically
          use ``row * width + col``, with an extra slot at the end for pass.
        - Games where only one dimension matters (Connect Four) use just
          ``col``, so the policy vector has length ``width`` (+ 1 for pass if
          needed).

        The pass action (``action.extra == "pass"``) must map to the last
        index, i.e. ``num_actions - 1``.

        Returns
        -------
        int
            A non-negative index in ``[0, num_actions)``.
        """

    @abstractmethod
    def index_to_action(self, index: int, player: int) -> "MoveAction":
        """Convert a flat policy index back to a ``MoveAction``.

        The inverse of ``action_to_index``.  The last index must map to the
        pass action (``MoveAction(row=-1, col=-1, player=player, extra="pass")``).

        Parameters
        ----------
        index:
            A flat policy index in ``[0, num_actions)``.
        player:
            The player for whom the action is constructed.

        Returns
        -------
        MoveAction
        """

    @property
    @abstractmethod
    def num_actions(self) -> int:
        """Total number of distinct actions in the policy vector.

        Must equal ``max(action_to_index(a)) + 1`` over all possible actions
        (including pass).  MCTS and the neural network use this to size the
        policy head output.
        """

    # ------------------------------------------------------------------
    # Concrete drawing helpers – subclasses may override for custom look
    # ------------------------------------------------------------------

    def init_gui(self, surface: Any) -> None:
        """Bind a pygame surface and initialise font resources.

        Must be called once after ``pygame.init()`` and before the first
        ``render()`` in GUI mode.

        Parameters
        ----------
        surface:
            The main ``pygame.Surface`` (i.e. the window surface returned by
            ``pygame.display.set_mode()``).
        """
        import pygame  # local import so non-GUI users don't need pygame
        self._surface = surface
        self._font = pygame.font.SysFont("Arial", self.GUI_FONT_SIZE)

    # ------------------------------------------------------------------
    # Internal GUI helpers
    # ------------------------------------------------------------------

    def _draw_text_centered(
        self,
        text: str,
        rect: Any,
        text_color: Tuple[int, int, int],
    ) -> None:
        """Render *text* centred inside *rect* on ``self._surface``."""
        text_surface = self._font.render(text, True, text_color)
        text_rect = text_surface.get_rect(center=rect.center)
        self._surface.blit(text_surface, text_rect)

    def _draw_button(
        self,
        label: str,
        rect: Any,
        *,
        bg_color: Optional[Tuple[int, int, int]] = None,
    ) -> None:
        """Draw a filled rounded rectangle button with centred label text."""
        import pygame
        color = bg_color if bg_color is not None else self.GUI_BUTTON_COLOR
        pygame.draw.rect(self._surface, color, rect, border_radius=6)
        self._draw_text_centered(label, rect, self.GUI_BUTTON_TEXT_COLOR)

    # ------------------------------------------------------------------
    # Concrete drawing helpers – subclasses may override for custom look
    # ------------------------------------------------------------------

    def draw_info(self) -> None:
        """Draw game info in the right panel: turns, current player, and winner.

        - **Terminal mode**: prints a one-line summary to stdout, using mode-aware
          player labels (e.g. ``"AI"`` / ``"Human"`` in hum-ai mode).
        - **GUI mode**: renders an info block at the top of the **right** panel,
          positioned to the right of the board.  The current player's colour is
          shown as a filled circle next to the player label.  The board pixel width
          is cached in ``self._gui_board_w`` by ``run_gui_demo`` before this is
          called.
        """
        if self.gui_mode:
            import pygame
            panel_margin = 16
            panel_x = self._gui_board_offset_x + self._gui_board_w + panel_margin
            panel_w = self.GUI_BUTTON_WIDTH

            # Info panel background with rounded corners.
            info_rect = pygame.Rect(panel_x, panel_margin, panel_w, self.GUI_INFO_BAR_HEIGHT)
            pygame.draw.rect(self._surface, self.GUI_INFO_BG_COLOR, info_rect, border_radius=10)

            # Determine what to display.
            p1_color, p2_color = self.player_colors
            if self.state.is_game_over:
                if self.state.winner == self.RESULT_TIE:
                    label_top = f"Turn {self.state.total_turns}"
                    label_bottom = "Tie!"
                    dot_color: Optional[Tuple[int, int, int]] = None
                else:
                    winner_label = self.get_player_label(self.state.winner)
                    label_top = "Winner"
                    label_bottom = winner_label
                    dot_color = p1_color if self.state.winner == self.PLAYER_FIRST else p2_color
            else:
                current_label = self.get_player_label(self.state.turn)
                label_top = f"Turn {self.state.total_turns}"
                label_bottom = current_label
                dot_color = p1_color if self.state.turn == self.PLAYER_FIRST else p2_color

            # Layout: colour circle on the left, two text lines stacked on the right.
            inner_pad = 10        # padding inside the panel
            circle_radius = self.GUI_INFO_BAR_HEIGHT // 2 - inner_pad
            circle_x = panel_x + inner_pad + circle_radius
            circle_y = panel_margin + self.GUI_INFO_BAR_HEIGHT // 2

            if dot_color is not None:
                # Fill circle.
                pygame.draw.circle(self._surface, dot_color, (circle_x, circle_y), circle_radius)
                # Double-border trick: outer bright ring + inner dark ring so both
                # black and white stones are visible on any background.
                pygame.draw.circle(self._surface, (255, 255, 255), (circle_x, circle_y), circle_radius, 2)
                pygame.draw.circle(self._surface, (30, 30, 30), (circle_x, circle_y), circle_radius - 2, 1)

            # Text area starts to the right of the circle.
            text_x = circle_x + circle_radius + inner_pad
            text_w = panel_x + panel_w - text_x - inner_pad
            row_h = (self.GUI_INFO_BAR_HEIGHT - inner_pad * 2) // 2
            top_rect    = pygame.Rect(text_x, panel_margin + inner_pad,           text_w, row_h)
            bottom_rect = pygame.Rect(text_x, panel_margin + inner_pad + row_h,   text_w, row_h)

            # Top row in muted colour, bottom row brighter for emphasis.
            muted_color = (180, 185, 195)
            self._draw_text_centered(label_top,    top_rect,    muted_color)
            self._draw_text_centered(label_bottom, bottom_rect, self.GUI_TEXT_COLOR)
            return

        # ---- Terminal mode ----
        current_label = self.get_player_label(self.state.turn)
        print(f"Turn {self.state.total_turns} | Current: {current_label}", end="")
        if self.state.is_game_over:
            if self.state.winner == self.RESULT_TIE:
                print(" | Result: Tie")
            else:
                winner_label = self.get_player_label(self.state.winner)
                print(f" | Winner: {winner_label}")
        else:
            print()

    def draw_pass_button(self) -> None:
        """Draw a terminal prompt when the current player must pass.

        In GUI mode the Pass button is rendered directly by ``run_gui_demo``
        so no drawing happens here.

        - **Terminal mode**: prints ``[P] Pass turn`` to stdout when needed.
        - **GUI mode**: no-op (handled by ``run_gui_demo``).
        """
        if not self.gui_mode and self.should_show_pass_button():
            print("[P] Pass turn")

    def draw_reset_panel(self) -> None:
        """Draw the reset overlay (terminal) or nothing (GUI – handled by run_gui_demo).

        In GUI mode the reset overlay is drawn directly inside ``run_gui_demo``
        so it can receive click events.  This method only handles terminal output.
        """
        if not self.gui_mode:
            print("[r] Restart")

    def toggle_reset_panel(self) -> None:
        """Show or hide the GUI reset panel overlay."""
        self._show_reset_panel = not self._show_reset_panel

    def draw_grids(self, pos_x: int, pos_y: int, grid_h: int, grid_w: int) -> None:
        """Draw a grid of alternating light/dark squares starting at pixel (pos_x, pos_y).

        Useful when rendering with square-cell grids (e.g. Chess, Checkers).
        Draws filled rectangles with a 1-pixel border, mimicking a standard chessboard pattern.
        """
        LIGHT_COLOR = (240, 217, 181)
        DARK_COLOR = (181, 136, 99)
        BORDER_COLOR = (80, 80, 80)

        for row in range(self.height):
            for col in range(self.width):
                cell_x = pos_x + col * grid_w
                cell_y = pos_y + row * grid_h
                cell_rect = pygame.Rect(cell_x, cell_y, grid_w, grid_h)
                fill_color = LIGHT_COLOR if (row + col) % 2 == 0 else DARK_COLOR
                pygame.draw.rect(self._surface, fill_color, cell_rect)
                pygame.draw.rect(self._surface, BORDER_COLOR, cell_rect, 1)

    def get_grid_board_size(
        self,
        grid_h: int,
        grid_w: int,
        stone_on_intersection: bool = False,
        padding: int = 20,
    ) -> Tuple[int, int]:
        """Return the pixel (board_width, board_height) for a grid-style board.

        The calculation differs depending on where stones are placed:

        - **stone_in_cell** (``stone_on_intersection=False``): stones sit inside
          each cell (e.g. Chess, Checkers, Gomoku on square board).
          The board spans exactly ``width`` columns × ``height`` rows of cells.

          .. code-block:: text

              board_width  = width  * grid_w
              board_height = height * grid_h

        - **stone_on_intersection** (``stone_on_intersection=True``): stones sit
          on grid-line intersections (e.g. Go, Gomoku on Go board).
          There are ``width`` vertical lines and ``height`` horizontal lines,
          so the board spans ``(width-1)`` gaps plus padding on both sides.

          .. code-block:: text

              board_width  = (width  - 1) * grid_w + padding * 2
              board_height = (height - 1) * grid_h + padding * 2

        Parameters
        ----------
        grid_h:
            Height of one grid cell (or vertical gap between intersections) in pixels.
        grid_w:
            Width of one grid cell (or horizontal gap between intersections) in pixels.
        stone_on_intersection:
            ``False`` → stone-in-cell layout; ``True`` → stone-on-intersection layout.
        padding:
            Extra margin around the outermost grid lines (only used when
            ``stone_on_intersection=True``).

        Returns
        -------
        Tuple[int, int]
            ``(board_pixel_width, board_pixel_height)``
        """
        if stone_on_intersection:
            board_w = (self.width - 1) * grid_w + padding * 2
            board_h = (self.height - 1) * grid_h + padding * 2
        else:
            board_w = self.width * grid_w
            board_h = self.height * grid_h
        return board_w, board_h

    def draw_grids_and_stones(
        self,
        pos_x: int,
        pos_y: int,
        grid_h: int,
        grid_w: int,
        stone_colors: Tuple,
        stone_on_intersection: bool = False,
        padding: int = 20,
    ) -> None:
        """Draw the grid board and all stones in one call.

        Supports two stone-placement styles and both GUI / terminal modes.

        Stone placement styles
        ----------------------
        - **stone_in_cell** (``stone_on_intersection=False``):
          Each stone is centred inside its grid cell.  The stone radius is
          ``min(grid_h, grid_w) // 2 - 2`` so it fits with a small margin.
          Grid cells are drawn as alternating light/dark squares (chessboard).

        - **stone_on_intersection** (``stone_on_intersection=True``):
          Each stone is centred on the intersection of grid lines (Go style).
          Grid lines are drawn as thin lines on a uniform background.
          The stone radius is ``min(grid_h, grid_w) // 2 - 1``.

        Parameters
        ----------
        pos_x:
            Pixel x-coordinate of the board origin (top-left corner).
        pos_y:
            Pixel y-coordinate of the board origin (top-left corner).
        grid_h:
            Height of one grid cell / vertical gap between intersections.
        grid_w:
            Width of one grid cell / horizontal gap between intersections.
        stone_colors:
            A tuple of ``(R, G, B)`` colours **indexed by board cell value**.
            Index 0 is reserved for empty cells (unused); index 1 is the
            colour for the first non-empty board value, index 2 for the
            second, etc.  Subclasses define their own board cell constants
            (e.g. ``BOARD_PLAYER_FIRST = 1``, ``BOARD_PLAYER_SECOND = 2``)
            to avoid collision with the empty-cell sentinel ``0``.
        stone_on_intersection:
            ``False`` → stone-in-cell; ``True`` → stone-on-intersection.
        padding:
            Board padding used only when ``stone_on_intersection=True``.
            Must match the value passed to ``get_grid_board_size()``.
        """
        if self.gui_mode:
            self._draw_grids_and_stones_gui(
                pos_x, pos_y, grid_h, grid_w,
                stone_colors, stone_on_intersection, padding,
            )
        else:
            self._draw_grids_and_stones_terminal(stone_colors, stone_on_intersection)

    def _draw_grids_and_stones_gui(
        self,
        pos_x: int,
        pos_y: int,
        grid_h: int,
        grid_w: int,
        stone_colors: Tuple[Tuple[int, int, int], Tuple[int, int, int]],
        stone_on_intersection: bool,
        padding: int,
    ) -> None:
        """Internal: pygame rendering for ``draw_grids_and_stones``."""
        board_w, board_h = self.get_grid_board_size(grid_h, grid_w, stone_on_intersection, padding)

        if stone_on_intersection:
            # Uniform board background – colour comes from the subclass attribute
            # so each game can define its own board colour (e.g. green for Othello).
            LINE_COLOR = (60, 40, 10)
            board_rect = pygame.Rect(pos_x, pos_y, board_w, board_h)
            pygame.draw.rect(self._surface, self.GUI_BOARD_BG_COLOR, board_rect)

            # Draw grid lines
            for row in range(self.height):
                line_y = pos_y + padding + row * grid_h
                pygame.draw.line(
                    self._surface, LINE_COLOR,
                    (pos_x + padding, line_y),
                    (pos_x + padding + (self.width - 1) * grid_w, line_y),
                    1,
                )
            for col in range(self.width):
                line_x = pos_x + padding + col * grid_w
                pygame.draw.line(
                    self._surface, LINE_COLOR,
                    (line_x, pos_y + padding),
                    (line_x, pos_y + padding + (self.height - 1) * grid_h),
                    1,
                )

            # Draw stones on intersections
            stone_radius = min(grid_h, grid_w) // 2 - 1
            for row in range(self.height):
                for col in range(self.width):
                    cell_value = int(self.state.board[row][col])
                    if cell_value == 0:  # 0 is always the empty sentinel
                        continue
                    stone_x = pos_x + padding + col * grid_w
                    stone_y = pos_y + padding + row * grid_h
                    self.draw_stone(stone_x, stone_y, stone_radius, stone_colors[cell_value])

        else:
            # Stone-in-cell layout.  The board background colour is taken from
            # ``GUI_BOARD_BG_COLOR`` so each subclass can define its own colour
            # by overriding that class attribute (e.g. blue for Connect Four).
            BORDER_COLOR = (80, 80, 80)
            stone_radius = min(grid_h, grid_w) // 2 - 2

            # Fill the whole board area with the game's background colour first.
            board_rect = pygame.Rect(pos_x, pos_y, board_w, board_h)
            pygame.draw.rect(self._surface, self.GUI_BOARD_BG_COLOR, board_rect)

            for row in range(self.height):
                for col in range(self.width):
                    cell_x = pos_x + col * grid_w
                    cell_y = pos_y + row * grid_h
                    cell_rect = pygame.Rect(cell_x, cell_y, grid_w, grid_h)
                    pygame.draw.rect(self._surface, BORDER_COLOR, cell_rect, 1)

                    cell_value = int(self.state.board[row][col])
                    if cell_value == 0:  # 0 is always the empty sentinel
                        continue
                    stone_x = cell_x + grid_w // 2
                    stone_y = cell_y + grid_h // 2
                    self.draw_stone(stone_x, stone_y, stone_radius, stone_colors[cell_value])

    def _draw_grids_and_stones_terminal(
        self,
        stone_colors: Tuple,
        stone_on_intersection: bool,
    ) -> None:
        """Internal: ASCII rendering for ``draw_grids_and_stones``."""
        # Symbols indexed by board cell value: 0 = empty, 1 = first, 2 = second.
        # Subclasses may use different board values; any non-zero value gets a symbol.
        board_symbols = {1: "●", 2: "○"}
        empty_symbol = "·" if stone_on_intersection else " "

        if stone_on_intersection:
            # Print column indices header
            col_header = "  " + "  ".join(f"{col:2d}" for col in range(self.width))
            print(col_header)
            for row in range(self.height):
                row_symbols = []
                for col in range(self.width):
                    cell_value = int(self.state.board[row][col])
                    row_symbols.append(board_symbols.get(cell_value, empty_symbol))
                connector = "--"
                print(f"{row:2d} " + connector.join(row_symbols))
        else:
            horizontal_border = "+" + ("---+" * self.width)
            for row in range(self.height):
                print(horizontal_border)
                row_cells = "|"
                for col in range(self.width):
                    cell_value = int(self.state.board[row][col])
                    symbol = board_symbols.get(cell_value, empty_symbol)
                    row_cells += f" {symbol} |"
                print(row_cells)
            print(horizontal_border)

    def draw_stone(self, pos_x: int, pos_y: int, radius: int, color: Tuple[int, int, int]) -> None:
        """Draw a circular stone centred at pixel (pos_x, pos_y).

        Useful for Go-style games (e.g. Go, Gomoku, Othello).
        Draws a filled circle with a 1-pixel darker border for depth.

        Parameters
        ----------
        pos_x:
            Pixel x-coordinate of the stone centre.
        pos_y:
            Pixel y-coordinate of the stone centre.
        radius:
            Stone radius in pixels.
        color:
            Fill colour as an ``(R, G, B)`` tuple.
        """
        pygame.draw.circle(self._surface, color, (pos_x, pos_y), radius)
        border_color = tuple(max(0, channel - 60) for channel in color)
        pygame.draw.circle(self._surface, border_color, (pos_x, pos_y), radius, 1)

    # ------------------------------------------------------------------
    # Game-loop helpers
    # ------------------------------------------------------------------

    def render(self) -> None:
        """Render the complete current game state.

        Delegates all board and piece drawing to ``draw_board()``, which
        subclasses are responsible for implementing fully.
        """
        self.draw_board()
        if self.gui_mode and self.last_action is not None and self.last_action.row >= 0:
            self.draw_last_move_marker(self.last_action)
        self.draw_info()
        self.draw_pass_button()
        self.draw_reset_panel()

    @abstractmethod
    def draw_last_move_marker(self, action: MoveAction) -> None:
        """Draw a visual marker on the cell/intersection of the last move (GUI only).

        Called by ``render()`` after ``draw_board()`` when ``self.last_action``
        is not ``None``.  Subclasses should draw a small highlight (e.g. a dot,
        ring, or border) at the position corresponding to *action*.

        Parameters
        ----------
        action:
            The most recent ``MoveAction`` (guaranteed to have valid row/col
            unless it is a pass action, which is filtered out before calling).
        """

    def draw_grid_marker(
        self,
        action: MoveAction,
        grid_h: int,
        grid_w: int,
        first_color: Tuple[int, int, int],
        second_color: Tuple[int, int, int],
        stone_on_intersection: bool = False,
        padding: int = 20,
    ) -> None:
        """Draw a contrasting dot marker on the last-moved grid cell or intersection.

        A shared helper that eliminates duplicated marker-drawing code across
        grid-based games (Othello, Connect4, Gomoku, TicTacToe).

        Parameters
        ----------
        action:
            The move to mark (must have valid ``row`` and ``col``).
        grid_h, grid_w:
            Pixel dimensions of one grid cell.
        first_color:
            Marker colour drawn on PLAYER_FIRST's stones (should contrast).
        second_color:
            Marker colour drawn on PLAYER_SECOND's stones (should contrast).
        stone_on_intersection:
            ``True`` for Go/Gomoku-style boards; ``False`` for cell-based boards.
        padding:
            Board padding (only used when ``stone_on_intersection=True``).
        """
        row, col = action.row, action.col
        pos_x = self._gui_board_offset_x
        pos_y = self._gui_board_offset_y

        if stone_on_intersection:
            center_x = pos_x + padding + col * grid_w
            center_y = pos_y + padding + row * grid_h
        else:
            center_x = pos_x + col * grid_w + grid_w // 2
            center_y = pos_y + row * grid_h + grid_h // 2

        marker_radius = max(3, min(grid_h, grid_w) // 8)
        cell_value = int(self.state.board[row][col])
        # cell_value 1 = PLAYER_FIRST stone → use first_color marker
        marker_color = first_color if cell_value == 1 else second_color
        pygame.draw.circle(self._surface, marker_color, (center_x, center_y), marker_radius)

    def reset(self) -> None:
        """Reset the game to its initial state.

        Reinitialises ``self.state`` with a blank numpy board and sets the
        first player as the active player.  Subclasses that need extra
        initialisation should call ``super().reset()`` first.
        """
        self.last_action = None
        self.state = self.reset_state(self.state)

    @abstractmethod
    def reset_state(self, state: GameState) -> GameState:
        """Reset the given *state* to its initial configuration and return it.

        Unlike ``reset()`` which always operates on ``self.state``, this
        method accepts an arbitrary ``GameState`` instance, mutates it in
        place, and returns it.  This is useful for MCTS / AI search where
        multiple states need to be independently reset without touching the
        live game state.

        Parameters
        ----------
        state:
            The ``GameState`` instance to reset.

        Returns
        -------
        GameState
            The same *state* instance after being reset.
        """

    def pass_turn(self) -> None:
        """Called when the current player has no legal actions and must pass.

        This is distinct from ``switch_turn``: ``pass_turn`` represents a
        forced skip (e.g. no valid moves in Go / Othello), while
        ``switch_turn`` is the normal end-of-move handoff.
        """
        self.state.turn = (
            self.PLAYER_SECOND
            if self.state.turn == self.PLAYER_FIRST
            else self.PLAYER_FIRST
        )

    def switch_turn(self) -> None:
        """Advance to the next player and increment the turn counter.

        Typically called at the end of a successful ``move()``.
        """
        self.state.total_turns += 1
        self.pass_turn()

    def end_game(self, winner: Optional[int]) -> None:
        """Mark the game as finished with the given *winner*.

        Parameters
        ----------
        winner:
            ``PLAYER_FIRST``, ``PLAYER_SECOND``, or ``RESULT_TIE``.
        """
        self.state.winner = winner
        self.state.is_game_over = True


# ---------------------------------------------------------------------------
# Generic terminal demo runner
# ---------------------------------------------------------------------------

def _prompt_first_player(game: BoardGame) -> int:
    """Ask who goes first and return the chosen player constant."""
    print("\nWho goes first?  (1) Player 1  (2) Player 2")
    while True:
        choice = input("Enter 1 or 2: ").strip()
        if choice == "1":
            return game.PLAYER_FIRST
        if choice == "2":
            return game.PLAYER_SECOND
        print("Please enter 1 or 2.")


def _play_one_game(game: BoardGame) -> bool:
    """Run a single game session.

    Returns
    -------
    bool
        ``True`` if the game finished normally, ``False`` if a mid-game reset
        was requested.
    """
    game.reset()

    player_names = {game.PLAYER_FIRST: "Player 1", game.PLAYER_SECOND: "Player 2"}

    while not game.state.is_game_over:
        game.render()
        current_player = game.state.turn
        print(f"\n{player_names[current_player]}'s turn.  [r] Reset  [q] Quit")

        user_input = input(">>> ").strip()

        if user_input.lower() == "r":
            print("\n--- Resetting game ---")
            return False
        if user_input.lower() == "q":
            raise SystemExit

        action = game.parse_terminal_input(user_input)
        if action is None:
            continue

        game.move(current_player, action)

    game.render()
    if game.state.winner == game.RESULT_TIE:
        print("\nIt's a tie!")
    else:
        winner_name = "Player 1" if game.state.winner == game.PLAYER_FIRST else "Player 2"
        print(f"\n{winner_name} wins!")
    return True


def run_gui_demo(game: BoardGame, args: "GameArgs") -> None:
    """Generic pygame GUI game loop supporting three play modes.

    Play modes (``args.mode``)
    --------------------------
    - ``"hum"``    – human vs human; board clicks on every turn.
    - ``"ai"``     – AI vs AI; moves generated automatically every frame.
    - ``"hum-ai"`` – one AI built from ``args``; Reset button opens an overlay
                     so the human can pick **Human First / AI First** before
                     each game.  ``is_first_player_ai`` tracks whose turn it is
                     to act automatically.

    Layout
    ------
    - **Left**: board (drawn by ``game.draw_board()``).
    - **Right**: info block → Reset button → Pass button (conditional).

    Parameters
    ----------
    game:
        A fully initialised ``BoardGame`` instance with ``gui_mode=True``.
    args:
        ``GameArgs`` providing ``grid_h``, ``grid_w``, ``title``, ``mode``,
        and (for AI modes) model / simulation parameters.
    """
    if not GUI_MODE_AVAILABLE:
        raise RuntimeError(
            "pygame is not installed. Install it with `pip install pygame` to use GUI mode."
        )

    grid_h = args.grid_h
    grid_w = args.grid_w
    mode   = args.mode   # "hum" | "ai" | "hum-ai"

    # Build AI player(s) based on mode.
    #
    # "ai"    – two independent AI objects so each side has its own MCTS tree
    #           and can load a different checkpoint via model_path_first / model_path_second.
    # "hum-ai"– one AI object; is_first_player_ai decides which player-id it acts for.
    # "hum"   – no AI objects.
    ai_player_first  = None  # PLAYER_FIRST  controller (ai-vs-ai only)
    ai_player_second = None  # PLAYER_SECOND controller (ai-vs-ai only)
    ai_player        = None  # single AI used in hum-ai mode

    if mode == "ai":
        path_first  = args.model_path_first  or args.model_path
        path_second = args.model_path_second or args.model_path
        ai_player_first  = _build_ai(game, args, "AI-First",  model_path=path_first)
        ai_player_second = _build_ai(game, args, "AI-Second", model_path=path_second)
    elif mode == "hum-ai" and args.model_path:
        ai_player = _build_ai(game, args, "AI")

    # is_first_player_ai: True  → PLAYER_FIRST is AI, PLAYER_SECOND is human  (hum-ai only)
    #                     False → PLAYER_FIRST is human, PLAYER_SECOND is AI   (hum-ai only)
    is_first_player_ai: bool = args.ai_plays_first

    def _current_ai():
        """Return the AI object that should move this turn, or None for a human turn."""
        if mode == "ai":
            return ai_player_first if game.state.turn == game.PLAYER_FIRST else ai_player_second
        if mode == "hum-ai":
            if game.state.turn == game.PLAYER_FIRST:
                return ai_player if is_first_player_ai else None
            else:
                return ai_player if not is_first_player_ai else None
        return None  # "hum": always human

    def _is_ai_turn() -> bool:
        return _current_ai() is not None

    pygame.init()
    window_w, window_h = game.get_window_size(0, 0, grid_h, grid_w)
    screen = pygame.display.set_mode((window_w, window_h))
    pygame.display.set_caption(f"AlphaLyn {args.title}")
    clock = pygame.time.Clock()

    def _reset_all_ai() -> None:
        """Reset every AI object that was built for this session."""
        for ai_obj in (ai_player_first, ai_player_second, ai_player):
            if ai_obj is not None:
                ai_obj.reset()

    game.init_gui(screen)
    _reset_all_ai()
    game.reset()

    bp = game.GUI_BOARD_PADDING
    panel_margin = 16
    board_w, board_h = game.get_board_size(0, 0, grid_h, grid_w)
    btn_w = game.GUI_BUTTON_WIDTH
    btn_h = game.GUI_BUTTON_HEIGHT

    board_offset_x = bp
    board_offset_y = bp
    game._gui_board_w        = board_w
    game._gui_board_offset_x = board_offset_x
    game._gui_board_offset_y = board_offset_y

    panel_x   = board_offset_x + board_w + panel_margin
    item_gap  = panel_margin * 2
    reset_top = panel_margin + game.GUI_INFO_BAR_HEIGHT + item_gap
    pass_top  = reset_top + btn_h + item_gap

    reset_button_rect = pygame.Rect(panel_x, reset_top, btn_w, btn_h)
    pass_button_rect  = pygame.Rect(panel_x, pass_top,  btn_w, btn_h)

    # ---- hum-vs-ai reset overlay ----
    # Only shown in "hum-ai" mode when Reset is clicked.
    overlay_panel_w = btn_w * 2 + panel_margin * 3
    overlay_panel_h = btn_h * 3 + panel_margin * 2
    overlay_x = (window_w - overlay_panel_w) // 2
    overlay_y = (window_h - overlay_panel_h) // 2
    overlay_title_rect  = pygame.Rect(overlay_x, overlay_y, overlay_panel_w, btn_h)
    overlay_human_rect  = pygame.Rect(overlay_x, overlay_y + btn_h + panel_margin, btn_w, btn_h)
    overlay_ai_rect     = pygame.Rect(overlay_x + btn_w + panel_margin, overlay_y + btn_h + panel_margin, btn_w, btn_h)
    overlay_cancel_rect = pygame.Rect(overlay_x, overlay_y + (btn_h + panel_margin) * 2, overlay_panel_w, btn_h)

    def _render_frame() -> None:
        """Draw board + right panel + optional hum-ai overlay."""
        screen.fill(game.GUI_BG_COLOR)
        game.draw_board()
        if game.last_action is not None and game.last_action.row >= 0:
            game.draw_last_move_marker(game.last_action)
        game.draw_info()
        game._draw_button("Reset", reset_button_rect)

        # Show Pass only on a human turn.
        if (game.should_show_pass_button()
                and not game.state.is_game_over
                and not _is_ai_turn()):
            game._draw_button("Pass", pass_button_rect, bg_color=(80, 120, 80))

        # hum-ai overlay (drawn on top of everything when active).
        if game._show_reset_panel and mode == "hum-ai":
            overlay_surf = pygame.Surface((window_w, window_h), pygame.SRCALPHA)
            overlay_surf.fill(game.GUI_OVERLAY_COLOR)
            screen.blit(overlay_surf, (0, 0))
            panel_bg = pygame.Rect(
                overlay_x - panel_margin, overlay_y - panel_margin,
                overlay_panel_w + panel_margin * 2, overlay_panel_h + panel_margin * 2,
            )
            pygame.draw.rect(screen, game.GUI_BG_COLOR, panel_bg, border_radius=10)
            game._draw_text_centered("Who goes first?", overlay_title_rect, game.GUI_TEXT_COLOR)
            game._draw_button("Human First", overlay_human_rect)
            game._draw_button("AI First", overlay_ai_rect)
            game._draw_button("Cancel", overlay_cancel_rect, bg_color=(120, 60, 60))

        pygame.display.flip()

    running = True
    while running:
        # ---- AI auto-move ----
        if not game.state.is_game_over and not game._show_reset_panel:
            current_ai = _current_ai()
            if current_ai is not None:
                action = current_ai.next_move()
                game.move(game.state.turn, action)
                # Notify the acting AI, then also notify the opponent AI (ai-vs-ai)
                # so both trees stay in sync with the actual game history.
                current_ai.observe_action(action)
                opponent_ai = ai_player_second if current_ai is ai_player_first else ai_player_first
                if opponent_ai is not None:
                    opponent_ai.observe_action(action)
                _render_frame()

        # ---- Event handling ----
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
                break

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                click_x, click_y = event.pos

                # hum-ai overlay has priority over all other clicks.
                if game._show_reset_panel and mode == "hum-ai":
                    if overlay_human_rect.collidepoint(click_x, click_y):
                        is_first_player_ai = False  # human is PLAYER_FIRST
                        game._is_first_player_ai = is_first_player_ai
                        game.toggle_reset_panel()
                        _reset_all_ai()
                        game.reset()
                        game.state.turn = game.PLAYER_FIRST  # human goes first
                    elif overlay_ai_rect.collidepoint(click_x, click_y):
                        is_first_player_ai = True   # AI is PLAYER_FIRST
                        game._is_first_player_ai = is_first_player_ai
                        game.toggle_reset_panel()
                        _reset_all_ai()
                        game.reset()
                        game.state.turn = game.PLAYER_FIRST  # AI goes first
                    elif overlay_cancel_rect.collidepoint(click_x, click_y):
                        game.toggle_reset_panel()
                    continue

                # Reset button.
                if reset_button_rect.collidepoint(click_x, click_y):
                    if mode == "hum-ai":
                        game.toggle_reset_panel()  # open overlay to choose who goes first
                    else:
                        _reset_all_ai()
                        game.reset()
                    continue

                # Pass button (human turns only).
                if (not _is_ai_turn()
                        and game.should_show_pass_button()
                        and not game.state.is_game_over
                        and pass_button_rect.collidepoint(click_x, click_y)):
                    pass_action = MoveAction(row=-1, col=-1, player=game.state.turn, extra="pass")
                    game.pass_turn()
                    # Notify the AI that the human passed (so it can update its MCTS tree).
                    current_ai = _current_ai()
                    if current_ai is not None:
                        current_ai.observe_action(pass_action)
                    continue

                # Board click – human turns only, inside board area.
                board_right  = board_offset_x + board_w
                board_bottom = board_offset_y + board_h
                if (not _is_ai_turn()
                        and not game.state.is_game_over
                        and board_offset_x <= click_x < board_right
                        and board_offset_y <= click_y < board_bottom):
                    action = game.handle_board_click(click_x, click_y, grid_h, grid_w)
                    if action is not None:
                        game.move(game.state.turn, action)
                        # After human moves, notify the opponent AI (now its turn).
                        next_ai = _current_ai()
                        if next_ai is not None:
                            next_ai.observe_action(action)
                        _render_frame()
                        continue

        # ---- Rendering ----
        _render_frame()
        clock.tick(60)

    pygame.quit()


def run_demo(args: GameArgs, game_cls: Any) -> None:
    """Unified entry point: constructs the game and dispatches to the right runner.

    ``args.mode`` selects the play mode for both GUI and terminal:

    - ``"hum"``    – human vs human (default).
    - ``"ai"``     – AI vs AI (requires ``args.model_path``).
    - ``"hum-ai"`` – human vs AI (requires ``args.model_path``).
    """
    game = game_cls(args)
    if args.gui_mode:
        run_gui_demo(game, args)
    else:
        run_terminal_demo(game, args)


def run_terminal_demo(game: BoardGame, args: Optional[GameArgs] = None) -> None:
    """Generic terminal game loop supporting three play modes.

    Modes (controlled by ``args.mode``):

    - ``"hum"``    – two humans take turns at the keyboard *(default)*.
    - ``"ai"``     – two AI instances play autonomously; a single game is run.
    - ``"hum-ai"`` – one human vs one AI; ``args.ai_plays_first`` picks sides.

    For AI modes a single game is played then the function returns.
    For ``"hum"`` mode the classic reset/replay loop is used.

    Parameters
    ----------
    game:
        A fully initialised ``BoardGame`` instance.
    args:
        Optional ``GameArgs``; if ``None`` or ``args.mode == "hum"``, falls
        back to the classic human-vs-human loop.
    """
    mode = args.mode if args is not None else "hum"

    # ---- AI vs AI ----
    if mode == "ai":
        has_any_path = (args and (args.model_path or args.model_path_first or args.model_path_second))
        if not has_any_path:
            print("Error: --model-path-first / --model-path-second (or --model-path as fallback) "
                  "is required for ai mode.")
            return
        game.reset()
        # Per-side paths fall back to the shared model_path when not specified.
        path_first  = args.model_path_first  or args.model_path
        path_second = args.model_path_second or args.model_path
        ai1 = _build_ai(game, args, "AI-First",  model_path=path_first)
        ai2 = _build_ai(game, args, "AI-Second", model_path=path_second)
        play_ai_vs_ai(game, ai1, ai2, verbose=True)
        return

    # ---- Human vs AI ----
    if mode == "hum-ai":
        if not args or not args.model_path:
            print("Error: --model_path is required for hum-ai mode.")
            return
        ai = _build_ai(game, args, "AI")
        ai_plays_first = args.ai_plays_first
        while True:
            print("\nWho goes first?  (h) Human  (a) AI")
            choice = input(">>> ").strip().lower()
            if choice == "h":
                ai_plays_first = False
            elif choice == "a":
                ai_plays_first = True
            else:
                print(f"Keeping previous choice: {'AI' if ai_plays_first else 'Human'} goes first.")
            ai.reset()
            game.reset()
            print(f"You are playing against AI ({'AI' if ai_plays_first else 'Human'} goes first).")
            play_human_vs_ai(game, ai, ai_plays_first=ai_plays_first, verbose=True)
            print("\nPlay again?  (y) Yes  (any) Quit")
            if input(">>> ").strip().lower() != "y":
                print("Thanks for playing!")
                break
        return

    # ---- Human vs Human (default) ----
    while True:
        game_finished = _play_one_game(game)

        if not game_finished:
            continue  # mid-game reset

        print("\nPlay again?  (y) Yes  (any) Quit")
        if input(">>> ").strip().lower() != "y":
            print("Thanks for playing!")
            break


def play_ai_vs_ai(
    game: BoardGame,
    first_player_ai: Any,
    second_player_ai: Any,
    verbose: bool = True,
) -> Optional[int]:
    """Run a complete game between two AI players and return the winner.

    Both AIs observe every action so their internal MCTS trees stay in sync.
    """
    first_player_ai.reset()
    second_player_ai.reset()

    while not game.state.is_game_over:
        current_player = game.state.turn
        ai = first_player_ai if current_player == game.PLAYER_FIRST else second_player_ai

        action = ai.next_move()
        game.move(current_player, action)

        first_player_ai.observe_action(action)
        second_player_ai.observe_action(action)

        if verbose:
            player_label = "First" if current_player == game.PLAYER_FIRST else "Second"
            print(f"Player {player_label} → row={action.row}, col={action.col}, extra={action.extra}")
            print(game.state.board)
            print()

    winner = game.state.winner
    if verbose:
        if winner == game.RESULT_TIE:
            print("Result: Tie")
        elif winner == game.PLAYER_FIRST:
            print("Result: First player wins")
        else:
            print("Result: Second player wins")
    return winner


def play_human_vs_ai(
    game: BoardGame,
    ai: Any,
    ai_plays_first: bool = False,
    verbose: bool = True,
) -> Optional[int]:
    """Run a complete game between a human (terminal input) and an AI."""
    ai.reset()

    ai_player_id    = game.PLAYER_FIRST  if ai_plays_first else game.PLAYER_SECOND
    human_player_id = game.PLAYER_SECOND if ai_plays_first else game.PLAYER_FIRST

    while not game.state.is_game_over:
        current_player = game.state.turn
        print(game.state.board)

        if current_player == ai_player_id:
            print("AI is thinking…")
            action = ai.next_move()
            print(f"AI plays → row={action.row}, col={action.col}, extra={action.extra}")
        else:
            legal_actions = game.get_legal_actions(human_player_id)
            if not legal_actions:
                print("No legal moves – you must pass.")
                action = MoveAction(row=-1, col=-1, player=human_player_id, extra="pass")
            else:
                if verbose:
                    print(f"Legal moves: {[(a.row, a.col) for a in legal_actions]}")
                while True:
                    raw = input("Your move (row,col): ").strip()
                    parsed_action = game.parse_terminal_input(raw)
                    if parsed_action is not None:
                        action = parsed_action
                        break

        game.move(current_player, action)
        ai.observe_action(action)
        print()

    print(game.state.board)
    winner = game.state.winner
    if winner == game.RESULT_TIE:
        print("Result: Tie")
    elif winner == ai_player_id:
        print("Result: AI wins")
    else:
        print("Result: You win!")
    return winner


def _build_ai(
    game: BoardGame,
    args: Any,
    player_label: str = "AI",
    model_path: str = "",
) -> Any:
    """Construct an ``AIPlayer`` from a checkpoint and the config in *args*.

    Uses lazy imports to avoid circular dependency (ai_player → board_game).

    Parameters
    ----------
    game:
        The live ``BoardGame`` instance the AI will play on.
    args:
        ``GameArgs`` providing model architecture params and ``num_simulations``.
    player_label:
        Human-readable name printed while loading.
    model_path:
        Path to the checkpoint file.  Overrides ``args.model_path`` when set,
        so ai-vs-ai mode can supply per-side paths while ``args.model_path``
        serves as the shared fallback.
    """
    from ai_player import AIPlayer, AIPlayerConfig  # lazy – avoids circular dep
    from model import PolicyValueNet, ResNetConfig
    from mcts import MCTSConfig
    import torch

    resolved_path = model_path or args.model_path
    print(f"Loading {player_label} from '{resolved_path}' ({args.sims} sims)…")

    net = PolicyValueNet(ResNetConfig(
        board_height=game.height,
        board_width=game.width,
        num_actions=game.num_actions,
        num_filters=args.num_filters,
        num_residual_blocks=args.num_residual_blocks,
        value_head_hidden_size=args.value_head_hidden_size,
    ))
    checkpoint = torch.load(resolved_path, map_location=args.device)
    net.load_state_dict(checkpoint)
    net.eval()

    config = AIPlayerConfig(
        mcts=MCTSConfig(
            num_simulations=args.sims,
            device=args.device,
            # Dirichlet noise is for training-time exploration only;
            # disable it entirely during play so the AI is deterministic.
            dirichlet_epsilon=0.0,
        ),
        greedy=True,
        batch_size=args.mcts_batch,
    )
    return AIPlayer(game, net, config)