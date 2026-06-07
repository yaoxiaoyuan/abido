"""AI player that uses MCTS + PolicyValueNet to generate moves.

This module provides ``AIPlayer``, a self-contained wrapper around ``MCTS``
that can be dropped into any ``BoardGame`` game loop.

Typical usage
-------------
>>> from game_othello import OthelloGame, OthelloArgs
>>> from model import PolicyValueNet, ResNetConfig
>>> from mcts import MCTSConfig
>>> from ai_player import AIPlayer, AIPlayerConfig
>>>
>>> game = OthelloGame(OthelloArgs())
>>> net = PolicyValueNet(ResNetConfig())
>>> config = AIPlayerConfig(mcts=MCTSConfig(num_simulations=400), greedy=False)
>>> ai = AIPlayer(game, net, config)
>>>
>>> # In your game loop:
>>> action = ai.next_move()          # let MCTS pick the best move
>>> game.move(game.state.turn, action)
>>> ai.observe_action(action)        # advance MCTS tree root
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from board_game import BoardGame, MoveAction
from model import PolicyValueNet, ResNetConfig
from mcts import MCTS, MCTSConfig


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class AIPlayerConfig:
    """Configuration for ``AIPlayer``.

    Attributes
    ----------
    mcts:
        ``MCTSConfig`` forwarded to the internal ``MCTS`` instance.
    temperature:
        Sampling temperature passed to ``MCTS.get_action_probs``.
        - ``1.0`` : sample proportionally to visit counts (exploratory)
        - ``0.0`` : always pick the most-visited action (greedy)
    greedy:
        If ``True``, ``select_action`` returns the highest-probability action
        deterministically (equivalent to ``temperature → 0``).
        If ``False``, sample according to the probability distribution.
    batch_size:
        Number of MCTS simulations to batch into a single network forward
        pass.  ``1`` disables batching (sequential mode).  Larger values
        (e.g. 8–32) improve GPU utilisation.
    """

    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    temperature: float = 1.0
    greedy: bool = False
    batch_size: int = 64


# ---------------------------------------------------------------------------
# AI Player
# ---------------------------------------------------------------------------

class AIPlayer:
    """Wraps ``MCTS`` to provide a simple ``next_move()`` interface.

    The internal search tree is preserved between moves so the AI re-uses
    work already done in previous turns.  Call ``observe_action`` after every
    move (whether made by the AI or by a human opponent) to advance the tree
    root to the correct node.

    Parameters
    ----------
    game:
        The shared ``BoardGame`` instance.  ``AIPlayer`` reads ``game.state``
        but never mutates it directly — all mutations happen inside MCTS on
        cloned states.
    net:
        A ``PolicyValueNet`` (trained or randomly initialised).
    config:
        ``AIPlayerConfig`` controlling search and sampling behaviour.
    """

    def __init__(
        self,
        game: BoardGame,
        net: PolicyValueNet,
        config: Optional[AIPlayerConfig] = None,
    ) -> None:
        self.game = game
        self.net = net
        self.config = config or AIPlayerConfig()
        self._mcts = MCTS(game, net, self.config.mcts)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def next_move(self) -> MoveAction:
        """Run MCTS from the current game state and return the chosen action.

        The tree root is **not** advanced here; call ``observe_action`` with
        the returned action afterwards to keep the tree in sync.

        Returns
        -------
        MoveAction
            The action selected by the AI.  This is always a legal move.
        """
        action_probs = self._mcts.get_action_probs(
            temperature=self.config.temperature,
            batch_size=self.config.batch_size,
        )
        return self._mcts.select_action(action_probs, greedy=self.config.greedy)

    def observe_action(self, action: MoveAction) -> None:
        """Advance the internal MCTS tree root to the child for *action*.

        Must be called after **every** move in the game (AI or human) so that
        the tree stays aligned with the real game state.  Re-uses the subtree
        built during ``next_move`` where possible; falls back to a fresh root
        if the child was never visited.

        Parameters
        ----------
        action:
            The action that was just applied to the game.
        """
        self._mcts.update_root_with_action(action)

    def reset(self) -> None:
        """Discard the search tree and start fresh.

        Call this whenever the game is reset so stale tree nodes are not
        carried over into the next game.
        """
        self._mcts.reset()

    def get_action_probs(self) -> List[Tuple[MoveAction, float]]:
        """Return the full action-probability distribution from MCTS.

        Useful for logging, analysis, or training data collection.
        The distribution reflects the visit counts from the most recent
        ``next_move`` call.

        Returns
        -------
        List of ``(MoveAction, probability)`` pairs summing to 1.
        """
        return self._mcts.get_action_probs(
            temperature=self.config.temperature,
            batch_size=self.config.batch_size,
        )

    # ------------------------------------------------------------------
    # Convenience factory
    # ------------------------------------------------------------------

    @classmethod
    def with_random_net(
        cls,
        game: BoardGame,
        config: Optional[AIPlayerConfig] = None,
        num_input_planes: int = 3,
    ) -> "AIPlayer":
        """Create an ``AIPlayer`` with a randomly initialised network.

        Useful for smoke-testing the game loop without a trained checkpoint.

        Parameters
        ----------
        game:
            The ``BoardGame`` instance the AI will play.
        config:
            Optional ``AIPlayerConfig``.  Defaults are used if omitted.
        num_input_planes:
            Number of input feature planes for the network (default 3).
        """
        net_config = ResNetConfig(
            board_height=game.height,
            board_width=game.width,
            num_input_planes=num_input_planes,
            num_actions=game.num_actions,
        )
        net = PolicyValueNet(net_config)
        return cls(game, net, config)
