"""Monte Carlo Tree Search (MCTS) with a neural network prior.

This module implements the AlphaZero-style MCTS that uses a ``PolicyValueNet``
to guide tree expansion and backup.  It is designed to work with any
``BoardGame`` subclass defined in ``board_game.py``.

Typical usage
-------------
>>> from game_othello import OthelloGame, OthelloArgs
>>> from model import PolicyValueNet, ResNetConfig, encode_board
>>> from mcts import MCTS, MCTSConfig
>>>
>>> game = OthelloGame(OthelloArgs())
>>> net = PolicyValueNet(ResNetConfig())
>>> config = MCTSConfig(num_simulations=400, c_puct=1.5)
>>> mcts = MCTS(game, net, config)
>>>
>>> action_probs = mcts.get_action_probs(temperature=1.0)
>>> action = mcts.select_action(action_probs)
>>> game.move(game.state.turn, action)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from board_game import BoardGame, MoveAction, GameState
from model import PolicyValueNet, encode_board


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MCTSConfig:
    """Hyper-parameters controlling MCTS behaviour.

    Attributes
    ----------
    num_simulations:
        Number of MCTS simulations (tree traversals) per ``get_action_probs``
        call.  Higher values give stronger play at the cost of more compute.
    c_puct:
        Exploration constant in the PUCT formula.  Larger values encourage
        more exploration of less-visited nodes.
    dirichlet_alpha:
        Alpha parameter for Dirichlet noise added to the root prior.  Set to
        ``0.0`` to disable noise (useful during evaluation / self-play testing).
    dirichlet_epsilon:
        Weight of Dirichlet noise mixed into the root prior.
    board_player_first_value:
        Board cell value representing PLAYER_FIRST's stone (default 1).
    board_player_second_value:
        Board cell value representing PLAYER_SECOND's stone (default 2).
    device:
        Torch device string (``"cpu"``, ``"cuda"``, ``"mps"``).
    """

    num_simulations: int = 400
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    board_player_first_value: int = 1
    board_player_second_value: int = 2
    device: str = "cpu"




# ---------------------------------------------------------------------------
# MCTS node
# ---------------------------------------------------------------------------

class MCTSNode:
    """A single node in the MCTS search tree.

    Each node corresponds to a unique game state reached via a particular
    action from its parent.

    Attributes
    ----------
    parent:
        Parent node, or ``None`` for the root.
    action_taken:
        The ``MoveAction`` that led from the parent to this node.
    prior_probability:
        Prior probability P(s, a) from the neural network for the action that
        created this node.
    visit_count:
        Number of times this node has been visited (N in AlphaZero notation).
    total_value:
        Accumulated value from all backpropagated simulations (W).
    children:
        Mapping from flat action index to child ``MCTSNode``.
    is_expanded:
        Whether ``expand()`` has been called on this node.
    game_state:
        The ``GameState`` snapshot at this node.  Stored so that simulations
        can read the board directly from the node instead of replaying moves
        from the root on a cloned game object.  This trades a small amount of
        memory (one numpy array per node) for a large reduction in CPU time
        (no ``copy.deepcopy`` per simulation).
    """

    __slots__ = (
        "parent",
        "action_taken",
        "prior_probability",
        "visit_count",
        "total_value",
        "children",
        "is_expanded",
        "game_state",
    )

    def __init__(
        self,
        parent: Optional[MCTSNode],
        action_taken: Optional[MoveAction],
        prior_probability: float,
        game_state: Optional["GameState"] = None,  # type: ignore[name-defined]
    ) -> None:
        self.parent = parent
        self.action_taken = action_taken
        self.prior_probability = prior_probability
        self.visit_count: int = 0
        self.total_value: float = 0.0
        self.children: Dict[int, MCTSNode] = {}
        self.is_expanded: bool = False
        self.game_state = game_state  # GameState | None

    @property
    def mean_value(self) -> float:
        """Q(s,a) = W / N, computed on the fly from total_value and visit_count."""
        return self.total_value / self.visit_count if self.visit_count > 0 else 0.0

    # ------------------------------------------------------------------
    # PUCT selection
    # ------------------------------------------------------------------

    def puct_score(self, c_puct: float) -> float:
        """Compute the PUCT score used to select which child to visit.

        PUCT(s, a) = -Q(s, a) + c_puct * P(s, a) * sqrt(N(s)) / (1 + N(s, a))

        ``mean_value`` is stored from the perspective of the player whose turn
        it is **at this child node** (i.e. the opponent of the parent node's
        player).  The parent selects the child that is best for itself, which
        means worst for the opponent — hence we negate ``mean_value`` before
        adding the exploration term.

        where N(s) is the parent's visit count and N(s, a) is this node's.
        """
        parent_visits = self.parent.visit_count if self.parent is not None else 1
        exploration = c_puct * self.prior_probability * math.sqrt(parent_visits) / (1 + self.visit_count)
        return -self.mean_value + exploration

    def best_child(self, c_puct: float) -> Tuple[int, "MCTSNode"]:
        """Return (action_index, child_node) with the highest PUCT score."""
        return max(self.children.items(), key=lambda item: item[1].puct_score(c_puct))

    # ------------------------------------------------------------------
    # Expansion & backup
    # ------------------------------------------------------------------

    def expand(
        self,
        action_priors: List[Tuple[int, float]],
        child_states: Dict[int, "GameState"],  # type: ignore[name-defined]
    ) -> None:
        """Create child nodes for all legal actions.

        Parameters
        ----------
        action_priors:
            List of ``(action_index, prior_probability)`` pairs from the
            neural network policy head.
        child_states:
            Mapping from action_index to the ``GameState`` that results from
            taking that action.  Each child node stores its own state so that
            simulations can read the board directly without replaying moves.
        """
        self.is_expanded = True
        for action_index, prior in action_priors:
            if action_index not in self.children:
                self.children[action_index] = MCTSNode(
                    parent=self,
                    action_taken=None,
                    prior_probability=prior,
                    game_state=child_states.get(action_index),
                )

    def backup(self, value: float) -> None:
        """Propagate *value* up to the root, flipping sign at each level.

        The value is from the perspective of the player who just moved into
        this node.  Each ancestor alternates perspective, so we negate at
        every step.
        """
        self.visit_count += 1
        self.total_value += value
        if self.parent is not None:
            self.parent.backup(-value)

    def is_leaf(self) -> bool:
        """Return True if this node has not been expanded yet."""
        return not self.is_expanded


# ---------------------------------------------------------------------------
# MCTS
# ---------------------------------------------------------------------------

class MCTS:
    """AlphaZero-style Monte Carlo Tree Search.

    The tree is re-used across calls to ``get_action_probs`` when
    ``reuse_tree=True`` (default), which amortises the cost of early
    simulations across moves.

    Parameters
    ----------
    game:
        A ``BoardGame`` instance whose state will be searched.  The MCTS
        operates on *clones* of the game state so the original is never
        mutated during search.
    net:
        Trained (or randomly initialised) ``PolicyValueNet``.
    config:
        ``MCTSConfig`` controlling search hyper-parameters.
    """

    def __init__(self, game: BoardGame, net: PolicyValueNet, config: MCTSConfig) -> None:
        self.game = game
        self.net = net
        self.config = config
        self._device = torch.device(config.device)
        self.net.to(self._device)
        self._root: Optional[MCTSNode] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Discard the current search tree (call after each game reset)."""
        self._root = None

    def get_action_probs(
        self, temperature: float = 1.0, batch_size: int = 1
    ) -> List[Tuple[MoveAction, float]]:
        """Run MCTS simulations and return a policy over legal actions.

        Parameters
        ----------
        temperature:
            Controls how peaked the returned distribution is.
            - ``temperature = 1.0`` : proportional to visit counts (exploration)
            - ``temperature → 0``   : argmax (exploitation)
            - ``temperature = 0``   : pure greedy (select most-visited child)
        batch_size:
            Number of simulations to group into a single batched network call.
            ``batch_size = 1`` falls back to the original sequential
            ``_simulate`` path.  Larger values (e.g. 8–32) improve GPU
            utilisation by amortising inference overhead across multiple
            leaf evaluations per forward pass.

        Returns
        -------
        List of ``(MoveAction, probability)`` pairs, summing to 1.
        """
        if self._root is None:
            self._root = MCTSNode(
                parent=None,
                action_taken=None,
                prior_probability=1.0,
                game_state=self.game.state.clone(),
            )

        # Add Dirichlet noise to the root prior for exploration during training
        if not self._root.is_expanded:
            self._expand_and_evaluate(self._root)
            #visualize_tree(self, "mcts_tree.svg", 2, 7)
            #input("1")
        if self.config.dirichlet_alpha > 0.0 and self._root.children:
            self._add_dirichlet_noise(self._root)

        if batch_size <= 1:
            for _ in range(self.config.num_simulations):
                self._simulate()
        else:
            completed = 0
            while completed < self.config.num_simulations:
                remaining = self.config.num_simulations - completed
                completed += self._simulate_batch(min(batch_size, remaining))

        return self._build_action_probs(temperature)

    def select_action(
        self, action_probs: List[Tuple[MoveAction, float]], greedy: bool = False
    ) -> MoveAction:
        """Sample or argmax an action from the distribution returned by ``get_action_probs``.

        Parameters
        ----------
        action_probs:
            Output of ``get_action_probs``.
        greedy:
            If ``True``, return the action with the highest probability.
            If ``False``, sample according to the distribution.
        """
        actions, probs = zip(*action_probs)
        if greedy:
            return actions[int(np.argmax(probs))]
        prob_array = np.array(probs, dtype=np.float64)
        prob_array /= prob_array.sum()  # re-normalise to avoid floating-point drift
        chosen_index = np.random.choice(len(actions), p=prob_array)
        return actions[chosen_index]

    def update_root_with_action(self, action: MoveAction) -> None:
        """Advance the root to the child corresponding to *action*.

        Call this after each move to reuse the subtree built so far.  If the
        child does not exist (e.g. the tree was not searched deeply enough),
        the root is reset to a fresh node.
        """
        if self._root is None:
            return

        action_index = self.game.action_to_index(action)
        child = self._root.children.get(action_index)
        if child is not None:
            child.parent = None  # detach from old tree to free memory
            self._root = child
        else:
            self._root = None  # subtree not found; start fresh next call

    # ------------------------------------------------------------------
    # Core simulation loop
    # ------------------------------------------------------------------

    def _simulate(self) -> None:
        """Run one MCTS simulation: select → expand → evaluate → backup.

        Each node now carries its own ``game_state`` snapshot, so Selection
        simply follows child pointers and reads ``node.game_state`` at the
        leaf.  No game object is cloned during the simulation.
        """
        node = self._root

        # --- Selection: traverse to a leaf using PUCT ---
        while not node.is_leaf() and not node.game_state.is_game_over:
            _action_index, node = node.best_child(self.config.c_puct)

        # --- Expansion & evaluation ---
        if node.game_state.is_game_over:
            value = self._terminal_value(node.game_state)
        else:
            value = self._expand_and_evaluate(node)

        # --- Backup ---
        node.backup(value)

    def _simulate_batch(self, batch_size: int) -> int:
        """Run up to *batch_size* simulations with a single batched network forward pass.

        This is a pseudo-parallel approach: no real threads are used.  Instead
        we perform the Selection phase for all simulations first, collecting
        every leaf that needs network evaluation.  We then stack their board
        tensors into one batch and call ``net.predict`` once, which lets the
        GPU process all leaves in parallel.  Finally we distribute the results
        back and run Expansion + Backup for each leaf.

        Virtual loss
        ------------
        To prevent all paths from collapsing to the same leaf during
        Selection, we apply a *virtual loss* of ``-1`` to every node along
        each selected path before the network call.  After Backup the real
        value replaces the virtual loss automatically (since backup adds to
        ``total_value`` and the virtual loss is subtracted first).

        Parameters
        ----------
        batch_size:
            Maximum number of leaves to collect before the batched inference
            call.  The actual number may be smaller when the tree has fewer
            distinct reachable leaves than *batch_size* (e.g. at game start).

        Returns
        -------
        int
            The number of simulations actually completed (i.e. the number of
            unique leaves evaluated and backed up).  This may be less than
            *batch_size* when duplicate paths are discarded.
        """
        VIRTUAL_LOSS = 1.0

        # ----------------------------------------------------------------
        # Phase 1 – Selection: walk each simulation to a leaf, apply
        #           virtual loss along the way.
        # ----------------------------------------------------------------
        # leaf_entries: unique (leaf_node, path) pairs — one per distinct leaf.
        # Duplicate paths (multiple simulations landing on the same unclaimed
        # leaf) have their virtual loss undone immediately and are discarded.
        leaf_entries: List[Tuple[MCTSNode, List[MCTSNode]]] = []
        claimed_leaf_ids: set = set()

        for _ in range(batch_size):
            node = self._root
            path: List[MCTSNode] = [node]

            node.visit_count += 1
            node.total_value += VIRTUAL_LOSS

            while not node.is_leaf() and not node.game_state.is_game_over:
                _action_index, node = node.best_child(self.config.c_puct)
                node.visit_count += 1
                node.total_value += VIRTUAL_LOSS
                path.append(node)

            if id(node) in claimed_leaf_ids:
                # Undo virtual loss immediately — this path is done.
                for visited_node in path:
                    visited_node.visit_count -= 1
                    visited_node.total_value -= VIRTUAL_LOSS
            else:
                claimed_leaf_ids.add(id(node))
                leaf_entries.append((node, path))

        # ----------------------------------------------------------------
        # Phase 2 – Batch inference: encode all non-terminal leaves and
        #           call the network once.
        # ----------------------------------------------------------------
        terminal_indices: List[int] = []
        live_indices: List[int] = []
        board_tensors: List[torch.Tensor] = []

        for entry_index, (leaf_node, _path) in enumerate(leaf_entries):
            leaf_state = leaf_node.game_state
            if leaf_state.is_game_over:
                terminal_indices.append(entry_index)
            else:
                live_indices.append(entry_index)
                board_tensor = encode_board(
                    leaf_state.board,
                    leaf_state.turn,
                    self.config.board_player_first_value,
                    self.config.board_player_second_value,
                )
                board_tensors.append(board_tensor)

        # One batched forward pass for all live leaves
        net_values: List[float] = []
        net_policies: List[np.ndarray] = []
        if board_tensors:
            batch_tensor = torch.cat(board_tensors, dim=0).to(self._device)  # (N, C, H, W)
            policy_probs_batch, value_batch = self.net.predict(batch_tensor)
            policy_probs_np = policy_probs_batch.cpu().numpy()  # (N, num_actions)
            value_np = value_batch.cpu().numpy()                 # (N,)
            for batch_idx in range(len(live_indices)):
                net_values.append(float(value_np[batch_idx]))
                net_policies.append(policy_probs_np[batch_idx])

        # ----------------------------------------------------------------
        # Phase 3 – Undo virtual loss, then Expand + Backup each leaf.
        # ----------------------------------------------------------------
        live_cursor = 0

        for entry_index, (leaf_node, path) in enumerate(leaf_entries):
            for visited_node in path:
                visited_node.visit_count -= 1
                visited_node.total_value -= VIRTUAL_LOSS

            if entry_index in terminal_indices:
                value = self._terminal_value(leaf_node.game_state)
            else:
                value = net_values[live_cursor]
                policy_np = net_policies[live_cursor]
                live_cursor += 1
                if not leaf_node.is_expanded:
                    self._expand_node_from_policy(leaf_node, policy_np)

            leaf_node.backup(value)

        return len(leaf_entries)

    def _expand_and_evaluate(self, node: MCTSNode) -> float:
        """Query the network, expand *node*, and return the value estimate.

        Reads the board directly from ``node.game_state`` — no game object
        clone is required.

        Returns
        -------
        float
            Value in ``[-1, 1]`` from the perspective of the player whose turn
            it is at this node.
        """
        node_state = node.game_state
        board_tensor = encode_board(
            node_state.board,
            node_state.turn,
            self.config.board_player_first_value,
            self.config.board_player_second_value,
        ).to(self._device)

        policy_probs, value_tensor = self.net.predict(board_tensor)
        value = float(value_tensor.item())

        policy_np = policy_probs.squeeze(0).cpu().numpy()  # shape (num_actions,)
        self._expand_node_from_policy(node, policy_np)
        return value

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _expand_node_from_policy(self, node: MCTSNode, policy_np: np.ndarray) -> None:
        """Expand *node* using a pre-computed policy array from the network.

        Computes the ``GameState`` for every legal child by applying each
        action to a temporary game shell (one ``state.clone()`` per child),
        then stores the resulting state on the child node.  This is the only
        place where ``GameState.clone()`` is called — once per child during
        expansion, not once per simulation during selection.

        Parameters
        ----------
        node:
            The leaf node to expand.  Must have a valid ``game_state``.
        policy_np:
            1-D numpy array of shape ``(num_actions,)`` with raw network
            probabilities (before legal-action masking).
        """
        node_state = node.game_state
        current_player = node_state.turn

        # Temporarily install the node's state into the game shell so we can
        # call get_legal_actions and move without cloning the whole game.
        original_state = self.game.state
        self.game.state = node_state

        legal_actions = self.game.get_legal_actions(current_player)

        action_priors: List[Tuple[int, float]] = []
        child_states: Dict[int, object] = {}
        prior_sum = 0.0

        for legal_action in legal_actions:
            idx = self.game.action_to_index(legal_action)
            prior = float(policy_np[idx])
            action_priors.append((idx, prior))
            prior_sum += prior

            # Compute the child state by applying the action to a clone.
            child_state = node_state.clone()
            self.game.state = child_state
            self.game.move(current_player, legal_action)
            #print(node_state, legal_action, self.game.state)
            #input("0")
            child_states[idx] = self.game.state  # may differ from child_state if move reassigns

        # Restore the original game state so the game object is left unchanged.
        self.game.state = original_state

        # Re-normalise over legal actions (network may assign mass to illegal ones)
        if prior_sum > 1e-8:
            action_priors = [(idx, p / prior_sum) for idx, p in action_priors]
        elif action_priors:
            uniform = 1.0 / len(action_priors)
            action_priors = [(idx, uniform) for idx, _ in action_priors]

        node.expand(action_priors, child_states)

    def _terminal_value(self, game_state: "GameState") -> float:  # type: ignore[name-defined]
        """Return the value of a terminal state from the current player's perspective.

        Returns +1 if the current player won, -1 if they lost, 0 for a tie.
        """
        winner = game_state.winner
        current_player = game_state.turn
        if winner == self.game.RESULT_TIE:
            return 0.0
        if winner == current_player:
            return 1.0
        return -1.0

    def _add_dirichlet_noise(self, root: MCTSNode) -> None:
        """Mix Dirichlet noise into the root node's child priors."""
        num_children = len(root.children)
        noise = np.random.dirichlet([self.config.dirichlet_alpha] * num_children)
        epsilon = self.config.dirichlet_epsilon
        for child_node, noise_value in zip(root.children.values(), noise):
            child_node.prior_probability = (
                (1 - epsilon) * child_node.prior_probability + epsilon * noise_value
            )

    def _build_action_probs(self, temperature: float) -> List[Tuple[MoveAction, float]]:
        """Convert root child visit counts into a probability distribution."""
        current_player = self.game.state.turn

        children_items = list(self._root.children.items())
        visit_counts = np.array([child.visit_count for _, child in children_items], dtype=np.float64)

        if temperature == 0.0 or temperature < 1e-8:
            # Greedy: put all probability mass on the most-visited action
            best_idx = int(np.argmax(visit_counts))
            probs = np.zeros(len(visit_counts))
            probs[best_idx] = 1.0
        else:
            # Soften visit counts by temperature
            visit_counts_tempered = visit_counts ** (1.0 / temperature)
            probs = visit_counts_tempered / visit_counts_tempered.sum()

        action_probs: List[Tuple[MoveAction, float]] = []
        for (action_index, _child), prob in zip(children_items, probs):
            action = self.game.index_to_action(action_index, current_player)
            action_probs.append((action, float(prob)))

        return action_probs


# ---------------------------------------------------------------------------
# Tree visualisation
# ---------------------------------------------------------------------------

def visualize_tree(
    mcts: MCTS,
    output_path: str = "mcts_tree.svg",
    max_depth: int = 3,
    max_children: int = 5,
) -> str:
    """Render the MCTS search tree as an SVG (or PNG) file using Graphviz.

    SVG is the recommended format for large trees: it is a vector format and
    can be opened in any browser, scaled arbitrarily, and is never limited by
    cairo bitmap constraints.  Pass a path ending in ``.png`` to get a PNG
    instead, but beware that very large trees may trigger the cairo scaling
    warning.

    Only the subtree rooted at ``mcts._root`` is rendered.  To keep the
    graph readable, nodes deeper than *max_depth* and nodes beyond the
    top-*max_children* most-visited children at each level are pruned.

    Parameters
    ----------
    mcts:
        An ``MCTS`` instance after at least one ``get_action_probs`` call so
        that ``_root`` is populated.
    output_path:
        Destination file path.  Supported extensions: ``.svg`` (default,
        recommended) and ``.png``.  The correct extension is added
        automatically if missing.
    max_depth:
        Maximum tree depth to render (root = depth 0).
    max_children:
        Maximum number of children to show per node (ranked by visit count).

    Returns
    -------
    str
        Absolute path of the saved file.

    Raises
    ------
    ImportError
        If the ``graphviz`` Python package is not installed.
    RuntimeError
        If ``mcts._root`` is ``None`` (tree has not been searched yet).
    """
    try:
        import graphviz
    except ImportError as exc:
        raise ImportError(
            "The 'graphviz' package is required for tree visualisation.\n"
            "Install it with:  pip install graphviz\n"
            "You also need the Graphviz system binaries: https://graphviz.org/download/"
        ) from exc

    if mcts._root is None:
        raise RuntimeError(
            "MCTS tree is empty. Call get_action_probs() at least once before visualising."
        )

    # Determine output format from the file extension.
    # SVG is strongly preferred: it is vector-based and avoids cairo bitmap limits.
    if output_path.lower().endswith(".png"):
        render_format = "png"
        if not output_path.endswith(".png"):
            output_path += ".png"
    else:
        render_format = "svg"
        if not output_path.lower().endswith(".svg"):
            output_path += ".svg"

    # Graphviz render() appends the format as an extension, so strip it first.
    output_base = output_path[: -len(f".{render_format}")]

    # Graph-level attributes.
    # For SVG we skip "dpi" and "size" — Graphviz sizes SVG in points and
    # browsers handle infinite zoom natively, so no canvas limit exists.
    # For PNG we cap the canvas and use ratio=compress to prevent cairo overflow.
    if render_format == "svg":
        extra_graph_attrs: dict = {
            "nodesep": "0.4",
            "ranksep": "0.6",
        }
    else:
        canvas_size = f"{min(max_children * max_depth * 2, 40)},{min(max_depth * 6, 40)}"
        extra_graph_attrs = {
            "size": canvas_size,
            "ratio": "compress",
            "dpi": "96",
            "nodesep": "0.3",
            "ranksep": "0.5",
        }

    dot = graphviz.Digraph(
        name="mcts_tree",
        graph_attr={
            "rankdir": "TB",
            "fontname": "Helvetica",
            "bgcolor": "#1e1e1e",
            **extra_graph_attrs,
        },
        node_attr={
            "shape": "box",
            "style": "filled,rounded",
            "fontname": "Helvetica",
            "fontsize": "10",
            "margin": "0.12,0.07",
        },
        edge_attr={"fontname": "Helvetica", "fontsize": "9"},
    )

    def node_label(node: MCTSNode) -> str:
        """Node shows only statistics; the move that led here is on the edge."""
        lines: List[str] = []
        player_turn = node.game_state.turn if node.game_state else 0
        lines.append(f"P{player_turn}")
        lines.append(f"N={node.visit_count}")
        lines.append(f"Q={node.mean_value:+.3f}")
        if node.game_state and node.game_state.is_game_over:
            lines.append("TERMINAL")
        return "\n".join(lines)

    def edge_label(child_node: MCTSNode, action_index: int) -> str:
        """Edge shows the move taken and its prior probability."""
        # The move was made by the parent's player, i.e. the opponent of
        # the child's current player.  We derive that player from the parent.
        parent = child_node.parent
        parent_turn = parent.game_state.turn if (parent and parent.game_state) else 0
        try:
            action = mcts.game.index_to_action(action_index, parent_turn)
            if action.extra == "pass":
                move_str = f"P{parent_turn}:pass"
            else:
                move_str = f"P{parent_turn}:({action.row},{action.col})"
        except Exception:
            move_str = f"P{parent_turn}:act{action_index}"
        return f"{move_str}\nprior={child_node.prior_probability:.2f}"

    def node_fill_color(node: MCTSNode) -> str:
        """Color nodes by Q-value: green = winning, red = losing, grey = neutral."""
        value = node.mean_value
        if value > 0.5:
            return "#2d6a4f"   # dark green
        if value < -0.5:
            return "#7b2d2d"   # dark red
        return "#3a3a5c"       # neutral blue-grey

    def add_nodes(
        parent_id: str,
        node: MCTSNode,
        action_index: Optional[int],
        depth: int,
    ) -> None:
        node_id = str(id(node))
        dot.node(
            node_id,
            label=node_label(node),
            fillcolor=node_fill_color(node),
            fontcolor="#f0f0f0",
        )

        if parent_id and action_index is not None:
            dot.edge(
                parent_id,
                node_id,
                label=edge_label(node, action_index),
                color="#aaaaaa",
                fontcolor="#cccccc",
            )

        if depth >= max_depth or not node.children:
            return

        # Render only the top-N children by visit count for readability.
        top_children = sorted(
            node.children.items(),
            key=lambda item: item[1].visit_count,
            reverse=True,
        )[:max_children]

        for child_action_index, child_node in top_children:
            add_nodes(node_id, child_node, child_action_index, depth + 1)

    add_nodes("", mcts._root, None, 0)
    dot.render(output_base, format=render_format, cleanup=True)

    import os
    return os.path.abspath(output_path)
