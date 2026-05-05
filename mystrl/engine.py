"""engine.py

Unified training and evaluation engine.

Registers environments, algorithms, and models, then dispatches
CLI arguments to train() or test_play(). The run_game() helper
provides a one-line entry point for game scripts.
"""
import sys
import argparse


class Engine:
    """
    Central engine that wires together environments, algorithms, and models.

    Responsibilities:
    - Maintain a registry of available RL algorithms (class-level).
    - Accept a per-game model map (algo_name -> model_class).
    - Parse ``--algo`` from the CLI and dispatch to the correct algorithm.
    - Expose ``train()`` and ``test_play()`` for programmatic use.

    Class-level algorithm registry
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    The registry is shared across all Engine instances so that algorithms
    registered once (e.g. at import time) are available everywhere::

        Engine.register_algo("sac", SACAlgorithm)

    Attributes:
        env_cls                  : Environment class (subclass of BaseGame).
        model_map                : Dict mapping algo name -> model class.
        add_custom_argument_func : Optional callable that adds extra CLI args.
    """

    # Class-level registry: algo_name -> BaseAlgorithm subclass
    # Populated lazily so heavy imports only happen when needed.
    _algo_registry: dict = {}

    def __init__(self, env_cls, model_map: dict, add_custom_argument_func=None):
        """
        Args:
            env_cls                  : Environment class.
            model_map                : {algo_name: model_class} mapping.
            add_custom_argument_func : Optional callable(parser) -> parser.
        """
        self.env_cls                  = env_cls
        self.model_map                = model_map
        self.add_custom_argument_func = add_custom_argument_func

    # ── Algorithm registry ────────────────────────────────────────────────

    @classmethod
    def register_algo(cls, name: str, algo_cls):
        """
        Register an RL algorithm under *name*.

        Args:
            name     : CLI name used with ``--algo`` (e.g. ``"ppo"``).
            algo_cls : A ``BaseAlgorithm`` subclass with ``train``,
                       ``test_play``, and ``build_argparser`` class methods.
        """
        cls._algo_registry[name] = algo_cls

    @classmethod
    def _ensure_defaults_registered(cls):
        """Lazily register built-in algorithms on first use."""
        if "dqn" not in cls._algo_registry:
            from dqn import DQNAlgorithm
            cls.register_algo("dqn", DQNAlgorithm)
        if "ppo" not in cls._algo_registry:
            from ppo import PPOAlgorithm
            cls.register_algo("ppo", PPOAlgorithm)

    # ── CLI dispatch ──────────────────────────────────────────────────────

    def _parse_algo(self):
        """
        Pre-parse ``--algo`` from sys.argv without consuming other flags.

        Removes ``--algo`` and its value from sys.argv so the chosen
        algorithm's own parser sees a clean argument list.

        Returns:
            str: The selected algorithm name (default: ``"dqn"``).
        """
        self._ensure_defaults_registered()
        available = list(self._algo_registry.keys())

        pre_parser = argparse.ArgumentParser(add_help=False)
        pre_parser.add_argument(
            "--algo",
            type=str,
            default="dqn",
            choices=available,
            help=f"RL algorithm to use. Available: {available}",
        )
        known, remaining = pre_parser.parse_known_args()
        sys.argv = [sys.argv[0]] + remaining
        return known.algo

    def run(self):
        """
        Parse CLI arguments and dispatch to train or test_play.

        The ``--algo`` flag selects the algorithm; all remaining flags are
        forwarded to the algorithm's own argument parser.
        """
        algo = self._parse_algo()
        self._validate(algo)

        algo_cls  = self._algo_registry[algo]
        model_cls = self.model_map[algo]
        algo_cls.main(self.env_cls, model_cls, self.add_custom_argument_func)

    # ── Programmatic API ──────────────────────────────────────────────────

    def train(self, algo: str, args):
        """
        Run training programmatically (without CLI parsing).

        Args:
            algo : Algorithm name (must be in model_map and registry).
            args : Pre-built argument namespace.
        """
        self._ensure_defaults_registered()
        self._validate(algo)
        self._algo_registry[algo].train(self.env_cls, self.model_map[algo], args)

    def test_play(self, algo: str, args):
        """
        Run evaluation programmatically (without CLI parsing).

        Args:
            algo : Algorithm name.
            args : Pre-built argument namespace.
        """
        self._ensure_defaults_registered()
        self._validate(algo)
        self._algo_registry[algo].test_play(self.env_cls, self.model_map[algo], args)

    def build_argparser(self, algo: str):
        """
        Return the argument parser for a specific algorithm.

        Useful for building args programmatically or extending the parser
        before calling ``train()`` / ``test_play()``.

        Args:
            algo : Algorithm name.

        Returns:
            argparse.ArgumentParser
        """
        self._ensure_defaults_registered()
        self._validate(algo)
        parser = self._algo_registry[algo].build_argparser()
        if self.add_custom_argument_func:
            parser = self.add_custom_argument_func(parser)
        return parser

    def _validate(self, algo: str):
        """Raise a clear error if *algo* is missing from the registry or model_map."""
        if algo not in self._algo_registry:
            available = list(self._algo_registry.keys())
            raise ValueError(
                f"Algorithm '{algo}' is not registered. "
                f"Use Engine.register_algo() to add it. Available: {available}"
            )
        if algo not in self.model_map:
            available = list(self.model_map.keys())
            raise ValueError(
                f"No model class provided for algo '{algo}'. "
                f"Add it to the model_map. Available in model_map: {available}"
            )


# ── Convenience wrapper ───────────────────────────────────────────────────────

def run_game(env_cls, model_map: dict, add_custom_argument_func=None):
    """
    One-line entry point for game files.

    Creates an ``Engine`` and immediately calls ``run()``, dispatching to
    the algorithm selected by ``--algo`` (default: ``dqn``).

    Args:
        env_cls                  : Environment class (subclass of BaseGame).
        model_map                : {algo_name: model_class} mapping.
        add_custom_argument_func : Optional callable(parser) -> parser.

    Example::

        if __name__ == "__main__":
            run_game(TetrisGame, {"dqn": TetrisDQN, "ppo": TetrisPPO}, add_custom_argument)
    """
    Engine(env_cls, model_map, add_custom_argument_func).run()
