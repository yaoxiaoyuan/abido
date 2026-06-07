"""base_algo.py

Abstract base class for all RL algorithms.

Defines the algorithm contract (train / test_play / build_argparser) and
provides shared utilities: set_seed, convert_states_to_tensors,
build_base_argparser, and a convenience main() entry point.
"""
import argparse
import random
from abc import ABC, abstractmethod
import numpy as np
import torch


class BaseAlgorithm(ABC):
    """
    Abstract base class for MystRL RL algorithms.

    Concrete subclasses (DQNAlgorithm, PPOAlgorithm, …) implement the three
    abstract methods below.  The ``Engine`` uses these methods to run training
    or evaluation without knowing which algorithm is active.
    """

    # ── Subclass contract ─────────────────────────────────────────────────

    @classmethod
    @abstractmethod
    def build_argparser(cls) -> argparse.ArgumentParser:
        """
        Return an ArgumentParser populated with all algorithm-specific flags.

        Implementations should call ``cls.build_base_argparser()`` first and
        then add their own arguments on top.

        Returns:
            argparse.ArgumentParser
        """

    @classmethod
    @abstractmethod
    def train(cls, env_cls, model_cls, args) -> None:
        """
        Run the full training loop.

        Args:
            env_cls   : Environment class (subclass of BaseGame).
            model_cls : Neural network model class.
            args      : Parsed argument namespace.
        """

    @classmethod
    @abstractmethod
    def test_play(cls, env_cls, model_cls, args) -> None:
        """
        Run a trained agent (or human) in the environment.

        Args:
            env_cls   : Environment class.
            model_cls : Neural network model class.
            args      : Parsed argument namespace.
        """

    # ── Shared helpers ────────────────────────────────────────────────────

    @classmethod
    def build_base_argparser(cls) -> argparse.ArgumentParser:
        """
        Build an ArgumentParser pre-populated with flags shared by all algorithms.

        Shared flags cover: task naming, mode, model path, device, rendering,
        learning-rate schedule, checkpointing, and frame stacking.

        Returns:
            argparse.ArgumentParser
        """
        description = "MystRL — a simple Python RL framework."
        usage = "For training and inference, please refer to the shell scripts."
        parser = argparse.ArgumentParser(
            usage=usage,
            description=description,
            formatter_class=argparse.RawTextHelpFormatter,
        )

        # ── Identity ──────────────────────────────────────────────────────
        parser.add_argument("--task_name", type=str, default="run_rl",
                            help="Task identifier used for log file naming")

        parser.add_argument("--mode", type=str, choices=["train", "test"],
                            default="train",
                            help="'train' to train, 'test' to evaluate a saved model")

        parser.add_argument("--model_path", type=str, default=None,
                            help="Path to saved model weights (required for test mode)")

        # ── Optimiser (shared defaults) ───────────────────────────────────
        parser.add_argument("--lr", type=float, default=1e-4,
                            help="Initial learning rate")

        parser.add_argument("--min_lr", type=float, default=1e-5,
                            help="Minimum learning rate after decay")

        parser.add_argument("--lr_scheduler", type=str,
                            choices=["constant", "cosine", "linear"],
                            default="constant",
                            help="Learning rate schedule")

        parser.add_argument("--grad_clip", type=float, default=1.0,
                            help="Gradient clipping max norm")

        # ── Checkpointing ─────────────────────────────────────────────────
        parser.add_argument("--print_every_game", type=int, default=100,
                            help="Print game stats every N completed games")
        
        parser.add_argument("--max_episode_steps", type=int, default=None,
                            help="Maximum environment steps per episode. None disables the limit")
        
        parser.add_argument("--seed", type=int, default=42,
                            help="Random seed for reproducibility (random / numpy / torch)")

        parser.add_argument("--save_path", type=str, default="model",
                            help="Directory for saving model checkpoints")

        # ── Environment / rendering ───────────────────────────────────────
        parser.add_argument("--device", type=str, default="cpu",
                            help="Compute device: 'cpu' or 'cuda'")

        parser.add_argument("--n_last_frames", type=int, default=4,
                            help="Number of stacked past frames used as state input")

        parser.add_argument("--render_mode", choices=["gui", "text"], default="gui",
                            help="'gui' uses pygame; 'text' prints to stdout")

        parser.add_argument("--fps", type=int, default=60,
                            help="Target frames per second when render_mode=gui")

        parser.add_argument("--speed", type=int, default=8,
                            help="Simulation speed multiplier")

        parser.add_argument("--human", default=False, action="store_true",
                            help="Let a human play instead of the AI agent")

        parser.add_argument("--mask_invalid_actions", default=False, action="store_true",
                            help="Mask invalid actions by calling env.get_valid_actions(); "
                                 "only valid actions are considered during selection")

        return parser

    @staticmethod
    def set_seed(seed: int):
        """
        Set random seeds for reproducibility across all relevant libraries.

        Covers Python ``random``, ``numpy``, ``torch`` (CPU and CUDA).
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def convert_states_to_tensors(state_list, device):
        """
        Convert a list of numpy state arrays to appropriately typed torch tensors.

        Integer arrays become ``torch.long``; float arrays become ``torch.float``.

        Args:
            state_list (list[np.ndarray]): State arrays as returned by the env.
            device (str | torch.device) : Target device.

        Returns:
            list[torch.Tensor]
        """
        tensor_list = []
        for array in state_list:
            if np.issubdtype(array.dtype, np.integer):
                tensor = torch.from_numpy(array).long().to(device)
            else:
                tensor = torch.from_numpy(array).float().to(device)
            tensor_list.append(tensor)
        return tensor_list

    @classmethod
    def main(cls, env_cls, model_cls, add_custom_argument_func=None):
        """
        Convenience entry point: parse args and dispatch to train or test_play.

        This mirrors the module-level ``main()`` functions in dqn.py / ppo.py
        so that game files can call either style interchangeably.

        Args:
            env_cls                  : Environment class.
            model_cls                : Model class.
            add_custom_argument_func : Optional callable that adds extra CLI args.
        """
        import sys
        parser = cls.build_argparser()
        if add_custom_argument_func:
            parser = add_custom_argument_func(parser)
        args = parser.parse_args(sys.argv[1:])

        if args.mode == "train":
            cls.train(env_cls, model_cls, args)
        elif args.mode == "test":
            cls.test_play(env_cls, model_cls, args)
