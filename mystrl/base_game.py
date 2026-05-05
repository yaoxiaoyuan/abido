"""base_game.py

Abstract base class for all game environments.

Defines the environment contract (reset / step / render / process_input)
and provides shared helpers: create_new_state, _init_pygame, _check_gui.
Also contains VectorizedEnv, a synchronous multi-environment wrapper
used by PPO for parallel rollout collection.
"""
from abc import ABC, abstractmethod
import numpy as np

GUI_RENDER_AVAILABLE = False
try:
    import pygame
    GUI_RENDER_AVAILABLE = True
except ImportError:
    pass


class BaseGame(ABC):
    """
    Abstract base class for MystRL game environments.

    Subclasses must implement:
        reset()         -> state
        step(action)    -> (state, reward, done)
        update_state()  -> state
        process_input() -> (running: bool, action: int | None)
        render()        -> None
    """

    # ── Subclass contract ─────────────────────────────────────────────────

    @abstractmethod
    def reset(self):
        """
        Reset the environment to its initial state.

        Returns:
            state (list[np.ndarray]): Initial observation.
        """

    @abstractmethod
    def step(self, action):
        """
        Apply *action* and advance the environment by one step.

        Args:
            action (int | None): Action index, or None for a no-op.

        Returns:
            state  (list[np.ndarray]): Next observation.
            reward (float)           : Scalar reward signal.
            done   (bool)            : True if the episode has ended.
        """

    @abstractmethod
    def update_state(self):
        """
        Rebuild ``self.state`` from the current internal game state.

        Called at the end of every ``step()`` and ``reset()``.

        Returns:
            state (list[np.ndarray]): Updated observation.
        """

    @abstractmethod
    def process_input(self):
        """
        Read player input (keyboard / stdin).

        Returns:
            running (bool)           : False if the user requested quit.
            action  (int | None)     : Detected action, or None if none.
        """

    @abstractmethod
    def render(self):
        """Render the current game state to the screen or stdout."""

    # ── Optional overrides ────────────────────────────────────────────────

    def get_valid_actions(self):
        """
        Return the list of valid action indices for the current state.

        The default implementation returns all actions (0 to n_actions-1).
        Subclasses may override this to mask illegal moves (e.g. 2048's
        invalid slides, Snake's reverse direction).

        Returns:
            list[int]: Valid action indices.
        """
        return list(range(self.n_actions))

    def __init__(self):
        """Set default instance attributes shared by all game environments."""
        self.name = self.__class__.__name__

    # ── Shared helpers ────────────────────────────────────────────────────

    def create_new_state(self):
        """
        Allocate a fresh list of zeroed numpy arrays matching ``self.state_info``.

        This is the canonical way to initialise or reset the observation
        buffer; subclasses should call it in ``reset()`` and ``update_state()``.

        Returns:
            list[np.ndarray]: One zeroed array per entry in ``state_info``.
        """
        state = []
        for info in self.state_info:
            dtype = np.int32 if info['dtype'] == 'int' else np.float32
            state.append(np.zeros(info['shape'], dtype=dtype))
        return state

    def _check_gui(self, args):
        """
        Verify that pygame is available; fall back to text mode if not.

        Should be called during ``__init__`` before any pygame calls.

        Args:
            args: Argument namespace with a ``render_mode`` attribute.
        """
        if args.render_mode == "gui" and not GUI_RENDER_AVAILABLE:
            from logger import logger
            logger.warning("pygame not installed — falling back to text render mode.")
            args.render_mode = "text"

    def _init_pygame(self, title: str, width: int, height: int):
        """
        Initialise pygame and create a display window.

        Safe to call even if pygame is already initialised.

        Args:
            title  : Window caption.
            width  : Window width in pixels.
            height : Window height in pixels.

        Returns:
            pygame.Surface: The created display surface.
        """
        if not pygame.get_init():
            pygame.init()
        screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(title)
        return screen


class VectorizedEnv:
    """
    Synchronous vectorized environment wrapper for PPO parallel data collection.

    Creates ``num_envs`` independent copies of the same environment and steps
    them all in lock-step.  Each call to ``step_all`` returns ``num_envs``
    transitions, effectively multiplying the data throughput by ``num_envs``
    without requiring multi-processing.

    Attributes are proxied from the first environment instance so that
    algorithm code can treat a ``VectorizedEnv`` like a single ``BaseGame``.

    Args:
        env_cls  : Environment class (subclass of BaseGame).
        num_envs : Number of parallel environment instances.
        args     : Argument namespace forwarded to each environment constructor.
    """

    def __init__(self, env_cls, num_envs: int, args):
        self.num_envs = num_envs
        self.envs     = [env_cls(args) for _ in range(num_envs)]
        base_name = self.envs[0].__class__.__name__
        for idx, env in enumerate(self.envs):
            env.name = f"{base_name}[{idx}]"

    # ── Attribute proxies (read from env 0) ───────────────────────────────

    @property
    def state_info(self):
        return self.envs[0].state_info

    @property
    def n_actions(self):
        return self.envs[0].n_actions

    @property
    def act2str(self):
        return self.envs[0].act2str

    @property
    def score(self):
        return self.envs[0].score

    @property
    def render_mode(self):
        return self.envs[0].render_mode

    # ── Vectorized API ────────────────────────────────────────────────────

    def reset_all(self):
        """
        Reset all environments and return their initial states.

        Returns:
            list[state]: One state per environment (each state is a
                         list of numpy arrays as returned by BaseGame.reset).
        """
        return [env.reset() for env in self.envs]

    def step_all(self, actions):
        """
        Step all environments with the corresponding action.

        Args:
            actions (list[int]): One action per environment.

        Returns:
            list[tuple]: Each element is ``(next_state, reward, done)``
                         for the corresponding environment.
        """
        return [env.step(action) for env, action in zip(self.envs, actions)]
