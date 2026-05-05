"""dqn.py

Deep Q-Network (DQN) algorithm implementation.

Features:
  - ReplayBuffer with n-step return support
  - epsilon-greedy exploration with exponential / linear decay
  - Double DQN (--use_double_dqn)
  - Dueling DQN architecture (--use_dueling_dqn)
  - Huber / MSE loss, gradient clipping, learning-rate scheduling
"""
import os
import sys
import math
import random
import time
import argparse
from collections import deque
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from base_algo import BaseAlgorithm
from logger import logger, add_file_handlers, print_formated_args


class ReplayBuffer:
    """
    Experience replay buffer with n-step return support.

    Pre-allocates fixed-size numpy arrays for efficiency and uses temporary
    deques to accumulate n-step transitions before committing them.

    Parameters:
        args       : Namespace with buffer_size, batch_size, n_step, gamma.
        state_info : List of dicts, each with 'shape' and 'dtype' keys.
    """

    def __init__(self, args, state_info):
        self.buffer_size = args.buffer_size
        self.batch_size  = args.batch_size
        self.n_step      = args.n_step
        self.gamma       = args.gamma

        self.count    = 0
        self.pos      = 0
        self.discount = np.power(self.gamma, np.arange(self.n_step))

        # Temporary deques for n-step accumulation
        self.state_buffers      = [deque(maxlen=self.n_step) for _ in range(len(state_info))]
        self.next_state_buffers = [deque(maxlen=self.n_step) for _ in range(len(state_info))]
        self.action_buffer      = deque(maxlen=self.n_step)
        self.reward_buffer      = deque(maxlen=self.n_step)
        self.done_buffer        = deque(maxlen=self.n_step)

        # Pre-allocated storage arrays
        self.states      = []
        self.next_states = []
        for info in state_info:
            dtype = np.int32 if info['dtype'] == 'int' else np.float32
            self.states.append(np.zeros((self.buffer_size, *info['shape']), dtype=dtype))
            self.next_states.append(np.zeros((self.buffer_size, *info['shape']), dtype=dtype))

        self.actions = np.zeros(self.buffer_size, dtype=np.int64)
        self.rewards = np.zeros(self.buffer_size, dtype=np.float32)
        self.dones   = np.zeros(self.buffer_size, dtype=np.float32)

    def store(self, state, action, reward, next_state, done):
        """
        Store a transition, computing n-step returns when the buffer is full.

        Parameters:
            state      : List of current state arrays.
            action     : Action taken.
            reward     : Reward received.
            next_state : List of next state arrays.
            done       : Episode termination flag (0/1).
        """
        for i in range(len(state)):
            self.state_buffers[i].append(state[i])
            self.next_state_buffers[i].append(next_state[i])

        self.action_buffer.append(action)
        self.reward_buffer.append(reward)
        self.done_buffer.append(done)

        if len(self.state_buffers[0]) >= self.n_step or done:
            rewards       = np.array([r for r in self.reward_buffer])
            n_step_return = (rewards * self.discount[:len(rewards)]).sum()

            for i in range(len(self.states)):
                self.states[i][self.pos] = self.state_buffers[i][0]
            for i in range(len(self.next_states)):
                self.next_states[i][self.pos] = next_state[i]

            self.actions[self.pos] = self.action_buffer[0]
            self.rewards[self.pos] = n_step_return
            self.dones[self.pos]   = done

            self.count = min(self.count + 1, self.buffer_size)
            self.pos   = (self.pos + 1) % self.buffer_size

        if len(self.state_buffers[0]) >= self.n_step:
            self.action_buffer.popleft()
            self.reward_buffer.popleft()
            self.done_buffer.popleft()
            for i in range(len(self.state_buffers)):
                self.state_buffers[i].popleft()
                self.next_state_buffers[i].popleft()

        if done:
            self.action_buffer.clear()
            self.reward_buffer.clear()
            self.done_buffer.clear()
            for i in range(len(self.state_buffers)):
                self.state_buffers[i].clear()
                self.next_state_buffers[i].clear()

    def sample(self):
        """
        Sample a random mini-batch from the buffer.

        Returns:
            (states, actions, rewards, next_states, dones)
        """
        if self.count < self.batch_size:
            raise ValueError("Not enough experiences in buffer to sample")

        indices           = random.sample(range(self.count), self.batch_size)
        batch_states      = [arr[indices] for arr in self.states]
        batch_next_states = [arr[indices] for arr in self.next_states]

        return (
            batch_states,
            self.actions[indices],
            self.rewards[indices],
            batch_next_states,
            self.dones[indices],
        )

    def __len__(self):
        return self.count


class DQNAlgorithm(BaseAlgorithm):
    """
    Deep Q-Network algorithm.

    Supports standard DQN and Double DQN with n-step returns, epsilon-greedy
    exploration, and a periodically-synced target network.

    All training logic is identical to the original procedural implementation;
    it is simply organised as class methods on ``BaseAlgorithm``.
    """

    # ── Loss computation ──────────────────────────────────────────────────

    @staticmethod
    def calculate_loss(model, target_model, batch, args):
        """
        Compute the DQN (or Double DQN) Bellman loss for a mini-batch.

        Args:
            model        : Online Q-network.
            target_model : Target Q-network (weights periodically synced).
            batch        : (states, actions, rewards, next_states, dones) tensors.
            args         : Namespace with use_double_dqn, gamma, n_step, loss.

        Returns:
            torch.Tensor: Scalar loss value.
        """
        states, actions, rewards, next_states, dones = batch

        q_values             = model(states)
        state_action_values  = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            if not args.use_double_dqn:
                next_q_values = target_model(next_states).max(1).values.detach()
            else:
                next_actions  = model(next_states).argmax(1)
                next_q_values = (
                    target_model(next_states)
                    .detach()
                    .gather(1, next_actions.unsqueeze(1))
                    .squeeze(1)
                )

        expected_values = rewards + (args.gamma ** args.n_step) * next_q_values * (1 - dones)

        if args.loss == "huber":
            return F.huber_loss(state_action_values, expected_values)
        return F.mse_loss(state_action_values, expected_values)

    # ── Action selection ──────────────────────────────────────────────────

    @staticmethod
    def get_q_values(model, state_tensors):
        """Forward-pass *state_tensors* through *model* in eval mode."""
        model.eval()
        with torch.no_grad():
            return model(state_tensors)

    @staticmethod
    def select_action(q_values, n_actions, eps, valid_actions=None):
        """
        ε-greedy action selection.

        Args:
            q_values      : Q-value tensor of shape (1, n_actions).
            n_actions     : Total number of actions.
            eps           : Current exploration rate ε.
            valid_actions : Optional list of valid action indices. When provided,
                            random exploration and greedy selection are both
                            restricted to this subset.
        """
        action_pool = valid_actions if valid_actions is not None else list(range(n_actions))
        if random.random() < eps:
            return random.choice(action_pool)
        if valid_actions is not None:
            mask = torch.full((q_values.shape[-1],), float('-inf'), device=q_values.device)
            mask[valid_actions] = 0.0
            q_values = q_values + mask
        return q_values.argmax(-1).item()

    @staticmethod
    def update_epsilon(steps, args):
        """Decay epsilon exponentially or linearly."""
        if args.decay_type == "exp":
            return max(
                args.min_eps,
                args.min_eps + (args.max_eps - args.min_eps) * np.exp(-steps / args.decay_rate),
            )
        return max(args.min_eps, args.max_eps - steps * args.decay_rate)

    @staticmethod
    def update_lr(optimizer, episode, args):
        """Adjust optimizer LR according to the configured schedule."""
        if args.lr_scheduler == "constant":
            return
        lr = args.min_lr
        if episode < args.total_decay_episodes:
            if args.lr_scheduler == "cosine":
                cos = math.cos(math.pi * episode / args.total_decay_episodes)
                lr  = args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + cos)
            elif args.lr_scheduler == "linear":
                ratio = episode / args.total_decay_episodes
                lr    = args.lr - (args.lr - args.min_lr) * ratio
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    # ── BaseAlgorithm interface ───────────────────────────────────────────

    @classmethod
    def build_argparser(cls):
        """Return an ArgumentParser with all DQN-specific flags."""
        parser = cls.build_base_argparser()

        parser.add_argument("--buffer_size", type=int, default=200000,
                            help="Maximum capacity of the experience replay buffer")
        parser.add_argument("--min_train_buffer_size", type=int, default=10000,
                            help="Minimum experiences required before starting training")
        parser.add_argument("--batch_size", type=int, default=64,
                            help="Number of experiences sampled per training iteration")
        parser.add_argument("--n_episodes", type=int, default=100000,
                            help="Total training episodes to execute")
        parser.add_argument("--update_frequency", type=int, default=2,
                            help="Perform network updates every N environment steps")
        parser.add_argument("--n_updates", type=int, default=2,
                            help="Consecutive gradient updates per training step")
        parser.add_argument("--max_eps", type=float, default=0.999,
                            help="Maximum exploration rate (ε) starting value")
        parser.add_argument("--min_eps", type=float, default=0.01,
                            help="Minimum exploration rate (ε) lower bound")
        parser.add_argument("--decay_rate", type=float, default=20000,
                            help="Controls ε decay speed")
        parser.add_argument("--decay_type", type=str, choices=["exp", "linear"], default="exp")
        parser.add_argument("--gamma", type=float, default=0.99,
                            help="Discount factor for future rewards")
        parser.add_argument("--target_update_frequency", type=int, default=2000,
                            help="Steps between syncing main and target networks")
        parser.add_argument("--use_double_dqn", default=False, action='store_true',
                            help="Enable Double DQN (reduces overestimation bias)")
        parser.add_argument("--loss", type=str, choices=["huber", "mse"], default="mse",
                            help="Loss function: 'huber' or 'mse'")
        parser.add_argument("--n_step", type=int, default=1,
                            help="Multi-step reward accumulation horizon")
        parser.add_argument("--use_dueling_dqn", default=False, action='store_true',
                            help="Enable Dueling DQN architecture")
        parser.add_argument("--total_decay_episodes", type=int, default=50000,
                            help="Episodes over which to decay LR from max to min")
        parser.add_argument("--save_every_episodes", type=int, default=500,
                            help="Save a checkpoint every N completed episodes")

        return parser

    @classmethod
    def train(cls, env_cls, model_cls, args):
        """
        Run the full DQN training loop.

        Collects experience into a replay buffer, performs epsilon-greedy
        exploration, and updates the online network every ``update_frequency``
        steps.  The target network is synced every ``target_update_frequency``
        steps.
        """
        logger_path = os.path.join("logger", f"{args.task_name}.log")
        add_file_handlers(logger_path)
        print_formated_args(args)
        cls.set_seed(args.seed)

        env          = env_cls(args)
        model        = model_cls(args).to(args.device)
        target_model = model_cls(args).to(args.device)

        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Total Model Params: {total_params}")

        target_model.load_state_dict(model.state_dict())
        target_model.eval()

        optimizer      = optim.Adam(model.parameters(), lr=args.lr)
        replay_buffer  = ReplayBuffer(args, env.state_info)

        episode_rewards = []
        episode_lengths = []
        total_steps     = 0
        eps             = args.max_eps
        updates         = 0
        episode_cnt     = 0

        for episode in range(1, args.n_episodes + 1):
            state          = env.reset()
            done           = False
            episode_reward = 0
            episode_length = 0

            while not done:
                state_tensors = cls.convert_states_to_tensors(state, args.device)
                state_tensors = [t.unsqueeze(0) for t in state_tensors]
                q_values      = cls.get_q_values(model, state_tensors)
                action        = cls.select_action(q_values, env.n_actions, eps)

                next_state, reward, done = env.step(action)
                replay_buffer.store(state, action, reward, next_state, done)

                state          = next_state
                episode_reward += reward
                episode_length += 1

                if len(replay_buffer) < args.min_train_buffer_size:
                    continue

                eps         = cls.update_epsilon(total_steps, args)
                total_steps += 1

                if total_steps % args.update_frequency == 0:
                    for _ in range(args.n_updates):
                        states, actions, rewards, next_states, dones = replay_buffer.sample()
                        batch = (
                            cls.convert_states_to_tensors(states, device=args.device),
                            torch.tensor(actions, dtype=torch.long,  device=args.device),
                            torch.tensor(rewards, dtype=torch.float, device=args.device),
                            cls.convert_states_to_tensors(next_states, device=args.device),
                            torch.tensor(dones,   dtype=torch.float, device=args.device),
                        )
                        optimizer.zero_grad()
                        loss = cls.calculate_loss(model, target_model, batch, args)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                        optimizer.step()
                        updates += 1

                if total_steps % args.target_update_frequency == 0:
                    target_model.load_state_dict(model.state_dict())

            if len(replay_buffer) >= args.min_train_buffer_size:
                episode_cnt += 1
                cls.update_lr(optimizer, episode_cnt, args)

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            logger.info(
                f"episode {episode}, len: {episode_length}, reward: {episode_reward}, "
                f"eps: {eps}, updates: {updates}, buffer: {len(replay_buffer)}"
            )

            if episode % args.save_every_episodes == 0:
                avg_reward = sum(episode_rewards) / episode
                avg_length = sum(episode_lengths) / episode
                max_reward = max(episode_rewards)
                logger.info("save model now.")
                logger.info(
                    f"avg_reward: {avg_reward}, avg_length: {avg_length}, max_reward: {max_reward}"
                )
                save_dir = os.path.join(args.save_path, f"episode-{episode}")
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, "model_weights")
                torch.save(model.state_dict(), save_path)
                logger.info("save model done.")

    @classmethod
    def test_play(cls, env_cls, model_cls, args):
        """Run a trained DQN agent (or human) in the environment."""
        env   = env_cls(args)
        model = None
        if not args.human:
            model = model_cls(args).to(args.device)
            model.load_state_dict(torch.load(args.model_path, map_location="cpu"))

        running     = True
        state       = env.reset()
        done        = False
        frame_count = 0
        action      = None
        env.render()

        while True:
            frame_count += 1
            if args.human:
                running, detect_action = env.process_input()
                if detect_action is not None:
                    action = detect_action
            else:
                if frame_count == 1:
                    state_tensors = cls.convert_states_to_tensors(state, args.device)
                    state_tensors = [t.unsqueeze(0) for t in state_tensors]
                    q_values      = cls.get_q_values(model, state_tensors)
                    valid_actions = env.get_valid_actions() if args.mask_invalid_actions else None
                    action        = cls.select_action(q_values, env.n_actions, args.min_eps, valid_actions)
                    action_name   = env.act2str[action]
                    flat_q        = q_values.flatten()
                    q_values_dic  = {env.act2str[a]: flat_q[a].item() for a in env.act2str}
                    logger.info(f"q_values: {q_values_dic}, action: {action_name}, score: {env.score}")
                if env.render_mode == "gui":
                    running, _ = env.process_input()

            if not running:
                return

            if env.render_mode == "text" or frame_count >= args.fps // min(30, args.speed):
                frame_count = 0
                state, reward, done = env.step(action)
                action = None

            env.render()

            if done:
                time.sleep(1)
                state = env.reset()
                done  = False
                env.render()
                frame_count = 0

            if env.render_mode == "gui":
                env.clock.tick(env.fps)
