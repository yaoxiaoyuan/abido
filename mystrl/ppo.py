"""ppo.py

Proximal Policy Optimization (PPO) algorithm implementation.

Features:
  - RolloutBuffer with (n_envs, rollout_steps) layout for vectorized GAE
  - Generalized Advantage Estimation (GAE, lambda=0.95)
  - Clipped surrogate objective, value loss, entropy bonus
  - Multi-environment parallel rollout via VectorizedEnv (--num_envs)
  - Learning-rate scheduling (cosine / linear / constant)
"""
import os
import sys
import math
import time
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from base_algo import BaseAlgorithm
from base_game import VectorizedEnv
from logger import logger, add_file_handlers, print_formated_args


class RolloutBuffer:
    """
    Fixed-size rollout buffer for PPO with vectorized GAE computation.

    Stores transitions in ``(n_envs, rollout_steps, ...)`` layout so that
    Generalized Advantage Estimation can be computed for all environments in
    a single vectorized pass instead of a per-step Python loop.

    Parameters:
        rollout_steps (int)       : Environment steps per rollout per env.
        n_envs        (int)       : Number of parallel environments.
        state_info    (list[dict]): Each dict has 'shape' and 'dtype' keys.
        gamma         (float)     : Discount factor γ.
        gae_lambda    (float)     : GAE smoothing parameter λ.
    """

    def __init__(self, rollout_steps, n_envs, state_info, gamma, gae_lambda):
        self.rollout_steps = rollout_steps
        self.n_envs        = n_envs
        self.gamma         = gamma
        self.gae_lambda    = gae_lambda
        self.state_info    = state_info
        self._reset_storage()

    def _reset_storage(self):
        """Allocate fresh numpy arrays shaped (n_envs, rollout_steps, ...) for one rollout."""
        self.states = [
            np.zeros(
                (self.n_envs, self.rollout_steps, *info['shape']),
                dtype=np.int32 if info['dtype'] == 'int' else np.float32,
            )
            for info in self.state_info
        ]
        self.actions   = np.zeros((self.n_envs, self.rollout_steps), dtype=np.int64)
        self.rewards   = np.zeros((self.n_envs, self.rollout_steps), dtype=np.float32)
        self.dones     = np.zeros((self.n_envs, self.rollout_steps), dtype=np.float32)
        self.values    = np.zeros((self.n_envs, self.rollout_steps), dtype=np.float32)
        self.log_probs = np.zeros((self.n_envs, self.rollout_steps), dtype=np.float32)
        self.step = 0

    def store_step(self, step, states_batch, actions_batch, rewards_batch,
                   dones_batch, values_batch, log_probs_batch):
        """
        Store one time-step of transitions for all environments.

        Args:
            step          (int)            : Current rollout step index.
            states_batch  (list[np.ndarray]): List of arrays each shaped (n_envs, *d).
            actions_batch (np.ndarray)     : Shape (n_envs,).
            rewards_batch (np.ndarray)     : Shape (n_envs,).
            dones_batch   (np.ndarray)     : Shape (n_envs,), float 0/1.
            values_batch  (np.ndarray)     : Shape (n_envs,).
            log_probs_batch (np.ndarray)   : Shape (n_envs,).
        """
        for i, arr in enumerate(states_batch):
            self.states[i][:, step] = arr
        self.actions[:, step]   = actions_batch
        self.rewards[:, step]   = rewards_batch
        self.dones[:, step]     = dones_batch
        self.values[:, step]    = values_batch
        self.log_probs[:, step] = log_probs_batch
        self.step = step + 1

    def is_full(self):
        return self.step >= self.rollout_steps

    def compute_returns_and_advantages(self, last_values):
        """
        Compute discounted returns and GAE advantages in a single vectorized pass.

        All environments are processed simultaneously — no Python loop over steps.

        Args:
            last_values (np.ndarray): Bootstrap V(s_{T+1}), shape (n_envs,).
        """
        # advantages shape: (n_envs, rollout_steps)
        advantages = np.zeros((self.n_envs, self.rollout_steps), dtype=np.float32)
        last_gae   = np.zeros(self.n_envs, dtype=np.float32)

        for step in reversed(range(self.rollout_steps)):
            if step == self.rollout_steps - 1:
                next_value = last_values                       # (n_envs,)
            else:
                next_value = self.values[:, step + 1]          # (n_envs,)

            non_terminal = 1.0 - self.dones[:, step]           # (n_envs,)
            delta    = (self.rewards[:, step]
                        + self.gamma * next_value * non_terminal
                        - self.values[:, step])                # (n_envs,)
            last_gae = delta + self.gamma * self.gae_lambda * non_terminal * last_gae
            advantages[:, step] = last_gae

        self.returns    = advantages + self.values   # (n_envs, rollout_steps)
        self.advantages = advantages                 # (n_envs, rollout_steps)

    def get_batches(self, batch_size, device):
        """
        Flatten (n_envs × rollout_steps) samples, shuffle, and yield mini-batches.

        Yields:
            (states, actions, old_log_probs, returns, advantages) — all tensors.
        """
        total = self.n_envs * self.rollout_steps
        indices = np.random.permutation(total)

        # Flatten (n_envs, rollout_steps, ...) -> (total, ...)
        flat_states = [arr.reshape(total, *arr.shape[2:]) for arr in self.states]
        flat_actions   = self.actions.reshape(total)
        flat_log_probs = self.log_probs.reshape(total)
        flat_returns   = self.returns.reshape(total)
        flat_advantages = self.advantages.reshape(total)

        for start in range(0, total, batch_size):
            idx = indices[start : start + batch_size]
            batch_states = [
                torch.from_numpy(arr[idx]).to(device).long()
                if np.issubdtype(arr.dtype, np.integer)
                else torch.from_numpy(arr[idx]).to(device).float()
                for arr in flat_states
            ]
            yield (
                batch_states,
                torch.from_numpy(flat_actions[idx]).long().to(device),
                torch.from_numpy(flat_log_probs[idx]).float().to(device),
                torch.from_numpy(flat_returns[idx]).float().to(device),
                torch.from_numpy(flat_advantages[idx]).float().to(device),
            )

    def reset(self):
        self._reset_storage()


class PPOAlgorithm(BaseAlgorithm):
    """
    Proximal Policy Optimization (PPO) algorithm.

    Implements the clipped surrogate objective with GAE advantage estimation,
    a value (critic) loss, and an entropy bonus for exploration.

    The Actor-Critic model must implement:
        forward(state) -> (logits, value)
    """

    # ── Loss computation ──────────────────────────────────────────────────

    @staticmethod
    def calculate_loss(model, batch, args):
        """
        Compute the clipped PPO surrogate loss.

        Total loss = policy_loss + value_loss_coef * value_loss - entropy_coef * entropy

        Args:
            model : Actor-Critic network; forward(state) -> (logits, value).
            batch : (states, actions, old_log_probs, returns, advantages).
            args  : Namespace with clip_eps, value_loss_coef, entropy_coef.

        Returns:
            Tuple (total_loss, policy_loss, value_loss, entropy).
        """
        states, actions, old_log_probs, returns, advantages = batch

        logits, values = model(states)
        values         = values.squeeze(-1)

        dist          = torch.distributions.Categorical(logits=logits)
        new_log_probs = dist.log_prob(actions)
        entropy       = dist.entropy().mean()

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        ratio               = torch.exp(new_log_probs - old_log_probs)
        surrogate_unclipped = ratio * advantages
        surrogate_clipped   = torch.clamp(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps) * advantages
        policy_loss         = -torch.min(surrogate_unclipped, surrogate_clipped).mean()

        value_loss = F.mse_loss(values, returns)
        total_loss = policy_loss + args.value_loss_coef * value_loss - args.entropy_coef * entropy

        # Approximate KL divergence for early stopping
        with torch.no_grad():
            approx_kl = ((ratio - 1) - (new_log_probs - old_log_probs)).mean()

        return total_loss, policy_loss, value_loss, entropy, approx_kl

    # ── Action selection ──────────────────────────────────────────────────

    @staticmethod
    def select_action_greedy(model, state_tensors, valid_actions=None):
        """
        Select the highest-probability action for evaluation.

        Args:
            valid_actions : Optional list of valid action indices to restrict
                            greedy selection to.
        """
        model.eval()
        with torch.no_grad():
            logits, _ = model(state_tensors)
        if valid_actions is not None:
            mask = torch.full_like(logits, float('-inf'))
            mask[0, valid_actions] = 0.0
            logits = logits + mask
        return logits.argmax(-1).item()

    @staticmethod
    def update_lr(optimizer, rollout_idx, args):
        """Adjust optimizer LR according to the configured schedule.

        LR decay is measured in rollout iterations (not episodes).
        ``args.total_decay_steps`` controls how many rollouts to decay over.
        """
        if args.lr_scheduler == "constant":
            return
        lr = args.min_lr
        if rollout_idx < args.total_decay_steps:
            if args.lr_scheduler == "cosine":
                cos = math.cos(math.pi * rollout_idx / args.total_decay_steps)
                lr  = args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + cos)
            elif args.lr_scheduler == "linear":
                ratio = rollout_idx / args.total_decay_steps
                lr    = args.lr - (args.lr - args.min_lr) * ratio
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    # ── BaseAlgorithm interface ───────────────────────────────────────────

    @classmethod
    def build_argparser(cls):
        """Return an ArgumentParser with all PPO-specific flags."""
        parser = cls.build_base_argparser()

        parser.add_argument("--rollout_steps", type=int, default=2048,
                            help="Environment steps collected per PPO update")
        parser.add_argument("--batch_size", type=int, default=64,
                            help="Mini-batch size for PPO gradient updates")
        parser.add_argument("--ppo_epochs", type=int, default=3,
                            help="Number of gradient passes over each rollout")
        parser.add_argument("--n_rollouts", type=int, default=100000,
                            help="Total rollout iterations (each collects rollout_steps × num_envs transitions)")
        parser.add_argument("--clip_eps", type=float, default=0.2,
                            help="PPO clipping parameter ε")
        parser.add_argument("--gamma", type=float, default=0.99,
                            help="Discount factor γ")
        parser.add_argument("--gae_lambda", type=float, default=0.95,
                            help="GAE smoothing parameter λ")
        parser.add_argument("--value_loss_coef", type=float, default=0.5,
                            help="Coefficient for the value (critic) loss term")
        parser.add_argument("--entropy_coef", type=float, default=0.01,
                            help="Coefficient for the entropy bonus")
        parser.add_argument("--num_envs", type=int, default=8,
                            help="Number of parallel environments for data collection. "
                                 "Total data per rollout = rollout_steps * num_envs. "
                                 "num_envs=1 is identical to the original single-env behaviour.")

        parser.add_argument("--print_every_steps", type=int, default=100,
                            help="Print training loss every N gradient update steps within a rollout")
        parser.add_argument("--save_every_rollout_steps", type=int, default=100,
                            help="Save a checkpoint every N rollout iterations")
        parser.add_argument("--target_kl", type=float, default=None,
                            help="KL divergence threshold for early stopping within ppo_epochs. "
                                 "If the approximate KL exceeds this value, the current epoch loop "
                                 "is aborted early. Typical values: 0.01 ~ 0.05. "
                                 "Set to None (default) to disable.")
        parser.add_argument("--total_decay_steps", type=int, default=1000,
                            help="Number of rollout iterations over which to decay LR from max to min. "
                                 "Replaces the episode-based total_decay_episodes for PPO.")

        # Override base defaults for PPO
        parser.set_defaults(lr=3e-4, grad_clip=0.5, task_name="run_ppo")

        return parser

    @classmethod
    def _collect_rollout_vec(cls, vec_env, model, rollout_buffer, args,
                             states, dones, episode_lengths):
        """
        Collect one rollout from ``num_envs`` parallel environments.

        Each step, all ``num_envs`` environments are stepped simultaneously.
        Transitions are stored via ``store_step`` into the ``(n_envs, rollout_steps)``
        buffer layout, enabling fully vectorized GAE computation afterwards.
        Bootstrap values are computed per-env for correctness.

        Returns updated (states, dones, episode_lengths, episodes_finished).
        """
        num_envs          = vec_env.num_envs
        episodes_finished = 0
        total_reward      = 0.0
        rollout_scores:   list[float] = []
        rollout_buffer.reset()

        for current_step in range(rollout_buffer.rollout_steps):
            # ── Batch forward: stack all env states into a single batch ──
            batch_state_tensors = []
            for array_idx in range(len(states[0])):
                stacked = np.stack([states[env_idx][array_idx] for env_idx in range(num_envs)])
                if np.issubdtype(stacked.dtype, np.integer):
                    batch_state_tensors.append(torch.from_numpy(stacked).long().to(args.device))
                else:
                    batch_state_tensors.append(torch.from_numpy(stacked).float().to(args.device))

            model.eval()
            with torch.no_grad():
                logits, values = model(batch_state_tensors)
                if args.mask_invalid_actions:
                    # Apply per-env action masks: invalid actions -> -inf
                    action_mask = torch.zeros_like(logits)
                    for env_idx in range(num_envs):
                        valid = vec_env.envs[env_idx].get_valid_actions()
                        invalid_mask = torch.full((logits.shape[-1],), float('-inf'), device=logits.device)
                        invalid_mask[valid] = 0.0
                        action_mask[env_idx] = invalid_mask
                    logits = logits + action_mask
                dist      = torch.distributions.Categorical(logits=logits)
                actions   = dist.sample()
                log_probs = dist.log_prob(actions)

            actions_np   = actions.cpu().numpy()             # (n_envs,)
            log_probs_np = log_probs.cpu().numpy()           # (n_envs,)
            values_np    = values.squeeze(-1).cpu().numpy()  # (n_envs,)

            results = vec_env.step_all(actions_np.tolist())

            rewards_np = np.zeros(num_envs, dtype=np.float32)
            dones_np   = np.zeros(num_envs, dtype=np.float32)
            # Build per-array stacked states for store_step: list of (n_envs, *d)
            states_batch = [
                np.stack([states[env_idx][array_idx] for env_idx in range(num_envs)])
                for array_idx in range(len(states[0]))
            ]

            for env_idx, (next_state, reward, done) in enumerate(results):
                episode_lengths[env_idx] += 1
                if (args.max_episode_steps is not None
                        and episode_lengths[env_idx] >= args.max_episode_steps):
                    done = True

                rewards_np[env_idx] = reward
                dones_np[env_idx]   = float(done)
                total_reward       += reward

                if done:
                    episodes_finished += 1
                    rollout_scores.append(float(vec_env.envs[env_idx].score))
                    states[env_idx] = vec_env.envs[env_idx].reset()
                    dones[env_idx]  = False
                    episode_lengths[env_idx] = 0
                else:
                    states[env_idx] = next_state
                    dones[env_idx]  = done

            rollout_buffer.store_step(
                current_step,
                states_batch,
                actions_np,
                rewards_np,
                dones_np,
                values_np,
                log_probs_np,
            )

        # ── Per-env bootstrap: batch-forward the last states of all envs ──
        batch_last_tensors = []
        for array_idx in range(len(states[0])):
            stacked = np.stack([states[env_idx][array_idx] for env_idx in range(num_envs)])
            if np.issubdtype(stacked.dtype, np.integer):
                batch_last_tensors.append(torch.from_numpy(stacked).long().to(args.device))
            else:
                batch_last_tensors.append(torch.from_numpy(stacked).float().to(args.device))

        with torch.no_grad():
            model.eval()
            _, last_values = model(batch_last_tensors)
            last_values_np = last_values.squeeze(-1).cpu().numpy()   # (n_envs,)

        last_dones_np = np.array(dones, dtype=np.float32)            # (n_envs,)
        rollout_buffer.compute_returns_and_advantages(
            last_values = last_values_np
        )

        avg_reward = total_reward / (num_envs * rollout_buffer.rollout_steps)
        return states, dones, episode_lengths, episodes_finished, avg_reward, rollout_scores

    @staticmethod
    def _save_checkpoint(rollout_idx, model, args):
        """
        Save a checkpoint every ``save_every_rollout_steps`` rollout iterations.
        """
        if rollout_idx % args.save_every_rollout_steps == 0:
            logger.info(f"save model at rollout {rollout_idx}.")
            save_dir = os.path.join(args.save_path, f"rollout-{rollout_idx}")
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(save_dir, "model_weights"))
            logger.info("save model done.")

    @classmethod
    def train(cls, env_cls, model_cls, args):
        """
        Run the full PPO training loop.

        When ``args.num_envs == 1`` (default), behaviour is identical to the
        original single-environment implementation.  When ``args.num_envs > 1``
        a ``VectorizedEnv`` is used to collect ``num_envs`` transitions per
        step, multiplying data throughput without multi-processing overhead.

        Total data per rollout = ``rollout_steps * num_envs``.
        """
        logger_path = os.path.join("logger", f"{args.task_name}.log")
        add_file_handlers(logger_path)
        print_formated_args(args)
        cls.set_seed(args.seed)

        num_envs = getattr(args, "num_envs", 1)
        env = VectorizedEnv(env_cls, num_envs, args)
        logger.info(f"Using {num_envs} parallel environment(s).")

        model = model_cls(args).to(args.device)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Total Model Params: {total_params}")

        optimizer = optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)

        rollout_buffer = RolloutBuffer(
            rollout_steps = args.rollout_steps,
            n_envs        = num_envs,
            state_info    = env.state_info,
            gamma         = args.gamma,
            gae_lambda    = args.gae_lambda,
        )

        episode_cnt = 0
        states      = env.reset_all()
        dones       = [False] * num_envs
        episode_lengths = [0] * num_envs
        recent_scores = []
      
        for rollout_idx in range(1, args.n_rollouts + 1):
            states, dones, episode_lengths, episodes_finished, avg_reward, rollout_scores = cls._collect_rollout_vec(
                env, model, rollout_buffer, args,
                states, dones, episode_lengths,
            )
            episode_cnt += episodes_finished

            cls.update_lr(optimizer, rollout_idx, args)
            recent_scores += rollout_scores
            avg_score = sum(recent_scores[-100:]) / len(recent_scores[-100:]) if recent_scores else 0.0

            log_msg = (
                f"[rollout {rollout_idx}/{args.n_rollouts}] "
                f"episodes_this_rollout: {episodes_finished}, "
                f"total_episodes: {episode_cnt}, "
                f"avg_reward: {avg_reward:.4f}, "
                f"avg_score: {avg_score:.2f}"
            )
            logger.info(log_msg)

            # ── PPO update ────────────────────────────────────────────────
            model.train()
            total_loss_sum  = 0.0
            policy_loss_sum = 0.0
            value_loss_sum  = 0.0
            entropy_sum     = 0.0
            step_cnt        = 0
            kl_early_stopped = False
            for epoch_idx in range(args.ppo_epochs):
                for batch in rollout_buffer.get_batches(args.batch_size, args.device):
                    total_loss, policy_loss, value_loss, entropy, approx_kl = cls.calculate_loss(model, batch, args)
                    optimizer.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    optimizer.step()
                    total_loss_sum  += total_loss.item()
                    policy_loss_sum += policy_loss.item()
                    value_loss_sum  += value_loss.item()
                    entropy_sum     += entropy.item()
                    step_cnt        += 1
                    if step_cnt % args.print_every_steps == 0:
                        logger.info(
                            f"[rollout {rollout_idx}/{args.n_rollouts}, step {step_cnt}] "
                            f"loss: {total_loss_sum/step_cnt:.4f}, "
                            f"policy_loss: {policy_loss_sum/step_cnt:.4f}, "
                            f"value_loss: {value_loss_sum/step_cnt:.4f}, "
                            f"entropy: {entropy_sum/step_cnt:.4f}"
                        )
                # target_kl early stopping: check after each epoch
                if args.target_kl is not None and approx_kl.item() > args.target_kl:
                    logger.info(
                        f"[rollout {rollout_idx}] early stop at epoch {epoch_idx + 1}/{args.ppo_epochs}, "
                        f"approx_kl={approx_kl.item():.4f} > target_kl={args.target_kl}"
                    )
                    kl_early_stopped = True
                    break
            if step_cnt > 0 and step_cnt % args.print_every_steps != 0:
                logger.info(
                    f"[rollout {rollout_idx}/{args.n_rollouts}, step {step_cnt}] "
                    f"loss: {total_loss_sum/step_cnt:.4f}, "
                    f"policy_loss: {policy_loss_sum/step_cnt:.4f}, "
                    f"value_loss: {value_loss_sum/step_cnt:.4f}, "
                    f"entropy: {entropy_sum/step_cnt:.4f}"
                )
            cls._save_checkpoint(rollout_idx, model, args)

    @classmethod
    def test_play(cls, env_cls, model_cls, args):
        """Run a trained PPO agent (or human) in the environment."""
        env   = env_cls(args)
        model = None
        if not args.human:
            model = model_cls(args).to(args.device)
            model.load_state_dict(torch.load(args.model_path, map_location="cpu"))

        state       = env.reset()
        done        = False
        frame_count = 0
        episode_length = 0
        action      = None
        running     = True
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
                    valid_actions = env.get_valid_actions() if args.mask_invalid_actions else None
                    action        = cls.select_action_greedy(model, state_tensors, valid_actions)
                    action_name   = env.act2str[action]
                    with torch.no_grad():
                        logits, value = model(state_tensors)
                    flat_logits  = logits.flatten()
                    logits_dic   = {env.act2str[a]: flat_logits[a].item() for a in env.act2str}
                    logger.info(
                        f"logits: {logits_dic}, action: {action_name}, "
                        f"value: {value.item():.3f}, score: {env.score}"
                    )
                if env.render_mode == "gui":
                    running, _ = env.process_input()

            if not running:
                return

            if env.render_mode == "text" or frame_count >= args.fps // min(30, args.speed):
                frame_count = 0
                state, reward, done = env.step(action)
                episode_length += 1
                if args.max_episode_steps is not None and episode_length >= args.max_episode_steps:
                    done = True
                action = None

            env.render()

            if done:
                time.sleep(1)
                state = env.reset()
                done  = False
                env.render()
                frame_count = 0
                episode_length = 0

            if env.render_mode == "gui":
                env.clock.tick(env.fps)

