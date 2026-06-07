"""game_breakout.py

Breakout game environment and neural network models.

GameBreakout    : classic brick-breaking environment on a discrete grid.
BreakoutDQN     : convolutional Q-network.
BreakoutPPO     : actor-critic network for PPO training.

State representation (3 channels, shape [3, GRID_H, GRID_W]):
    Channel 0 — brick layer  : 1.0 where a brick exists, 0.0 otherwise
    Channel 1 — ball layer   : 1.0 at the ball's grid cell, 0.0 elsewhere
    Channel 2 — paddle layer : 1.0 under the paddle cells, 0.0 elsewhere

Actions: 0 = left, 1 = stay, 2 = right
"""
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from base_game import BaseGame, GUI_RENDER_AVAILABLE
from engine import run_game
from logger import logger

try:
    import pygame
except ImportError:
    pass

# ─────────────────────────────────────────────
# Rendering colour constants
# ─────────────────────────────────────────────

BG_COLOR         = (20,  20,  30)
BRICK_COLORS     = [
    (231,  76,  60),
    (230, 126,  34),
    (241, 196,  15),
    ( 39, 174,  96),
]
PADDLE_COLOR     = (52, 152, 219)
BALL_COLOR       = (236, 240, 241)
GRID_LINE_COLOR  = (40,  40,  55)
PANEL_BG_COLOR   = (30,  30,  45)
LABEL_COLOR      = (149, 165, 166)
VALUE_COLOR      = (236, 240, 241)


# ─────────────────────────────────────────────
# 1. Game environment
# ─────────────────────────────────────────────

class GameBreakout(BaseGame):
    """
    Classic Breakout environment on a discrete GRID_W × GRID_H grid.

    The ball moves one cell diagonally per step.  Collisions with walls,
    the paddle, and bricks are resolved before the state is updated.

    Reward shaping:
        +brick_reward   for each brick destroyed
        +clear_reward   bonus when all bricks are cleared
        -miss_penalty   when the ball falls below the paddle (episode ends)
        -invalid_reward for no-op when the paddle is already at the wall
    """

    def __init__(self, args):
        super().__init__()
        self.args           = args
        self.n_actions      = 3          # 0=left, 1=stay, 2=right
        self.normalize_reward = args.normalize_reward
        self.act2str = {0:"0", 1:"1",2:"2"}
        # Episode statistics
        self.cnt         = 0
        self.scores      = []
        self.highest     = 0
        self.success_cnt = 0   # episodes where all bricks were cleared

        # ── Grid dimensions from args ─────────────────────────────────────
        self.n_brick_rows = args.n_brick_rows
        self.n_brick_cols = args.n_brick_cols
        # Grid size: each brick occupies brick_width × brick_height cells.
        # grid_w = total columns = n_brick_cols × brick_width
        # grid_h = brick rows (n_brick_rows × brick_height) + open play area + paddle row
        self.grid_w = args.n_brick_cols * args.brick_width
        self.grid_h = 3 * args.n_brick_rows * args.brick_height 

        # State: 3 channels per frame × n_last_frames (frame stacking)
        self.n_last_frames = args.n_last_frames
        self.state_info = [
            {
                "shape": [3 * self.n_last_frames, self.grid_h, self.grid_w],
                "dtype": "float",
            }
        ]

        # ── Rendering size (derived from ball_size as the base unit) ────────
        # cell_size   : pixels per grid cell = ball_size
        # paddle_px_w : paddle pixel width   = paddle_width × ball_size
        # brick_px_w  : brick pixel width    = brick_width  × ball_size
        # brick_px_h  : brick pixel height   = brick_height × ball_size
        self.cell_size   = args.ball_size
        self.paddle_px_w = args.paddle_width * args.ball_size
        self.brick_px_w  = args.brick_width  * args.ball_size
        self.brick_px_h  = args.brick_height * args.ball_size
        self.win_w       = self.grid_w * self.cell_size + 200
        self.win_h       =  self.grid_h * self.cell_size + 40

        self.render_mode = args.render_mode
        if args.render_mode == "gui" and not GUI_RENDER_AVAILABLE:
            logger.warn("pygame not installed, using text render mode.")
            args.render_mode = "text"

        if args.render_mode == "gui":
            self.fps   = args.fps
            self.clock = pygame.time.Clock()
            self.screen = pygame.display.set_mode((self.win_w, self.win_h))
            if not pygame.get_init():
                pygame.init()
            pygame.display.set_caption("MystRL Breakout")
            self.font_title  = pygame.font.SysFont("Arial", 28, bold=True)
            self.font_label  = pygame.font.SysFont("Arial", 18)
            self.font_value  = pygame.font.SysFont("Arial", 22, bold=True)

        self.reset()

    # ── State helpers ────────────────────────────────────────────────────

    def create_new_state(self):
        state = []
        for info in self.state_info:
            dtype = np.int32 if info["dtype"] == "int" else np.float32
            state.append(np.zeros(info["shape"], dtype=dtype))
        return state

    def update_state(self):
        new_state = self.create_new_state()

        # Shift previous frames forward: copy channels [3:] from old state
        # into channels [:-3] of new state (drop the oldest 3-channel frame).
        if self.n_last_frames > 1:
            new_state[0][:-3] = self.state[0][3:]

        # Write current observation into the last 3 channels.
        base = 3 * (self.n_last_frames - 1)

        # Channel base+0: bricks
        for row in range(self.n_brick_rows):
            for col in range(self.n_brick_cols):
                if self.bricks[row][col]:
                    new_state[0][base][row][col] = 1.0

        # Channel base+1: ball
        ball_row = int(np.clip(self.ball_y, 0, self.grid_h - 1))
        ball_col = int(np.clip(self.ball_x, 0, self.grid_w - 1))
        new_state[0][base + 1][ball_row][ball_col] = 1.0

        # Channel base+2: paddle
        for col in range(self.paddle_x, self.paddle_x + self.paddle_grid_w):
            if 0 <= col < self.grid_w:
                new_state[0][base + 2][self.grid_h - 1][col] = 1.0

        self.state = new_state
        return self.state

    # ── Episode lifecycle ────────────────────────────────────────────────

    def reset(self):
        # Bricks: randomised initial state.
        # Each brick is independently alive with probability sampled from a
        # Beta(α, β) distribution, so the "density" itself varies per episode.
        # Alpha > Beta → most episodes start nearly full; tweak to taste.
        alpha, beta_param = 5.0, 1.5
        density = np.random.beta(alpha, beta_param)   # e.g. ~0.7-1.0 typical
        self.bricks = [
            [random.random() < density for _ in range(self.n_brick_cols)]
            for _ in range(self.n_brick_rows)
        ]
        # Ensure at least one brick exists so the episode is always playable.
        if not any(cell for row in self.bricks for cell in row):
            r = random.randrange(self.n_brick_rows)
            c = random.randrange(self.n_brick_cols)
            self.bricks[r][c] = True
        self.total_bricks = sum(cell for row in self.bricks for cell in row)

        # Ball starts in the centre of the open play area (below bricks, above paddle)
        brick_zone_h = self.n_brick_rows * self.args.brick_height
        paddle_row   = self.grid_h - 1
        self.ball_x  = self.grid_w // 2
        self.ball_y  = brick_zone_h + (paddle_row - brick_zone_h) // 2
        self.ball_dx = random.choice([1, -1])    # +1 = right, -1 = left
        self.ball_dy = -1   # -1 = up,    +1 = down

        # Paddle centred at the bottom row; width in grid cells = paddle_width (in ball units)
        self.paddle_grid_w = max(1, self.args.paddle_width)
        self.paddle_x = (self.grid_w - self.paddle_grid_w) // 2

        self.score = 0
        self.done  = False

        self.step_reward = -0.005
        self.brick_reward = 1.5
        self.state = self.create_new_state()
        self.update_state()
        return self.state

    # ── Core game logic ──────────────────────────────────────────────────

    def _move_paddle(self, action):
        """Apply paddle action and return whether the move was valid."""
        if action == 0:   # left
            if self.paddle_x > 0:
                self.paddle_x -= 1
                return True
            return False
        elif action == 2:  # right
            if self.paddle_x + self.paddle_grid_w < self.grid_w:
                self.paddle_x += 1
                return True
            return False
        return True        # stay is always valid

    def _step_ball(self):
        """
        Advance the ball by one step and resolve collisions along its path.

        The ball may move more than one grid cell per frame. Instead of only
        checking the final position, split the movement into grid-sized substeps
        and resolve the first wall, ceiling, brick, or paddle collision on the
        actual path.

        Returns:
            reward (float): reward earned this step from brick hits.
            missed (bool) : True if the ball fell below the paddle row.
        """
        reward = self.step_reward
        self.step_reward = max(-0.015, self.step_reward - 0.0001)

        paddle_row = self.grid_h - 1
        brick_width = self.args.brick_width
        brick_height = self.args.brick_height
        brick_zone_h = self.n_brick_rows * brick_height
        brick_zone_w = self.n_brick_cols * brick_width

        def sign(value):
            """Return -1, 0, or 1 for an integer velocity component."""
            if value > 0:
                return 1
            if value < 0:
                return -1
            return 0

        def brick_at(grid_y, grid_x):
            """Return True if grid cell (grid_y, grid_x) is inside a live brick."""
            if 0 <= grid_y < brick_zone_h and 0 <= grid_x < brick_zone_w:
                return self.bricks[grid_y // brick_height][grid_x // brick_width]
            return False

        def destroy_brick(grid_y, grid_x):
            """Destroy the brick at grid cell (grid_y, grid_x) and return reward."""
            row = grid_y // brick_height
            col = grid_x // brick_width
            if not self.bricks[row][col]:
                return 0.0

            self.bricks[row][col] = False
            self.score += 1
            max_row = max(self.n_brick_rows - 1, 1)
            row_multiplier = 1.0 - 0.5 * (row / max_row)
            self.step_reward = -0.005
            self.brick_reward = min(self.brick_reward + 0.1, 3)
            return self.brick_reward * row_multiplier

        def resolve_paddle(grid_x, grid_y):
            """Resolve paddle collision or miss if the path reaches paddle row."""
            if grid_y < paddle_row:
                return False, False, grid_x, grid_y

            clamped_x = int(np.clip(grid_x, 0, self.grid_w - 1))
            if not self.paddle_x <= clamped_x < self.paddle_x + self.paddle_grid_w:
                return True, True, clamped_x, paddle_row

            hit_ratio = (clamped_x - self.paddle_x) / self.paddle_grid_w
            dx_map = [-2, -1, -1, -1, 0, 0, 1, 1, 1, 2]
            dy_map = [-1, -2, -2, -2, -2, -2, -2, -2, -2, -1]
            zone_count = len(dx_map)
            zone = min(int(hit_ratio * zone_count), zone_count - 1)
            self.ball_dx = dx_map[zone]
            self.ball_dy = dy_map[zone]
            self.brick_reward = 1.5
            return True, False, clamped_x + self.ball_dx, paddle_row - 1
        
        current_x = self.ball_x
        current_y = self.ball_y
        movement_dx = self.ball_dx
        movement_dy = self.ball_dy
        abs_dx = abs(movement_dx)
        abs_dy = abs(movement_dy)
        path_steps = max(abs_dx, abs_dy, 1)
        step_sign_x = sign(movement_dx)
        step_sign_y = sign(movement_dy)

        for step_index in range(path_steps):
            previous_x_progress = (abs_dx * step_index) // path_steps
            next_x_progress = (abs_dx * (step_index + 1)) // path_steps
            previous_y_progress = (abs_dy * step_index) // path_steps
            next_y_progress = (abs_dy * (step_index + 1)) // path_steps

            delta_x = step_sign_x * (next_x_progress - previous_x_progress)
            delta_y = step_sign_y * (next_y_progress - previous_y_progress)
            next_x = current_x + delta_x
            next_y = current_y + delta_y

            if delta_x != 0:
                if next_x < 0:
                    next_x = -next_x
                    self.ball_dx = -self.ball_dx
                    step_sign_x = sign(self.ball_dx)
                elif next_x >= self.grid_w:
                    next_x = 2 * (self.grid_w - 1) - next_x
                    self.ball_dx = -self.ball_dx
                    step_sign_x = sign(self.ball_dx)

            if delta_y != 0 and next_y < 0:
                next_y = -next_y
                self.ball_dy = -self.ball_dy
                step_sign_y = sign(self.ball_dy)

            hit_x = delta_x != 0 and brick_at(current_y, next_x)
            hit_y = delta_y != 0 and brick_at(next_y, current_x)
            hit_corner = delta_x != 0 and delta_y != 0 and brick_at(next_y, next_x)

            if hit_x or hit_y or hit_corner:
                if hit_x:
                    reward += destroy_brick(current_y, next_x)
                    self.ball_dx = -self.ball_dx
                    next_x = current_x
                if hit_y:
                    reward += destroy_brick(next_y, current_x)
                    self.ball_dy = -self.ball_dy
                    next_y = current_y
                if hit_corner and not hit_x and not hit_y:
                    reward += destroy_brick(next_y, next_x)
                    self.ball_dx = -self.ball_dx
                    self.ball_dy = -self.ball_dy
                    next_x = current_x
                    next_y = current_y

                current_x = int(np.clip(next_x, 0, self.grid_w - 1))
                current_y = int(np.clip(next_y, 0, self.grid_h - 1))
                break

            reached_paddle, missed, current_x, current_y = resolve_paddle(next_x, next_y)
            if reached_paddle:
                if missed:
                    self.ball_x = current_x
                    self.ball_y = current_y
                    return reward, True

                reward = reward - self.step_reward
                self.ball_x = int(np.clip(current_x, 0, self.grid_w - 1))
                self.ball_y = int(np.clip(current_y, 0, self.grid_h - 1))
                return reward, False

            current_x = next_x
            current_y = next_y

        self.ball_x = int(np.clip(current_x, 0, self.grid_w - 1))
        self.ball_y = int(np.clip(current_y, 0, self.grid_h - 1))
        return reward, False
    
    def step(self, action):
        if action is None:
            return self.state, 0.0, False

        miss_penalty   = -10.0
        clear_reward   = 10.0

        # Move paddle
        valid_move = self._move_paddle(action)
        reward = 0.0

        # Advance ball
        ball_reward, missed = self._step_ball()
        reward += ball_reward

        done = False
        if missed:
            reward += (miss_penalty * 0.5 + miss_penalty * 0.5 * abs(self.ball_x - self.paddle_x - self.paddle_grid_w//2) / self.grid_w)
            done = True
            self.cnt += 1
            self.scores.append(self.score)
            avg = sum(self.scores[-100:]) / min(100, len(self.scores))
            bricks_left = sum(cell for row in self.bricks for cell in row)
            if bricks_left == 0:
                self.success_cnt += 1
            self.highest = max(self.highest, self.score)
            if self.cnt % 100 == 0:
                success_rate = self.success_cnt / self.cnt if self.cnt > 0 else 0.0
                logger.info(
                    f"[{self.name}] {self.cnt} game done, "
                    f"score: {self.score}, avg: {avg:.1f}, "
                    f"highest: {self.highest}, "
                    f"clear_rate: {success_rate:.2%}"
                )

        # All bricks cleared
        bricks_remaining = sum(cell for row in self.bricks for cell in row)
        if bricks_remaining == 0 and not done:
            reward += clear_reward
            done = True
            self.cnt += 1
            self.success_cnt += 1
            self.scores.append(self.score)
            avg = sum(self.scores[-100:]) / min(100, len(self.scores))
            self.highest = max(self.highest, self.score)
            if self.cnt % 100 == 0:
                success_rate = self.success_cnt / self.cnt if self.cnt > 0 else 0.0
                logger.info(
                    f"[{self.name}] {self.cnt} game done (CLEAR!), "
                    f"score: {self.score}, avg: {avg:.1f}, "
                    f"highest: {self.highest}, "
                    f"clear_rate: {success_rate:.2%}"
                )

        if self.normalize_reward:
            reward = float(np.clip(reward / 10.0, -1.0, 1.0))

        self.update_state()
        return self.state, reward, done

    # ── Input / render ───────────────────────────────────────────────────

    def process_input(self):
        if self.render_mode == "text":
            while True:
                raw = input("0: left, 1: stay, 2: right\nInput action: ")
                try:
                    action = int(raw)
                    assert action in [0, 1, 2]
                    return True, action
                except Exception:
                    print("Invalid action!")
        else:
            # Drain the event queue to catch QUIT; use get_pressed() for
            # responsive continuous movement (no key-repeat delay).
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False, None

            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                action = 0
            elif keys[pygame.K_RIGHT]:
                action = 2
            else:
                action = 1   # stay
            return True, action

    def render(self):
        if self.render_mode == "text":
            bw = self.args.brick_width
            bh = self.args.brick_height
            grid = [["." for _ in range(self.grid_w)] for _ in range(self.grid_h)]
            for brick_row in range(self.n_brick_rows):
                for brick_col in range(self.n_brick_cols):
                    if self.bricks[brick_row][brick_col]:
                        for dr in range(bh):
                            for dc in range(bw):
                                grid[brick_row * bh + dr][brick_col * bw + dc] = "#"
            grid[self.ball_y][self.ball_x] = "O"
            for col in range(self.paddle_x, self.paddle_x + self.paddle_grid_w):
                if 0 <= col < self.grid_w:
                    grid[self.grid_h - 1][col] = "="
            print("\n".join("".join(row) for row in grid))
            print(f"score: {self.score}")
            return

        self.screen.fill(BG_COLOR)

        cs = self.cell_size

        # Draw grid lines
        for col in range(self.grid_w + 1):
            x = col * cs
            pygame.draw.line(self.screen, GRID_LINE_COLOR, (x, 0), (x, self.grid_h * cs))
        for row in range(self.grid_h + 1):
            y = row * cs
            pygame.draw.line(self.screen, GRID_LINE_COLOR, (0, y), (self.grid_w * cs, y))

        # Draw bricks; each brick occupies brick_width × brick_height grid cells.
        bw = self.args.brick_width
        bh = self.args.brick_height
        for row in range(self.n_brick_rows):
            color = BRICK_COLORS[row % len(BRICK_COLORS)]
            for col in range(self.n_brick_cols):
                if self.bricks[row][col]:
                    rect = pygame.Rect(
                        col * bw * cs + 2,
                        row * bh * cs + 2,
                        self.brick_px_w - 4,
                        self.brick_px_h - 4,
                    )
                    pygame.draw.rect(self.screen, color, rect, border_radius=3)

        # Draw ball
        ball_cx = self.ball_x * cs + cs // 2
        ball_cy = self.ball_y * cs + cs // 2
        pygame.draw.circle(self.screen, BALL_COLOR, (ball_cx, ball_cy), cs // 2 - 3)

        # Draw paddle
        paddle_rect = pygame.Rect(
            self.paddle_x * cs + 2,
            (self.grid_h - 1) * cs + 4,
            self.paddle_grid_w * cs - 4,
            cs - 8,
        )
        pygame.draw.rect(self.screen, PADDLE_COLOR, paddle_rect, border_radius=4)

        # Right info panel
        panel_x = self.grid_w * cs + 10
        panel_w = self.win_w - panel_x - 10
        avg = sum(self.scores[-100:]) / max(1, min(100, len(self.scores)))
        success_rate = self.success_cnt / max(1, self.cnt)
        bricks_left  = sum(cell for row in self.bricks for cell in row)
        infos = [
            ("Games",       self.cnt),
            ("Score",       self.score),
            ("Best",        self.highest),
            ("Avg(100)",    f"{avg:.1f}"),
            ("Clear Rate",  f"{success_rate:.1%}"),
            ("Bricks Left", bricks_left),
        ]
        for index, (label, value) in enumerate(infos):
            panel_y = 20 + index * 50
            pygame.draw.rect(
                self.screen, PANEL_BG_COLOR,
                (panel_x, panel_y, panel_w, 60),
                border_radius=6,
            )
            label_surf = self.font_label.render(label, True, LABEL_COLOR)
            value_surf = self.font_value.render(str(value), True, VALUE_COLOR)
            self.screen.blit(label_surf, (panel_x + 10, panel_y + 6))
            self.screen.blit(value_surf, (panel_x + 10, panel_y + 30))

        pygame.display.flip()
        self.clock.tick(self.fps)


# ─────────────────────────────────────────────
# 2. Shared convolutional backbone
# ─────────────────────────────────────────────

def _build_conv_backbone(in_channels: int) -> nn.Sequential:
    """Strided conv backbone for default 3×64×84 Breakout frames.

    Spatial progression (default 64×84 input):
        64×84  → conv1 (8×8, stride 4) → 16×21   RF =  8
        16×21  → conv2 (4×4, stride 2) →  8×11   RF = 22
         8×11  → conv3 (3×3, stride 2) →  4× 6   RF = 38
         4× 6  → conv4 (3×3, stride 1) →  4× 6   RF = 54
         4× 6  → conv5 (3×3, stride 1) →  4× 6   RF = 70
         4× 6  → conv6 (3×3, stride 1) →  4× 6   RF = 86

    Receptive field = 86×86, fully covers the 64×84 grid.
    Output: [batch, 256, 4, 6] → pooled to [batch, 256, 4, 4] = 4096 features.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, 32, kernel_size=8, stride=4, padding=2),
        nn.ReLU(inplace=True),
        nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
    )

# Pool remaining spatial dims to 4×4 then flatten → 64*4*4 = 1024 features.
_POOL_SIZE    = 4
_FEATURE_SIZE = 256 * _POOL_SIZE * _POOL_SIZE   # 4096

# ─────────────────────────────────────────────
# 3. DQN network
# ─────────────────────────────────────────────

class BreakoutDQN(nn.Module):
    """
    Convolutional Q-network for Breakout.

    Input : state[0]  shape [batch, 3, grid_h, grid_w]
    Output: Q-values  shape [batch, n_actions]
    """

    def __init__(self, args):
        super(BreakoutDQN, self).__init__()
        n_actions   = 3
        in_channels = 3 * args.n_last_frames   # brick / ball / paddle channels

        self.conv_layers = _build_conv_backbone(in_channels)
        self.pool        = nn.AdaptiveAvgPool2d((_POOL_SIZE, _POOL_SIZE))
        self.feature_compressor = nn.Sequential(
            nn.Linear(_FEATURE_SIZE, 256),
            nn.ReLU(inplace=True),
        )

        self.use_dueling_dqn = args.use_dueling_dqn
        if self.use_dueling_dqn:
            self.value_stream     = nn.Linear(256, 1)
            self.advantage_stream = nn.Linear(256, n_actions)
        else:
            self.fc = nn.Linear(256, n_actions)

    def forward(self, state):
        x        = state[0].float()
        conv_out = self.conv_layers(x)
        pooled   = self.pool(conv_out)
        flat     = pooled.view(pooled.size(0), -1)
        features = self.feature_compressor(flat)

        if self.use_dueling_dqn:
            values     = self.value_stream(features)
            advantages = self.advantage_stream(features)
            return values + advantages - advantages.mean(1, keepdim=True)
        return self.fc(features)

# ─────────────────────────────────────────────
# 4. PPO network
# ─────────────────────────────────────────────

class BreakoutPPO(nn.Module):
    """
    Actor-Critic network for Breakout (PPO).

    Input : state[0]  shape [batch, 3, grid_h, grid_w]
    Output: (logits [batch, n_actions], value [batch, 1])
    """

    def __init__(self, args):
        super(BreakoutPPO, self).__init__()
        n_actions   = 3
        in_channels = 3 * args.n_last_frames

        self.conv_layers = _build_conv_backbone(in_channels)
        self.pool        = nn.AdaptiveAvgPool2d((_POOL_SIZE, _POOL_SIZE))
        self.feature_compressor = nn.Sequential(
            nn.Linear(_FEATURE_SIZE, 256),
            nn.ReLU(inplace=True),
        )

        self.actor  = nn.Linear(256, n_actions)
        self.critic = nn.Linear(256, 1)

    def forward(self, state):
        x        = state[0].float()
        conv_out = self.conv_layers(x)
        pooled   = self.pool(conv_out)
        flat     = pooled.view(pooled.size(0), -1)
        features = self.feature_compressor(flat)
        return self.actor(features), self.critic(features)

# ─────────────────────────────────────────────
# 5. CLI arguments
# ─────────────────────────────────────────────

def add_custom_argument(parser):
    parser.add_argument(
        "--ball_size",
        type=int,
        default=10,
        help="pixels per grid cell (ball size as base unit); controls overall scale (default: 10)",
    )
    parser.add_argument(
        "--paddle_width",
        type=int,
        default=10,
        help="paddle width as multiples of ball_size in pixels (default: 10)",
    )
    parser.add_argument(
        "--brick_width",
        type=int,
        default=6,
        help="brick width as multiples of ball_size in pixels (default: 6)",
    )
    parser.add_argument(
        "--brick_height",
        type=int,
        default=2,
        help="brick height as multiples of ball_size in pixels (default: 1)",
    )
    parser.add_argument(
        "--n_brick_rows",
        type=int,
        default=5,
        help="number of brick rows (default: 5)",
    )
    parser.add_argument(
        "--n_brick_cols",
        type=int,
        default=10,
        help="number of brick columns (default: 10)",
    )
    parser.add_argument(
        "--normalize_reward",
        action="store_true",
        default=False,
        help="clip rewards to [-1, 1] by dividing by 10 (default: False)",
    )
    return parser

# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    run_game(
        GameBreakout,
        {"dqn": BreakoutDQN, "ppo": BreakoutPPO},
        add_custom_argument,
    )
