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
        self.grid_h = 4 * args.n_brick_rows * args.brick_height 

        # State: 3 channels × grid_h × grid_w (float32)
        self.state_info = [
            {
                "shape": [3, self.grid_h, self.grid_w],
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

        # Channel 0: bricks
        for row in range(self.n_brick_rows):
            for col in range(self.n_brick_cols):
                if self.bricks[row][col]:
                    new_state[0][0][row][col] = 1.0

        # Channel 1: ball
        ball_row = int(np.clip(self.ball_y, 0, self.grid_h - 1))
        ball_col = int(np.clip(self.ball_x, 0, self.grid_w - 1))
        new_state[0][1][ball_row][ball_col] = 1.0

        # Channel 2: paddle
        for col in range(self.paddle_x, self.paddle_x + self.paddle_grid_w):
            if 0 <= col < self.grid_w:
                new_state[0][2][self.grid_h - 1][col] = 1.0

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
        Advance the ball by one step and resolve all collisions.

        Collision resolution order:
          1. Left / right wall bounce.
          2. Ceiling bounce.
          3. Brick hit (vertical bounce; only the destination cell is checked).
          4. Paddle hit or miss (checked before committing new_y).

        Returns:
            reward (float): reward earned this step from brick hits.
            missed (bool) : True if the ball fell below the paddle row.
        """
        step_reward       = -0.01
        reward = step_reward
        brick_reward = 1.0

        paddle_row   = self.grid_h - 1

        # Propose new position
        new_x = self.ball_x + self.ball_dx
        new_y = self.ball_y + self.ball_dy

        # ── Wall collisions (left / right) ───────────────────────────────
        if new_x < 0 or new_x >= self.grid_w:
            self.ball_dx = -self.ball_dx
            new_x = self.ball_x + self.ball_dx

        # ── Ceiling collision ────────────────────────────────────────────
        if new_y < 0:
            self.ball_dy = -self.ball_dy
            new_y = self.ball_y + self.ball_dy

        # ── Brick collision ──────────────────────────────────────────────
        # Ball moves in grid coordinates; each brick occupies brick_width × brick_height
        # grid cells. Convert grid position to brick index before lookup.
        # Detect x-side and y-side collisions separately to correctly reflect dx or dy.
        bw = self.args.brick_width
        bh = self.args.brick_height
        brick_zone_h = self.n_brick_rows * bh   # grid rows occupied by bricks
        brick_zone_w = self.n_brick_cols * bw   # grid cols occupied by bricks

        def brick_at(gy, gx):
            """Return True if grid cell (gy, gx) is inside a live brick."""
            if 0 <= gy < brick_zone_h and 0 <= gx < brick_zone_w:
                return self.bricks[gy // bh][gx // bw]
            return False

        def destroy_brick(gy, gx):
            """Destroy the brick at grid cell (gy, gx) and return reward."""
            self.bricks[gy // bh][gx // bw] = False
            self.score += 1
            return brick_reward

        # Check x-side collision: ball moved horizontally into a brick
        # (new_x entered a brick column but ball_y row was already clear).
        hit_x = brick_at(self.ball_y, new_x)
        # Check y-side collision: ball moved vertically into a brick
        # (new_y entered a brick row but ball_x column was already clear).
        hit_y = brick_at(new_y, self.ball_x)
        # Check corner collision: both axes entered a brick simultaneously.
        hit_corner = brick_at(new_y, new_x)

        if hit_x:
            reward += destroy_brick(self.ball_y, new_x)
            self.ball_dx = -self.ball_dx
            new_x = self.ball_x + self.ball_dx
        if hit_y:
            reward += destroy_brick(new_y, self.ball_x)
            self.ball_dy = -self.ball_dy
            new_y = self.ball_y + self.ball_dy
        if hit_corner and not hit_x and not hit_y:
            # Pure corner hit: reflect both axes
            reward += destroy_brick(new_y, new_x)
            self.ball_dx = -self.ball_dx
            self.ball_dy = -self.ball_dy
            new_x = self.ball_x + self.ball_dx
            new_y = self.ball_y + self.ball_dy

        # ── Paddle collision or miss ──────────────────────────────────────
        # The ball reaches the paddle row (GRID_H-1) or overshoots it.
        if new_y >= paddle_row:
            clamped_x = int(np.clip(new_x, 0, self.grid_w - 1))
            if self.paddle_x <= clamped_x < self.paddle_x + self.paddle_grid_w:
                # Hit the paddle — bounce upward.
                # Classic Breakout logic: ball_dx is determined purely by the
                # hit position ratio on the paddle (left half → left, right half → right).
                # This avoids the jitter caused by conditional direction correction.
                # Divide paddle into 8 zones; each zone gives a different (dx, dy).
                # Outer zones → steep angle (|dx|=2, dy=-1); inner zones → shallow (|dx|=1, dy=-2);
                # centre zone → straight up (dx=0, dy=-2).
                hit_ratio = (clamped_x - self.paddle_x) / self.paddle_grid_w  # 0.0 … 1.0
                zone = min(int(hit_ratio * 8), 8)   # 0…4
                dx_map = [-2, -2, -1, -1,  1, 1,  2, 2]
                dy_map = [-1, -1, -2, -2,  -2, -2, -1, -1]
                self.ball_dx = dx_map[zone]
                self.ball_dy = dy_map[zone]
                new_x = self.ball_x + self.ball_dx
                new_y = self.ball_y + self.ball_dy
                reward = reward - step_reward
            else:
                # Missed the paddle — let the ball fall to the bottom row before ending
                self.ball_x = int(np.clip(new_x, 0, self.grid_w - 1))
                self.ball_y = paddle_row
                return reward, True

        self.ball_x = int(np.clip(new_x, 0, self.grid_w - 1))
        self.ball_y = int(np.clip(new_y, 0, self.grid_h - 1))
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
            panel_y = 20 + index * 70
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
    """Three-layer conv backbone shared by DQN and PPO."""
    return nn.Sequential(
        nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
    )

# Pool spatial dims to 4×4 then flatten → fixed 128*4*4 = 2048 features.
_POOL_SIZE    = 4
_FEATURE_SIZE = 128 * _POOL_SIZE * _POOL_SIZE   # 2048

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
        in_channels = 3   # brick / ball / paddle channels

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
        in_channels = 3

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
        help="pixels per grid cell (ball size as base unit); controls overall scale (default: 30)",
    )
    parser.add_argument(
        "--paddle_width",
        type=int,
        default=8,
        help="paddle width as multiples of ball_size in pixels (default: 8)",
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
        default=8,
        help="number of brick rows (default: 8)",
    )
    parser.add_argument(
        "--n_brick_cols",
        type=int,
        default=14,
        help="number of brick columns (default: 14)",
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
