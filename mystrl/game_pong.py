"""game_pong.py

Pong game environment and neural network models.

PongGame : single-player Pong environment; the agent controls the left paddle
           and plays against a simple rule-based right paddle.
PongDQN  : convolutional Q-network for DQN training.
PongPPO  : actor-critic network for PPO training.

State representation:
  A stack of (n_last_frames + 1) grayscale frames, each of shape (height, width).
  The last channel encodes the current frame; earlier channels encode past frames.

Actions:
  0 — stay, 1 — move up, 2 — move down
"""
import random
import numpy as np
import torch.nn as nn
from base_game import BaseGame, GUI_RENDER_AVAILABLE
from engine import run_game
from logger import logger

try:
    import pygame
except ImportError:
    pass

# ── Colour palette ────────────────────────────────────────────────────────────
BACKGROUND    = (15, 15, 25)
COURT_COLOR   = (25, 25, 40)
PADDLE_COLOR  = (80, 200, 200)
BALL_COLOR    = (230, 230, 100)
NET_COLOR     = (50, 50, 80)
TEXT_COLOR    = (180, 210, 255)
SCORE_COLOR   = (250, 250, 255)
PANEL_BG      = (30, 30, 50, 220)

WINDOW_WIDTH  = 800
WINDOW_HEIGHT = 600


class PongGame(BaseGame):
    """
    Single-player Pong environment.

    The agent controls the left paddle (3 actions: stay / up / down).
    The right paddle is a simple rule-based opponent that tracks the ball
    with a configurable speed limit, making it beatable.

    Episode ends when either side reaches ``args.win_score`` points.
    Reward:  +1 for scoring a point, -1 for conceding a point.
    """

    def __init__(self, args):
        super().__init__()
        self.args = args

        self.court_width  = args.width
        self.court_height = args.height

        self.state_info = [
            {
                "shape": [args.n_last_frames + 1, self.court_height, self.court_width],
                "dtype": "float",
            }
        ]

        self.n_actions = 3
        self.act2str   = {0: "stay", 1: "up", 2: "down"}

        # Paddle geometry (in grid units)
        self.paddle_height = max(2, self.court_height // 5)
        self.paddle_width  = 1

        self.ball_speed_x  = args.ball_speed
        self.ball_speed_y  = args.ball_speed
        self.opponent_speed = args.opponent_speed

        self.highest = 0
        self.scores  = []
        self.cnt     = 0
        self.score   = 0

        self.render_mode = args.render_mode
        self._check_gui(args)

        if args.render_mode == "gui":
            self.fps   = args.fps
            self.clock = pygame.time.Clock()
            self.screen = self._init_pygame(
                "MystRL Pong", WINDOW_WIDTH, WINDOW_HEIGHT
            )
            self.font_title = pygame.font.SysFont("Arial", 22)
            self.font_value = pygame.font.SysFont("Arial", 30, bold=True)

        self.reset()

    # ── Internal helpers ──────────────────────────────────────────────────

    def _reset_ball(self):
        """Place the ball at the centre with a random diagonal direction."""
        self.ball_x = self.court_width  / 2.0
        self.ball_y = self.court_height / 2.0
        direction_x = 1 if random.random() > 0.5 else -1
        direction_y = 1 if random.random() > 0.5 else -1
        self.vel_x  = self.ball_speed_x * direction_x
        self.vel_y  = self.ball_speed_y * direction_y

    def _render_frame_to_array(self):
        """
        Render the current game state into a 2-D float array of shape
        (court_height, court_width) with values in [0, 1].
        """
        frame = np.zeros((self.court_height, self.court_width), dtype=np.float32)

        # Left paddle
        for row in range(self.left_paddle_y, min(self.left_paddle_y + self.paddle_height, self.court_height)):
            frame[row, 0] = 0.8

        # Right paddle
        right_paddle_y = int(self.right_paddle_y)
        for row in range(right_paddle_y, min(right_paddle_y + self.paddle_height, self.court_height)):
            frame[row, self.court_width - 1] = 0.6

        # Ball
        ball_row = int(np.clip(self.ball_y, 0, self.court_height - 1))
        ball_col = int(np.clip(self.ball_x, 0, self.court_width  - 1))
        frame[ball_row, ball_col] = 1.0

        return frame

    def _move_opponent(self):
        """Simple rule-based right paddle: track the ball with limited speed."""
        paddle_center = self.right_paddle_y + self.paddle_height / 2.0
        if paddle_center < self.ball_y - 0.5:
            self.right_paddle_y = min(
                self.right_paddle_y + self.opponent_speed,
                self.court_height - self.paddle_height,
            )
        elif paddle_center > self.ball_y + 0.5:
            self.right_paddle_y = max(
                self.right_paddle_y - self.opponent_speed,
                0,
            )

    # ── BaseGame interface ────────────────────────────────────────────────

    def reset(self):
        self.left_paddle_y  = (self.court_height - self.paddle_height) // 2
        self.right_paddle_y = (self.court_height - self.paddle_height) // 2
        self.agent_score    = 0
        self.opponent_score = 0
        self.score          = 0
        self._reset_ball()

        self.state = self.create_new_state()
        self.update_state()
        return self.state

    def update_state(self):
        new_state = self.create_new_state()
        # Shift past frames
        new_state[0][:-1] = self.state[0][1:]
        # Write current frame into the last channel
        new_state[0][-1]  = self._render_frame_to_array()
        self.state = new_state
        return self.state

    def step(self, action):
        # Move agent paddle
        if action == 1:
            self.left_paddle_y = max(self.left_paddle_y - 1, 0)
        elif action == 2:
            self.left_paddle_y = min(
                self.left_paddle_y + 1, self.court_height - self.paddle_height
            )

        # Move opponent paddle
        self._move_opponent()

        # Move ball with sub-step CCD to prevent tunnelling through paddles
        prev_x = self.ball_x
        prev_y = self.ball_y
        self.ball_x += self.vel_x
        self.ball_y += self.vel_y

        reward = 0.0
        done   = False

        # Top / bottom wall bounce
        if self.ball_y <= 0:
            self.ball_y = 0
            self.vel_y  = abs(self.vel_y)
        elif self.ball_y >= self.court_height - 1:
            self.ball_y = self.court_height - 1
            self.vel_y  = -abs(self.vel_y)

        # Interpolate ball_y at the moment it crosses the paddle boundary
        def _interp_y_at_x(target_x):
            """Linearly interpolate ball_y at the moment ball_x == target_x."""
            if abs(self.vel_x) < 1e-9:
                return self.ball_y
            t = (target_x - prev_x) / self.vel_x
            return prev_y + t * self.vel_y

        # Left paddle collision (agent): ball crossed x=1 from right to left
        if prev_x > 1 and self.ball_x <= 1:
            hit_y = _interp_y_at_x(1.0)
            if self.left_paddle_y <= hit_y <= self.left_paddle_y + self.paddle_height:
                self.ball_x = 1
                self.ball_y = hit_y
                self.vel_x  = abs(self.vel_x)
                hit_offset  = hit_y - (self.left_paddle_y + self.paddle_height / 2.0)
                self.vel_y  = hit_offset * 0.4
            elif self.ball_x < 0:
                self.opponent_score += 1
                reward = -1.0
                self._reset_ball()

        # Right paddle collision (opponent): ball crossed x=court_width-2 from left to right
        elif prev_x < self.court_width - 2 and self.ball_x >= self.court_width - 2:
            hit_y = _interp_y_at_x(float(self.court_width - 2))
            if self.right_paddle_y <= hit_y <= self.right_paddle_y + self.paddle_height:
                self.ball_x = self.court_width - 2
                self.ball_y = hit_y
                self.vel_x  = -abs(self.vel_x)
                hit_offset  = hit_y - (self.right_paddle_y + self.paddle_height / 2.0)
                self.vel_y  = hit_offset * 0.4
            elif self.ball_x >= self.court_width:
                self.agent_score += 1
                reward = 1.0
                self.score += 1
                self._reset_ball()

        # Ball exits left (no paddle hit)
        elif self.ball_x < 0:
            self.opponent_score += 1
            reward = -1.0
            self._reset_ball()

        # Ball exits right (no paddle hit)
        elif self.ball_x >= self.court_width:
            self.agent_score += 1
            reward = 1.0
            self.score += 1
            self._reset_ball()

        # Episode end
        if self.agent_score >= self.args.win_score or self.opponent_score >= self.args.win_score:
            done = True

        self.update_state()

        if done:
            self.cnt += 1
            self.scores.append(self.score)
            self.highest = max(self.highest, self.score)
            avg = sum(self.scores[-100:]) / min(100, len(self.scores))
            if self.cnt % self.args.print_every_game == 0:
                logger.info(
                    f"[{self.name}] {self.cnt} game done, "
                    f"score: {self.score}, avg: {avg:.2f}, highest: {self.highest}"
                )

        return self.state, reward, done

    def process_input(self):
        if self.render_mode == "text":
            while True:
                raw = input("0: stay, 1: up, 2: down\nInput action: ")
                try:
                    action = int(raw)
                    assert action in (0, 1, 2)
                    return True, action
                except Exception:
                    print("Invalid action!")
        else:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False, None

            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                action = 1
            elif keys[pygame.K_DOWN]:
                action = 2
            else:
                action = 0

            return True, action

    def render(self):
        if self.render_mode == "text":
            court = [["." for _ in range(self.court_width)] for _ in range(self.court_height)]
            for row in range(self.left_paddle_y, min(self.left_paddle_y + self.paddle_height, self.court_height)):
                court[row][0] = "|"
            for row in range(int(self.right_paddle_y), min(int(self.right_paddle_y) + self.paddle_height, self.court_height)):
                court[row][self.court_width - 1] = "|"
            ball_row = int(np.clip(self.ball_y, 0, self.court_height - 1))
            ball_col = int(np.clip(self.ball_x, 0, self.court_width  - 1))
            court[ball_row][ball_col] = "O"
            for row in court:
                print("".join(row))
            print(f"Agent: {self.agent_score}  Opponent: {self.opponent_score}")
        else:
            self.screen.fill(BACKGROUND)

            # Court area (left 70% of window)
            court_pixel_w = int(0.68 * WINDOW_WIDTH)
            court_pixel_h = WINDOW_HEIGHT - 40
            court_top_x   = 10
            court_top_y   = 20

            pygame.draw.rect(
                self.screen, COURT_COLOR,
                (court_top_x, court_top_y, court_pixel_w, court_pixel_h)
            )

            # Net (dashed centre line)
            net_x = court_top_x + court_pixel_w // 2
            dash_height = 12
            for dash_y in range(court_top_y, court_top_y + court_pixel_h, dash_height * 2):
                pygame.draw.rect(
                    self.screen, NET_COLOR,
                    (net_x - 1, dash_y, 2, dash_height)
                )

            cell_w = court_pixel_w / self.court_width
            cell_h = court_pixel_h / self.court_height

            # Left paddle
            left_px = pygame.Rect(
                court_top_x + 4,
                court_top_y + int(self.left_paddle_y * cell_h),
                max(6, int(cell_w * 1.2)),
                int(self.paddle_height * cell_h) - 2,
            )
            pygame.draw.rect(self.screen, PADDLE_COLOR, left_px, border_radius=3)

            # Right paddle
            right_px = pygame.Rect(
                court_top_x + court_pixel_w - max(6, int(cell_w * 1.2)) - 4,
                court_top_y + int(self.right_paddle_y * cell_h),
                max(6, int(cell_w * 1.2)),
                int(self.paddle_height * cell_h) - 2,
            )
            pygame.draw.rect(self.screen, (200, 100, 100), right_px, border_radius=3)

            # Ball
            ball_px_x = court_top_x + int(self.ball_x * cell_w) + int(cell_w / 2)
            ball_px_y = court_top_y + int(self.ball_y * cell_h) + int(cell_h / 2)
            ball_radius = max(4, int(min(cell_w, cell_h) / 2))
            pygame.draw.circle(self.screen, BALL_COLOR, (ball_px_x, ball_px_y), ball_radius)
            pygame.draw.circle(self.screen, (255, 255, 200), (ball_px_x, ball_px_y), ball_radius // 2)

            # Score display in court
            score_font = pygame.font.SysFont("Arial", 36, bold=True)
            agent_text    = score_font.render(str(self.agent_score),    True, SCORE_COLOR)
            opponent_text = score_font.render(str(self.opponent_score), True, (220, 120, 120))
            self.screen.blit(agent_text,    (net_x - 60, court_top_y + 10))
            self.screen.blit(opponent_text, (net_x + 30, court_top_y + 10))

            # Info panel (right 30%)
            panel_surface = pygame.Surface(
                (int(0.26 * WINDOW_WIDTH), WINDOW_HEIGHT - 40), pygame.SRCALPHA
            )
            panel_surface.fill(PANEL_BG)
            self.screen.blit(panel_surface, (int(0.72 * WINDOW_WIDTH), 20))

            avg = sum(self.scores[-100:]) / max(1, min(100, len(self.scores)))
            infos = [
                ("Games Played", len(self.scores)),
                ("Current Score", self.score),
                ("Highest Score", self.highest),
                ("Average Score", f"{avg:.1f}"),
            ]
            info_x = int(0.74 * WINDOW_WIDTH)
            for idx, (label, value) in enumerate(infos):
                label_surf = self.font_title.render(label, True, TEXT_COLOR)
                value_surf = self.font_value.render(str(value), True, SCORE_COLOR)
                self.screen.blit(label_surf, (info_x, 40 + idx * 130))
                self.screen.blit(value_surf, (info_x, 40 + idx * 130 + 36))

            pygame.display.flip()
            self.clock.tick(self.fps)

        return True


# ── Neural network models ─────────────────────────────────────────────────────

class PongDQN(nn.Module):
    """Convolutional Q-network for Pong (DQN)."""

    def __init__(self, args):
        super().__init__()
        n_actions   = 3
        in_channels = args.n_last_frames + 1

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        feature_size = 128 * 4 * 4

        self.feature_compressor = nn.Sequential(
            nn.Linear(feature_size, 256),
            nn.ReLU(inplace=True),
        )

        self.use_dueling_dqn = args.use_dueling_dqn
        if args.use_dueling_dqn:
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
            return values + advantages - advantages.mean(dim=1, keepdim=True)
        return self.fc(features)


class PongPPO(nn.Module):
    """
    Actor-Critic network for Pong (PPO).

    forward(state) -> (logits, value)
    """

    def __init__(self, args):
        super().__init__()
        n_actions   = 3
        in_channels = args.n_last_frames + 1

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        feature_size = 128 * 4 * 4

        self.feature_compressor = nn.Sequential(
            nn.Linear(feature_size, 256),
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


# ── CLI argument extensions ───────────────────────────────────────────────────

def add_custom_argument(parser):
    """Add Pong-specific CLI arguments."""
    parser.add_argument("--width",          type=int,   default=40,
                        help="Court width in grid cells")
    parser.add_argument("--height",         type=int,   default=20,
                        help="Court height in grid cells")
    parser.add_argument("--ball_speed",     type=float, default=0.6,
                        help="Ball speed in grid cells per step")
    parser.add_argument("--opponent_speed", type=float, default=0.5,
                        help="Opponent paddle speed in grid cells per step")
    parser.add_argument("--win_score",      type=int,   default=5,
                        help="Points needed to win an episode")
    parser.set_defaults(speed=30)
    return parser


if __name__ == "__main__":
    run_game(PongGame, {"dqn": PongDQN, "ppo": PongPPO}, add_custom_argument)
