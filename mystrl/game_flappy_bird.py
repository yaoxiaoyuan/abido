"""game_flappy_bird.py

Flappy Bird game environment and neural network models.

FlappyBirdGame : side-scrolling obstacle-avoidance environment.
                 The agent controls a bird that falls under gravity and can
                 flap to gain upward velocity.  Pipes scroll from right to
                 left; the agent scores one point per pipe pair cleared.
FlappyBirdDQN  : convolutional Q-network for DQN training.
FlappyBirdPPO  : actor-critic network for PPO training.

State representation:
  A stack of (n_last_frames + 1) grayscale frames, each of shape
  (height, width), normalised to [0, 1].

Actions:
  0 — do nothing (fall under gravity)
  1 — flap (apply upward impulse)
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
BACKGROUND_TOP    = (30,  90, 160)
BACKGROUND_BOTTOM = (80, 160,  80)
PIPE_COLOR        = (60, 180,  60)
PIPE_DARK         = (40, 130,  40)
BIRD_COLOR        = (240, 210,  50)
BIRD_EYE          = (20,  20,  20)
BIRD_WING         = (200, 160,  30)
GROUND_COLOR      = (180, 140,  80)
TEXT_COLOR        = (240, 240, 255)
SCORE_COLOR       = (255, 255, 255)
PANEL_BG          = (20,  20,  40, 210)

WINDOW_WIDTH  = 800
WINDOW_HEIGHT = 600


class FlappyBirdGame(BaseGame):
    """
    Flappy Bird environment.

    The bird occupies a single cell on a (height × width) grid.
    Pipe pairs scroll leftward one cell per step.  A new pipe pair is
    spawned every ``pipe_interval`` steps.  The gap between the top and
    bottom pipe is ``gap_size`` cells tall.

    Episode ends when the bird hits a pipe, the ground, or the ceiling.
    Reward: +1 for each pipe pair cleared, -1 on death.
    """

    def __init__(self, args):
        super().__init__()
        self.args = args

        self.grid_width  = args.width
        self.grid_height = args.height

        self.state_info = [
            {
                "shape": [args.n_last_frames + 1, self.grid_height, self.grid_width],
                "dtype": "float",
            }
        ]

        self.n_actions = 2
        self.act2str   = {0: "nothing", 1: "flap"}

        self.gravity      = args.gravity
        self.flap_impulse = args.flap_impulse
        self.pipe_speed   = args.pipe_speed
        self.pipe_interval = args.pipe_interval
        self.gap_size     = args.gap_size
        self.pipe_width   = args.pipe_width

        self.highest = 0
        self.scores  = []
        self.cnt     = 0
        self.score   = 0

        self.render_mode = args.render_mode
        self._check_gui(args)

        if args.render_mode == "gui":
            self.fps    = args.fps
            self.clock  = pygame.time.Clock()
            self.screen = self._init_pygame(
                "MystRL Flappy Bird", WINDOW_WIDTH, WINDOW_HEIGHT
            )
            self.font_title = pygame.font.SysFont("Arial", 22)
            self.font_value = pygame.font.SysFont("Arial", 30, bold=True)
            self.font_big   = pygame.font.SysFont("Arial", 48, bold=True)

        self.reset()

    # ── Internal helpers ──────────────────────────────────────────────────

    def _spawn_pipe(self, col):
        """Return a (top_height, gap_start) tuple for a new pipe at column col."""
        max_gap_start = self.grid_height - self.gap_size - 1
        gap_start = random.randint(1, max(1, max_gap_start))
        return {"col": col, "gap_start": gap_start}

    def _render_frame_to_array(self):
        """Encode the current state as a normalised 2-D float array."""
        frame = np.zeros((self.grid_height, self.grid_width), dtype=np.float32)

        # Pipes
        for pipe in self.pipes:
            col_start = int(pipe["col"])
            gap_start = pipe["gap_start"]
            gap_end   = gap_start + self.gap_size
            for col in range(col_start, col_start + self.pipe_width):
                if 0 <= col < self.grid_width:
                    for row in range(self.grid_height):
                        if row < gap_start or row >= gap_end:
                            frame[row, col] = 0.6

        # Bird
        bird_row = int(np.clip(self.bird_y, 0, self.grid_height - 1))
        frame[bird_row, self.bird_x] = 1.0

        return frame

    def _check_collision(self):
        """Return True if the bird has hit a pipe, the ground, or the ceiling."""
        bird_row = int(self.bird_y)

        # Ceiling / ground
        if bird_row < 0 or bird_row >= self.grid_height:
            return True

        # Pipes
        for pipe in self.pipes:
            col_start = int(pipe["col"])
            col_end   = col_start + self.pipe_width
            if col_start <= self.bird_x < col_end:
                gap_start = pipe["gap_start"]
                gap_end   = gap_start + self.gap_size
                if not (gap_start <= bird_row < gap_end):
                    return True

        return False

    # ── BaseGame interface ────────────────────────────────────────────────

    def reset(self):
        self.bird_x   = self.grid_width // 4
        self.bird_y   = float(self.grid_height // 2)
        self.bird_vel = 0.0
        self.score    = 0
        self.steps    = 0

        self._pending_flap = False

        # Pre-populate pipes so the screen is never empty from the start.
        # The first pipe starts at bird_x + first_pipe_distance; subsequent
        # pipes are spaced pipe_interval apart toward the right edge.
        self.pipes = []
        col = self.bird_x + self.args.first_pipe_distance
        while col < self.grid_width + self.pipe_width:
            self.pipes.append(self._spawn_pipe(col))
            col += self.pipe_interval

        self.state = self.create_new_state()
        self.update_state()
        return self.state

    def update_state(self):
        new_state = self.create_new_state()
        new_state[0][:-1] = self.state[0][1:]
        new_state[0][-1]  = self._render_frame_to_array()
        self.state = new_state
        return self.state

    def step(self, action):
        self.steps += 1

        # Apply flap or gravity
        if action == 1:
            self.bird_vel = -self.flap_impulse
        else:
            self.bird_vel += self.gravity

        self.bird_y += self.bird_vel

        # Scroll pipes leftward
        for pipe in self.pipes:
            pipe["col"] -= self.pipe_speed

        # Remove pipes that have scrolled off screen
        self.pipes = [p for p in self.pipes if p["col"] > -1]

        # Spawn new pipe when the rightmost pipe has scrolled far enough left
        rightmost_col = max((p["col"] for p in self.pipes), default=-1)
        if rightmost_col <= self.grid_width - self.pipe_interval:
            self.pipes.append(self._spawn_pipe(self.grid_width + 2))

        # Check scoring: bird passed a pipe (pipe's trailing edge just crossed bird_x)
        reward = 0.0
        for pipe in self.pipes:
            pipe_trailing_prev = pipe["col"] + self.pipe_width + self.pipe_speed
            pipe_trailing_curr = pipe["col"] + self.pipe_width
            if pipe_trailing_prev >= self.bird_x and pipe_trailing_curr < self.bird_x:
                self.score += 1
                reward = 0.01
                break

        # Check collision
        done = self._check_collision()
        if done:
            reward = -1.0
        else:
            reward += 0.001
            
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
            raw = input("0: nothing, 1: flap\nInput action: ")
            try:
                action = int(raw)
                assert action in (0, 1)
                return True, action
            except Exception:
                return True, 0
        else:
            # Flappy Bird needs edge-triggered flap (KEYDOWN), not level-triggered.
            # We latch the flap in self._pending_flap so it is not lost between
            # the input poll and the next env.step() call (which may be several
            # render frames later when fps > game speed).
            action = 0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False, None
                if event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_SPACE, pygame.K_UP):
                        self._pending_flap = True
            if self._pending_flap:
                action = 1
                self._pending_flap = False
            return True, action

    def render(self):
        if self.render_mode == "text":
            bird_row = int(np.clip(self.bird_y, 0, self.grid_height - 1))
            for row in range(self.grid_height):
                line = []
                for col in range(self.grid_width):
                    is_pipe = False
                    for pipe in self.pipes:
                        if int(pipe["col"]) == col:
                            gap_start = pipe["gap_start"]
                            gap_end   = gap_start + self.gap_size
                            if not (gap_start <= row < gap_end):
                                is_pipe = True
                    if row == bird_row and col == self.bird_x:
                        line.append("B")
                    elif is_pipe:
                        line.append("|")
                    else:
                        line.append(".")
                print("".join(line))
            print(f"score: {self.score}  vel: {self.bird_vel:.2f}")
        else:
            # ── Background gradient ───────────────────────────────────────
            sky_rect    = pygame.Rect(0, 0, WINDOW_WIDTH, int(WINDOW_HEIGHT * 0.85))
            ground_rect = pygame.Rect(0, int(WINDOW_HEIGHT * 0.85), WINDOW_WIDTH, int(WINDOW_HEIGHT * 0.15))
            self.screen.fill(BACKGROUND_TOP, sky_rect)
            self.screen.fill(BACKGROUND_BOTTOM, ground_rect)

            # Game area occupies left 70% of window
            game_w = int(0.68 * WINDOW_WIDTH)
            game_h = int(WINDOW_HEIGHT * 0.85)
            cell_w = game_w / self.grid_width
            cell_h = game_h / self.grid_height

            # ── Pipes ─────────────────────────────────────────────────────
            for pipe in self.pipes:
                col = pipe["col"]
                if col + self.pipe_width < 0 or col >= self.grid_width:
                    continue
                gap_start = pipe["gap_start"]
                gap_end   = gap_start + self.gap_size
                pipe_px_x = int(col * cell_w)
                pipe_px_w = max(8, int(cell_w * self.pipe_width))

                # Top pipe
                top_h = int(gap_start * cell_h)
                if top_h > 0:
                    pygame.draw.rect(self.screen, PIPE_COLOR,
                                     (pipe_px_x, 0, pipe_px_w, top_h))
                    pygame.draw.rect(self.screen, PIPE_DARK,
                                     (pipe_px_x, top_h - 6, pipe_px_w + 4, 6))

                # Bottom pipe
                bottom_y = int(gap_end * cell_h)
                bottom_h = game_h - bottom_y
                if bottom_h > 0:
                    pygame.draw.rect(self.screen, PIPE_COLOR,
                                     (pipe_px_x, bottom_y, pipe_px_w, bottom_h))
                    pygame.draw.rect(self.screen, PIPE_DARK,
                                     (pipe_px_x - 2, bottom_y, pipe_px_w + 4, 6))

            # ── Bird ──────────────────────────────────────────────────────
            bird_row   = float(np.clip(self.bird_y, 0, self.grid_height - 1))
            bird_px_x  = int(self.bird_x * cell_w + cell_w / 2)
            bird_px_y  = int(bird_row    * cell_h + cell_h / 2)
            bird_radius = max(8, int(min(cell_w, cell_h) * 0.7))

            pygame.draw.circle(self.screen, BIRD_COLOR, (bird_px_x, bird_px_y), bird_radius)
            # Eye
            pygame.draw.circle(self.screen, BIRD_EYE,
                                (bird_px_x + bird_radius // 2, bird_px_y - bird_radius // 3),
                                max(2, bird_radius // 4))
            # Wing
            wing_pts = [
                (bird_px_x - bird_radius // 2, bird_px_y),
                (bird_px_x - bird_radius,      bird_px_y + bird_radius // 2),
                (bird_px_x,                    bird_px_y + bird_radius // 3),
            ]
            pygame.draw.polygon(self.screen, BIRD_WING, wing_pts)

            # ── Score overlay ─────────────────────────────────────────────
            score_surf = self.font_big.render(str(self.score), True, SCORE_COLOR)
            self.screen.blit(score_surf, (game_w // 2 - score_surf.get_width() // 2, 10))

            # ── Info panel ────────────────────────────────────────────────
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

class FlappyBirdDQN(nn.Module):
    """Convolutional Q-network for Flappy Bird (DQN)."""

    def __init__(self, args):
        super().__init__()
        n_actions   = 2
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


class FlappyBirdPPO(nn.Module):
    """
    Actor-Critic network for Flappy Bird (PPO).

    forward(state) -> (logits, value)
    """

    def __init__(self, args):
        super().__init__()
        n_actions   = 2
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
    """Add Flappy Bird-specific CLI arguments."""
    parser.add_argument("--width",         type=int,   default=40,
                        help="Grid width in cells")
    parser.add_argument("--height",        type=int,   default=20,
                        help="Grid height in cells")
    parser.add_argument("--gravity",       type=float, default=0.05,
                        help="Downward acceleration per step")
    parser.add_argument("--flap_impulse",  type=float, default=0.5,
                        help="Upward velocity applied on flap")
    parser.add_argument("--pipe_speed",    type=float, default=0.5,
                        help="Pipe scroll speed in cells per step")
    parser.add_argument("--pipe_interval", type=int,   default=20,
                        help="Steps between consecutive pipe spawns")
    parser.add_argument("--gap_size",      type=int,   default=6,
                        help="Vertical gap between top and bottom pipe (cells)")
    parser.add_argument("--pipe_width",    type=int,   default=2,
                        help="Horizontal width of each pipe in grid cells")
    parser.add_argument("--first_pipe_distance", type=int, default=30,
                        help="Distance in grid cells from the bird to the first pipe at episode start")
    # Flappy Bird steps every frame for smooth physics; override base default of 8
    parser.set_defaults(speed=30)
    return parser


if __name__ == "__main__":
    run_game(FlappyBirdGame, {"dqn": FlappyBirdDQN, "ppo": FlappyBirdPPO}, add_custom_argument)
