"""game_2048.py

2048 game environment and neural network models.

Game2048    : sliding-tile puzzle environment (4x4 board).
Game2048DQN : convolutional Q-network (supports embedding input).
Game2048PPO : actor-critic network for PPO training.
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

BACKGROUND_COLOR = (250, 248, 239)
GRID_COLOR = (187, 173, 160)
EMPTY_CELL_COLOR = (205, 193, 180)
SCORE_BG_COLOR = (185, 173, 160)
TEXT_COLOR_DARK = (119, 110, 101)
TEXT_COLOR_LIGHT = (249, 246, 242)
SCORE_TEXT_COLOR = (238, 228, 218)
HIGHLIGHT_COLOR = (250, 250, 255)

TILE_COLORS = {
    2: (238, 228, 218),
    4: (237, 224, 200),
    8: (242, 177, 121),
    16: (245, 149, 99),
    32: (246, 124, 95),
    64: (246, 94, 59),
    128: (237, 207, 114),
    256: (237, 204, 97),
    512: (237, 200, 80),
    1024: (237, 197, 63),
    2048: (237, 194, 46),
    4096: (237, 190, 40),
    8192: (237, 187, 30)
}

TEXT_COLORS = {
    2: TEXT_COLOR_DARK,
    4: TEXT_COLOR_DARK,
    8: TEXT_COLOR_LIGHT,
    16: TEXT_COLOR_LIGHT,
    32: TEXT_COLOR_LIGHT,
    64: TEXT_COLOR_LIGHT,
    128: TEXT_COLOR_LIGHT,
    256: TEXT_COLOR_LIGHT,
    512: TEXT_COLOR_LIGHT,
    1024: TEXT_COLOR_LIGHT,
    2048: TEXT_COLOR_LIGHT,
    4096: TEXT_COLOR_LIGHT,
    8192: TEXT_COLOR_LIGHT
}

BOARD_TOP = 80
TILE_SIZE = 100
TILE_MARGIN = 10
BOARD_SIZE = 4*TILE_SIZE + 5*TILE_MARGIN

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600

class Game2048(BaseGame):
    """
    2048 sliding-tile puzzle environment.

    Inherits state allocation and pygame helpers from BaseGame.
    All game logic (merging, scoring, reward shaping) is unchanged.
    """
    def __init__(self, args):
        """
        """
        super().__init__()
        self.args = args
        self.n_last_frames = args.n_last_frames
        self.use_emb = args.use_emb
        self.normalize_reward = args.normalize_reward
        if self.use_emb:
            self.state_info = [                                                                             
                {                                                                                           
                    "shape": [args.n_last_frames+1, 4, 4],
                    "dtype": "int"                                                                        
                }                                                                                           
            ]
        else:
            self.state_info = [                                                                             
                {              
                    "shape": [16*args.n_last_frames, 4, 4],
                    "dtype": "float"
                }                                                                                       
            ]

        self.n_actions = 4

        self.up = 0
        self.right = 1
        self.down = 2
        self.left = 3
        self.act2str = {
            self.up: "up", 
            self.right: "right",
            self.left: "left",
            self.down: "down"
        }

        self.board = np.zeros((4,4), dtype=int)
        self.score = 0
        self.done = False
        self.reset()
        self.cnt = 0

        self.highest = 0
        self.highest_tile = 0
        self.success_cnt = 0
        self.scores = []
 
        self.render_mode = args.render_mode
        if args.render_mode == "gui" and not GUI_RENDER_AVAILABLE :
            logger.warn("pygame not install, use text render mode.")
            args.render_mode = "text"
        
        if args.render_mode == "gui":
            self.fps = args.fps
            self.clock = pygame.time.Clock()
            
            self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))

            if not pygame.get_init():
                pygame.init()

            pygame.display.set_caption('MystRL 2048')
            
            self.font_tile_large = pygame.font.SysFont('Arial', 50, bold=True)
            self.font_tile_small = pygame.font.SysFont('Arial', 36, bold=True)
            self.font_title = pygame.font.SysFont('Arial', 48, bold=True)
            self.font_score_title = pygame.font.SysFont('Arial', 24)
            self.font_score_value = pygame.font.SysFont('Arial', 32, bold=True)

    def create_new_state(self):
        """
        """  
        state = []
        for info in self.state_info:
            dtype = np.int32 if info['dtype'] == 'int' else np.float32
            state.append(np.zeros(
                info['shape'], dtype=dtype
            ))
        return state

           
    def update_state(self):
        """
        """
        new_state = self.create_new_state()
        if self.use_emb:
            new_state[0][:(self.n_last_frames-1),:,:] = self.state[0][-(self.n_last_frames-1):,:,:]
        else:
            new_state[0][:16*(self.n_last_frames-1),:,:] = self.state[0][-16*(self.n_last_frames-1):,:,:]

        for i in range(4):
            for j in range(4):
                if self.board[i][j] == 0:
                    x = 0
                else:
                    x = int(np.log2(self.board[i][j]))
                if self.use_emb:
                    new_state[0][-1,i,j] = x 
                else:
                    new_state[0][-x,i,j] = 1 
        self.state = new_state
        return self.state        


    def reset(self):
        """
        """
        self.board = np.zeros((4, 4), dtype=int)
        self.score = 0
        self.done = False
        self.add_random_tile()
        self.add_random_tile()  
        self.state = self.create_new_state()
        self.update_state()
        self.valid_cnt = 0
        self.invalid_cnt = 0
        return self.state
 

    def get_valid_actions(self):
        """
        Return actions that would actually change the board.

        An action is valid if applying it produces a board different from
        the current one (i.e. at least one tile moves or merges).

        Returns:
            list[int]: Valid action indices.
        """
        valid = []
        rotation_map = {
            self.up:    1,
            self.right: 2,
            self.down:  3,
            self.left:  0,
        }
        for action in range(self.n_actions):
            rotated = np.rot90(self.board, k=rotation_map[action])
            # A left-slide changes the board iff any row has:
            #   (a) a non-zero tile with a zero to its left (tile can slide), or
            #   (b) two adjacent equal non-zero tiles (can merge).
            changed = False
            for row in rotated:
                non_zero_mask = row != 0
                non_zero = row[non_zero_mask]
                # Condition (a): any zero appears before the last non-zero tile
                if non_zero_mask.any() and not non_zero_mask.all():
                    last_nonzero_idx = np.where(non_zero_mask)[0][-1]
                    if (row[:last_nonzero_idx] == 0).any():
                        changed = True
                        break
                # Condition (b): adjacent equal non-zero tiles
                if len(non_zero) >= 2 and np.any(non_zero[:-1] == non_zero[1:]):
                    changed = True
                    break
            if changed:
                valid.append(action)
        return valid if valid else list(range(self.n_actions))

    def no_available_move(self):
        """
        """
        if not self.board.all():
            return False
        for i in range(4): 
            for j in range(4):
                if i < 3 and self.board[i, j] == self.board[i+1, j]:
                    return False
                if j < 3 and self.board[i, j] == self.board[i, j+1]:
                    return False
        return True
   
 
    def step(self, action):
        """
        """
        if action is None:
            return self.state, 0, False
        
        prev_board = self.board.copy()

        if action == self.up:
            self.board = np.rot90(self.board, k=1)
        elif action == self.right:        
            self.board = np.rot90(self.board, k=2)
        elif action == self.down:
            self.board = np.rot90(self.board, k=3)
        elif action == self.left:
            self.board = np.rot90(self.board, k=0)

        reward = self.move_left()

        if action == self.up:
            self.board = np.rot90(self.board, k=-1)
        elif action == self.right:
            self.board = np.rot90(self.board, k=-2)
        elif action == self.down:
            self.board = np.rot90(self.board, k=-3)
        elif action == self.left:
            self.board = np.rot90(self.board, k=0)

        dead_reward = 0
        invalid_reward = -1000

        done = False
        if not np.array_equal(prev_board, self.board):
            self.add_random_tile()
            if self.no_available_move():   
                reward = reward - dead_reward
                done = True
                self.cnt += 1 
            self.valid_cnt += 1   
        else:
            self.invalid_cnt += 1
            reward = invalid_reward

        if self.normalize_reward:
            reward = float(np.clip(reward / 100.0, -1.0, 1.0))

        self.update_state()  
 
        self.highest = max(self.highest, self.score)
        self.highest_tile = max(self.highest_tile, int(self.board.max()))
        if done:
            self.scores.append(self.score)
            avg = sum(self.scores[-100:]) / min(100, len(self.scores))
            max_num = int(self.board.max())
            if max_num >= 2048:
                self.success_cnt += 1
            if self.cnt % self.args.print_every_game == 0:
                success_rate = self.success_cnt / self.cnt if self.cnt > 0 else 0.0
                logger.info(f"[{self.name}] {self.cnt} game done, score: {self.score}, avg: {avg}, highest:{self.highest}, max_tile:{max_num}, best_tile:{self.highest_tile}, success_rate:{success_rate:.2%}, valid:{self.valid_cnt}, invalid:{self.invalid_cnt}")

        return self.state, reward, done


    def add_random_tile(self):
        """
        """
        empty_cells = [(i, j) for i in range(4) 
                      for j in range(4) if self.board[i, j] == 0]
        
        if empty_cells:
            i, j = random.choice(empty_cells)
            self.board[i, j] = 2 if random.random() < 0.9 else 4
            #self.score += self.board[i, j]

    
    def move_left(self):
        """
        """
        alpha = 0.1
        beta = 0.2

        reward = 0
        for i in range(4):
            j = 0 
            row = [x for x in self.board[i] if x != 0]
            while j < len(row):
                if j < len(row) - 1 and row[j] == row[j+1]:
                    row[j] = 2 * row[j]
                    reward = reward + np.log2(row[j]) * alpha
                    self.score += row[j]
                    row.pop(j+1)
                j = j + 1
            self.board[i, :len(row)] = row            
            self.board[i, len(row):] = 0
 
        reward = reward + (16 - np.count_nonzero(self.board)) * beta

        return reward

    def process_input(self):
        """
        """
        if self.render_mode == "text":
            while True:
                action = input("0: up, 1: right, 2: down, 3:left\nInput action:")
                try:
                    action = int(action)
                    assert action in [0, 1, 2, 3]
                    return True, action
                except:
                    print("invalid action!")
        else:                                                                                     
            running, action = True, None
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running, action = False,None
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        running, action = True,self.up
                    elif event.key == pygame.K_DOWN:
                        running, action = True,self.down
                    elif event.key == pygame.K_LEFT:
                        running, action = True,self.left
                    elif event.key == pygame.K_RIGHT:
                        running, action = True,self.right
            
            return running, action

    def render(self):
        """
        """
        if self.render_mode == "text":
            print(self.board)
            print(f"score:{self.score}")
        else:
            clock = pygame.time.Clock()
            fps = self.fps
            left_margin = 50
            
            self.screen.fill(BACKGROUND_COLOR)
            
            pygame.draw.rect(self.screen, GRID_COLOR, 
                             (left_margin, BOARD_TOP, 
                              BOARD_SIZE, BOARD_SIZE),
                             border_radius=8)
    
            for row in range(4):
                for col in range(4):
                    value = self.board[row][col]
                    tile_x = left_margin + TILE_MARGIN + col*(TILE_SIZE + TILE_MARGIN)
                    tile_y = BOARD_TOP + TILE_MARGIN + row*(TILE_SIZE + TILE_MARGIN)
            
                    if value == 0:  
                        pygame.draw.rect(self.screen, EMPTY_CELL_COLOR, 
                                         (tile_x, tile_y, TILE_SIZE, TILE_SIZE),
                                         border_radius=5)
                    else: 
                        tile_color = TILE_COLORS.get(value, TILE_COLORS[8192])
                        text_color = TEXT_COLORS.get(value, TEXT_COLORS[8192])
                        
                        font_tile = self.font_tile_large if value < 1000 else self.font_tile_small
                
                        pygame.draw.rect(self.screen, tile_color, 
                                         (tile_x, tile_y, TILE_SIZE, TILE_SIZE),
                                         border_radius=5)
                
                        text = font_tile.render(str(value), True, text_color)
                        text_rect = text.get_rect(center=(tile_x+TILE_SIZE//2, tile_y+TILE_SIZE//2))
                        self.screen.blit(text, text_rect)
    
            avg = 0
            if len(self.scores) > 0:
                avg = sum(self.scores) // len(self.scores)
            success_rate = self.success_cnt / self.cnt if self.cnt > 0 else 0.0
            infos = [
                ("Games Played", len(self.scores)),
                ("Current Score", self.score),
                ("Highest Score", self.highest),
                ("Average Score", avg),
                ("Best Tile", self.highest_tile),
                ("2048 Rate", f"{success_rate:.1%}"),
            ]
            
            info_x_start = left_margin + BOARD_SIZE + 50
            right_panel_width = 200
            info_y_start = 35
            
            for i, (label, value) in enumerate(infos):
                y = info_y_start + i * 90
                
                pygame.draw.rect(self.screen, SCORE_BG_COLOR, 
                                 (info_x_start, y - 15, right_panel_width, 120),
                                 border_radius=8)
                
                label_text = self.font_score_title.render(label, True, SCORE_TEXT_COLOR)
                self.screen.blit(label_text, (info_x_start + 20, y))
                
                value_text = self.font_score_value.render(str(value), True, HIGHLIGHT_COLOR)
                self.screen.blit(value_text, (info_x_start + 20, y + 40))
    
            pygame.display.flip()
            
            clock.tick(fps)


class ConvBolck(nn.Module):
   def __init__(self, in_channels, out_channels):
       super(ConvBolck, self).__init__()
       self.conv_1_2 = nn.Conv2d(in_channels, out_channels, (1, 2), padding='same')
       self.conv_2_1 = nn.Conv2d(in_channels, out_channels, (2, 1), padding='same')
       self.conv_2 = nn.Conv2d(in_channels, out_channels, 2, padding='same')
       self.conv_3 = nn.Conv2d(in_channels, out_channels, 3, padding='same')

   def forward(self, x): 
       output_1_2 = self.conv_1_2(x)
       output_2_1 = self.conv_2_1(x) 
       output2 = self.conv_2(x)
       output3 = self.conv_3(x)
       return torch.cat((output_1_2, output_2_1, output2, output3), dim=1)


class Game2048DQN(nn.Module):
   def __init__(self, args):
       super(Game2048DQN, self).__init__()
       n_actions, in_channels = 4, 16*args.n_last_frames
      
       self.use_emb = args.use_emb 
       if self.use_emb:
           self.embed_size = args.embed_size
           self.embeding = nn.Embedding(16, self.embed_size)
           in_channels = (args.n_last_frames + 1)*self.embed_size
 
       self.conv_block_1 = ConvBolck(in_channels, 64)
       self.conv_block_2 = ConvBolck(64*4, 128)  

       self.feature_size = 64 * 4 * 4 * 4 + 128 * 4 * 4 * 4

       self.feature_compressor = nn.Sequential(                                                        
           nn.Linear(self.feature_size, 256),                                                          
           nn.ReLU(inplace=True),
       )                                                                                               
      
       self.use_dueling_dqn = args.use_dueling_dqn                                                     
       if args.use_dueling_dqn: 
           self.value_stream = nn.Linear(256, 1)                                                           
           self.advantage_stream = nn.Linear(256, n_actions)    
       else:
           self.fc = nn.Linear(256, 4)       

   def forward(self, state):
       x = state[0]
       if self.use_emb:
           x = self.embeding(x.long())
           x = torch.permute(x, [0,1,4,2,3])
           x = torch.cat(torch.unbind(x, 1), 1)
 
       conv_out_1 = F.relu(self.conv_block_1(x))
       conv_out_2 = F.relu(self.conv_block_2(conv_out_1))
       conv_out_concat = torch.cat((conv_out_1, conv_out_2), dim=1) 
       flat = nn.Flatten()(conv_out_concat)
                                                                                                       
       features = self.feature_compressor(flat)
                                                    
       if self.use_dueling_dqn:                                                    
           values = self.value_stream(features) 
           advantages = self.advantage_stream(features)
                                                                                                           
           return values + advantages - advantages.mean(1, keepdim=True) 
       return self.fc(features)


def add_custom_argument(parser):
    """
    Adds custom command-line arguments to an argument parser.

    This function extends an existing ArgumentParser object with game configuration
    and model selection options. 
 
    Parameters:
    parser (argparse.ArgumentParser): The argument parser to enhance

    Returns:
    argparse.ArgumentParser: The modified parser with added arguments
    """
    parser.add_argument("--use_emb",
                        default=False,
                        action='store_true',
                        help="if true, use embeded")

    parser.add_argument("--embed_size",
                        type=int,
                        default=64,
                        help="convert to embeded size")

    parser.add_argument("--normalize_reward",
                        default=False,
                        action="store_true",
                        help="clip reward to (-1, 1); valid rewards divided by 5, invalid -1000 clipped to -1")

    return parser
 

class Game2048PPO(nn.Module):
    """
    Actor-Critic network for 2048 (PPO).

    Shares the convolutional feature extractor with Game2048DQN, then
    splits into separate actor (policy) and critic (value) heads.
    forward(state) -> (logits, value)  as required by ppo.py.
    """

    def __init__(self, args):
        super(Game2048PPO, self).__init__()
        n_actions, in_channels = 4, 16 * args.n_last_frames

        self.use_emb = args.use_emb
        if self.use_emb:
            self.embed_size = args.embed_size
            self.embeding   = nn.Embedding(16, self.embed_size)
            in_channels     = (args.n_last_frames + 1) * self.embed_size

        self.conv_block_1 = ConvBolck(in_channels, 64)
        self.conv_block_2 = ConvBolck(64 * 4, 128)

        feature_size = 64 * 4 * 4 * 4 + 128 * 4 * 4 * 4

        self.feature_compressor = nn.Sequential(
            nn.Linear(feature_size, 256),
            nn.ReLU(inplace=True),
        )

        self.actor  = nn.Linear(256, n_actions)
        self.critic = nn.Linear(256, 1)

    def forward(self, state):
        x = state[0]
        if self.use_emb:
            x = self.embeding(x.long())
            x = torch.permute(x, [0, 1, 4, 2, 3])
            x = torch.cat(torch.unbind(x, 1), 1)

        conv_out_1      = F.relu(self.conv_block_1(x))
        conv_out_2      = F.relu(self.conv_block_2(conv_out_1))
        conv_out_concat = torch.cat((conv_out_1, conv_out_2), dim=1)
        flat            = nn.Flatten()(conv_out_concat)
        features        = self.feature_compressor(flat)

        return self.actor(features), self.critic(features)


if __name__ == "__main__":
    run_game(Game2048, {"dqn": Game2048DQN, "ppo": Game2048PPO}, add_custom_argument)
