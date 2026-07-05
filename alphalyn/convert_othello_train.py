import zipfile
import random
import argparse
import os
import numpy as np
from board_game import GameArgs
from model import encode_board
from game_othello import OthelloArgs, OthelloGame, MoveAction

def merge():
    lines = []
    for zf in ["data/Egaroucid_Train_Data_v0002_0.zip", "data/Egaroucid_Train_Data_v0002_1.zip"]:
        z = zipfile.ZipFile(zf)
        for f in z.namelist():
            if not f.endswith(".txt"):
                continue
        
            r = int(f.split("/")[-2])
        
            move_history = []
            for idx,line in enumerate(z.read(f).decode("utf-8").strip().split("\n")):
                line = line.strip()
                if line:
                    lines.append([r, line])

    print(len(lines))
    random.shuffle(lines)

    with open("data/Egaroucid_merge.txt", "w", encoding="utf-8") as fo:
        for r, line in lines:
            fo.write(str(r) + " " + line + "\n")

def convert():

    game = OthelloGame(OthelloArgs(argparse.Namespace()))  

    save_idx = 1
    boards = []
    policies = []
    values = []
    
    for idx, line in enumerate(open("data/Egaroucid_merge.txt")):

        game.reset()
        start = 0
        player = game.PLAYER_FIRST
    
        r, line = line.strip().split()
        r = int(r)
        move_history = []

        while start < len(line):
        
            col = ord(line[start]) - ord('a')
            row = ord(line[start + 1] ) - ord('1')
            
            legal_moves = game.get_legal_actions(player)
            if game._is_legal(player, row, col):
                action = MoveAction(row, col, player)
                start += 2
            else:
                #print([row, col], legal_moves)
                assert len(legal_moves) == 1 and legal_moves[0].extra == "pass"
                action = legal_moves[0]
            
            policy = np.zeros(game.num_actions)
            action_idx = game.action_to_index(action)
            if len(legal_moves) > 1:
                policy[action_idx] = 0.9
                for other in legal_moves:
                    other_idx = game.action_to_index(other)
                    if other_idx != action_idx:
                        policy[other_idx] = 0.1 / (len(legal_moves) - 1)
            else:
                policy[action_idx] = 1

            if start // 2 + 1 > r: 
                board_encoding = encode_board(
                    game.state.board,
                    player,
                ).squeeze(0).numpy()  # (num_input_planes, H, W)
                move_history.append((board_encoding, policy, player))

            game.move(player, action)

            if player == game.PLAYER_FIRST:
                player = game.PLAYER_SECOND
            else:
                player = game.PLAYER_FIRST

        winner = game.state.winner
        for board_encoding, policy, player_who_moved in move_history:
            if winner == game.RESULT_TIE:
                value = 0.0
            elif winner == player_who_moved:
                value = 1.0
            else:
                value = -1.0
            boards.append(board_encoding)
            policies.append(policy)
            values.append(value)
            
            if len(boards) % 1000000 == 0:
                print(idx, save_idx*len(boards))
                os.makedirs(f"data/supervised/iter_{save_idx}", exist_ok=True)
                npz_path = f"data/supervised/iter_{save_idx}/worker0.npz"
                np.savez_compressed(
                    npz_path,
                    boards=np.array(boards, dtype=np.int8),
                    policies=np.array(policies, dtype=np.float32),
                    values=np.array(values, dtype=np.int8),
                )
                open(f"data/supervised/iter_{save_idx}/worker0.DONE", "w").close()
                boards = []
                policies = []
                values = []
                save_idx += 1

    if len(boards) > 0:
        npz_path = f"data/supervised/Egaroucid_{save_idx}.npz"
        np.savez_compressed(
            npz_path,
            boards=np.array(boards, dtype=np.int8),
            values=np.array(values, dtype=np.int8),
        )


import sys
convert()


