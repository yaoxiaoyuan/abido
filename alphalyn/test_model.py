import numpy as np
import torch

from model import ResNetConfig, PolicyValueNet

net_config = ResNetConfig(
        board_height=8,
        board_width=8,
        num_input_planes=3,
        num_actions=65,
        num_filters=256,
        num_residual_blocks=15,
        value_head_hidden_size=256,
    )
net = PolicyValueNet(net_config)

net.load_state_dict(torch.load("model/othello_v2_s4/checkpoint.pt", map_location="cpu"))

data = np.load("data/Egaroucid_0_0.npz")
boards = data["boards"]
values = data["values"]

for board,value in zip(boards, values):

    pred = net.predict(torch.from_numpy(board).unsqueeze(0))

    print(pred, value)

    input()