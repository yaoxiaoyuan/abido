"""ResNet-based policy-value network for AlphaZero-style board game agents.

Architecture
------------
- Input  : (batch, channels, height, width)
            channels = num_input_planes (e.g. 2 for current/opponent stones + 1 for turn)
- Backbone: stack of ResidualBlock layers (configurable depth & width)
- Policy head : conv → flatten → linear → softmax over (height * width + 1) actions
                the extra "+1" slot represents the pass action
- Value head  : conv → flatten → linear → tanh → scalar in [-1, 1]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ResNetConfig:
    """Hyper-parameters for the ResNet policy-value network.

    Attributes
    ----------
    board_height:
        Number of rows on the board.
    board_width:
        Number of columns on the board.
    num_input_planes:
        Number of input feature planes (channels).
        Typical value: 2 (current player stones, opponent stones) + 1 (turn indicator).
    num_filters:
        Number of convolutional filters in the backbone (residual width).
    num_residual_blocks:
        Depth of the ResNet backbone.
    policy_head_filters:
        Filters used in the policy head's 1×1 conv.
    value_head_filters:
        Filters used in the value head's 1×1 conv.
    value_head_hidden_size:
        Hidden units in the value head's fully-connected layer.
    """

    board_height: int = 8
    board_width: int = 8
    num_input_planes: int = 3
    num_filters: int = 128
    num_residual_blocks: int = 10
    policy_head_filters: int = 2
    value_head_filters: int = 1
    value_head_hidden_size: int = 256
    num_actions: Optional[int] = None  # if None, defaults to board_height * board_width + 1


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ConvBnRelu(nn.Module):
    """Conv2d → BatchNorm2d → ReLU."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)), inplace=True)


class ResidualBlock(nn.Module):
    """Standard pre-activation residual block (two 3×3 convolutions).

    The skip connection is a plain identity mapping; both convolutions share
    the same number of filters so no projection is needed.
    """

    def __init__(self, num_filters: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual, inplace=True)


# ---------------------------------------------------------------------------
# Full network
# ---------------------------------------------------------------------------

class PolicyValueNet(nn.Module):
    """ResNet backbone with a policy head and a value head.

    Parameters
    ----------
    config:
        ``ResNetConfig`` instance controlling all architectural choices.

    Outputs (forward pass)
    ----------------------
    log_policy : torch.Tensor, shape (batch, num_actions)
        Log-softmax over all legal action slots.
        ``num_actions`` is taken from ``config.num_actions`` when provided;
        otherwise it defaults to ``board_height * board_width + 1``.
    value : torch.Tensor, shape (batch,)
        Scalar game value estimate in ``[-1, 1]`` from the current player's
        perspective.  +1 means the current player is winning.
    """

    def __init__(self, config: ResNetConfig) -> None:
        super().__init__()
        self.config = config
        board_area = config.board_height * config.board_width
        num_actions = config.num_actions if config.num_actions is not None else board_area + 1

        # --- Backbone ---
        self.input_conv = ConvBnRelu(config.num_input_planes, config.num_filters)
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(config.num_filters) for _ in range(config.num_residual_blocks)]
        )

        # --- Policy head ---
        self.policy_conv = nn.Conv2d(config.num_filters, config.policy_head_filters, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(config.policy_head_filters)
        self.policy_fc = nn.Linear(config.policy_head_filters * board_area, num_actions)

        # --- Value head ---
        self.value_conv = nn.Conv2d(config.num_filters, config.value_head_filters, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(config.value_head_filters)
        self.value_fc1 = nn.Linear(config.value_head_filters * board_area, config.value_head_hidden_size)
        self.value_fc2 = nn.Linear(config.value_head_hidden_size, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        x:
            Input tensor of shape ``(batch, num_input_planes, height, width)``.

        Returns
        -------
        log_policy:
            Log-probabilities over actions, shape ``(batch, num_actions)``.
        value:
            Scalar value estimate per sample, shape ``(batch,)``.
        """
        # Backbone
        features = self.input_conv(x)
        features = self.residual_blocks(features)

        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(features)), inplace=True)
        policy = policy.flatten(start_dim=1)
        log_policy = F.log_softmax(self.policy_fc(policy), dim=1)

        # Value head
        value = F.relu(self.value_bn(self.value_conv(features)), inplace=True)
        value = value.flatten(start_dim=1)
        value = F.relu(self.value_fc1(value), inplace=True)
        value = torch.tanh(self.value_fc2(value)).squeeze(1)

        return log_policy, value

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run inference without gradient tracking.

        Returns probabilities (not log-probabilities) and value.
        """
        self.eval()
        with torch.no_grad():
            log_policy, value = self.forward(x)
        return log_policy.exp(), value

    @property
    def num_actions(self) -> int:
        """Total number of actions (board cells + 1 pass)."""
        return self.config.board_height * self.config.board_width + 1


# ---------------------------------------------------------------------------
# Board encoding utility
# ---------------------------------------------------------------------------

def encode_board(
    board: "np.ndarray",  # type: ignore[name-defined]
    current_player: int,
    board_player_first_value: int = 1,
    board_player_second_value: int = 2,
) -> torch.Tensor:
    """Encode a raw board array into a 3-plane input tensor.

    Planes
    ------
    0 : cells occupied by the current player (1.0 where True)
    1 : cells occupied by the opponent (1.0 where True)
    2 : constant plane indicating whose turn it is
        (1.0 = PLAYER_FIRST's turn, 0.0 = PLAYER_SECOND's turn)

    Parameters
    ----------
    board:
        ``np.ndarray`` of shape ``(height, width)`` with integer cell values.
    current_player:
        ``BoardGame.PLAYER_FIRST`` (0) or ``BoardGame.PLAYER_SECOND`` (1).
    board_player_first_value:
        Board cell value that represents PLAYER_FIRST's stone (default 1).
    board_player_second_value:
        Board cell value that represents PLAYER_SECOND's stone (default 2).

    Returns
    -------
    torch.Tensor
        Shape ``(1, 3, height, width)``, ready to be fed into ``PolicyValueNet``.
    """
    import numpy as np

    if current_player == 0:  # PLAYER_FIRST
        current_value = board_player_first_value
        opponent_value = board_player_second_value
        turn_plane_fill = 1.0
    else:  # PLAYER_SECOND
        current_value = board_player_second_value
        opponent_value = board_player_first_value
        turn_plane_fill = 0.0

    height, width = board.shape
    planes = np.zeros((3, height, width), dtype=np.float32)
    planes[0] = (board == current_value).astype(np.float32)
    planes[1] = (board == opponent_value).astype(np.float32)
    planes[2] = turn_plane_fill

    return torch.from_numpy(planes).unsqueeze(0)  # (1, 3, H, W)


if __name__ == "__main__":

    net = PolicyValueNet(ResNetConfig())
    print(sum(p.numel() for p in net.parameters()))
