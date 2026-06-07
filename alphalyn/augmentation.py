"""augmentation.py – Board-game-aware data augmentation for AlphaZero training.

Supports different symmetry groups depending on the game:
- Square-board games (othello, gomoku, tictactoe): full D4 group (8 symmetries).
- Connect4: horizontal flip only (non-square board, column-based policy).
- Others (hex, etc.): no augmentation.

Each augmentation function operates on batched tensors
``(boards, policies, values)`` and returns augmented copies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional, Tuple

import torch

if TYPE_CHECKING:
    from board_game import BoardGame


# Type alias for the augmentation function signature.
AugmentFn = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]


def _apply_d4_transform(
    board: torch.Tensor,
    spatial_policy: torch.Tensor,
    transform_id: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply one of the 8 D4 transforms to a single (C,H,W) board and (H,W) policy.

    transform_id 0..3: rotate by 0/90/180/270 degrees.
    transform_id 4..7: horizontal flip then rotate by 0/90/180/270 degrees.
    """
    if transform_id >= 4:
        board = torch.flip(board, dims=[2])
        spatial_policy = torch.flip(spatial_policy, dims=[1])
        transform_id -= 4

    if transform_id > 0:
        board = torch.rot90(board, k=transform_id, dims=[1, 2])
        spatial_policy = torch.rot90(spatial_policy, k=transform_id, dims=[0, 1])

    return board, spatial_policy

def _augment_square_d4(
    boards: torch.Tensor,
    policies: torch.Tensor,
    values: torch.Tensor,
    board_height: int,
    board_width: int,
    has_pass_action: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Randomly apply one of 8 D4 symmetries per sample (batch size unchanged).

    Parameters
    ----------
    boards:  (B, C, H, W)
    policies: (B, num_actions)  where the first H*W entries correspond to
              row-major board positions, and an optional last entry is the
              pass action.
    values:  (B,) — unchanged.
    """
    batch_size = boards.shape[0]
    spatial_actions = board_height * board_width

    # Random transform id per sample: 0..7
    transform_ids = torch.randint(0, 8, (batch_size,))

    spatial = policies[:, :spatial_actions].view(-1, board_height, board_width)
    pass_part = policies[:, spatial_actions:] if has_pass_action else None

    augmented_boards = torch.empty_like(boards)
    augmented_spatial = torch.empty_like(spatial)

    for i in range(batch_size):
        augmented_boards[i], augmented_spatial[i] = _apply_d4_transform(
            boards[i], spatial[i], transform_ids[i].item(),
        )

    augmented_policies = augmented_spatial.reshape(-1, spatial_actions)
    if has_pass_action:
        augmented_policies = torch.cat([augmented_policies, pass_part], dim=1)

    return augmented_boards, augmented_policies, values

def _augment_connect4_flip(
    boards: torch.Tensor,
    policies: torch.Tensor,
    values: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Randomly flip each sample horizontally (batch size unchanged).

    Connect4 policy is column-indexed (length = board_width), so flipping
    the board horizontally just reverses the policy vector.

    Parameters
    ----------
    boards:   (B, C, H, W)
    policies: (B, num_columns)
    values:   (B,)
    """
    # Per-sample coin flip: True = apply horizontal flip.
    flip_mask = torch.randint(0, 2, (boards.shape[0],), dtype=torch.bool)

    flipped_boards = torch.flip(boards, dims=[3])
    flipped_policies = torch.flip(policies, dims=[1])

    # Expand mask to broadcast: (B,1,1,1) for boards, (B,1) for policies.
    board_mask = flip_mask.view(-1, 1, 1, 1).to(boards.device)
    policy_mask = flip_mask.view(-1, 1).to(policies.device)

    augmented_boards = torch.where(board_mask, flipped_boards, boards)
    augmented_policies = torch.where(policy_mask, flipped_policies, policies)

    return augmented_boards, augmented_policies, values


def make_augment_fn(game: "BoardGame") -> Optional[AugmentFn]:
    """Return a batch augmentation function appropriate for *game*, or ``None``.

    Parameters
    ----------
    game:
        A ``BoardGame`` instance.  Uses ``game.name``, ``game.height``,
        ``game.width``, and ``game.num_actions`` to select the symmetry group.
    """
    board_height = game.height
    board_width = game.width
    spatial_actions = board_height * board_width
    has_pass_action = game.num_actions > spatial_actions

    if game.name in ("othello", "gomoku", "tictactoe"):
        if board_height != board_width:
            return None

        def augment_d4(
            boards: torch.Tensor,
            policies: torch.Tensor,
            values: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            return _augment_square_d4(
                boards, policies, values,
                board_height, board_width, has_pass_action,
            )

        return augment_d4

    if game.name == "connect4":
        return _augment_connect4_flip

    # hex and other games: no known safe augmentation.
    return None