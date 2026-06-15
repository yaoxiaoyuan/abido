"""
DDPM (Denoising Diffusion Probabilistic Models) implementation.

A minimal but complete implementation including:
- TimeEmbedding: sinusoidal (fixed) or learnable timestep embeddings
- ResidualBlock: conv block with time conditioning and skip connection
- Unet: encoder-decoder noise prediction network with skip connections
- DDPM: forward diffusion, loss computation, and reverse sampling

Supports MNIST, CIFAR-10, CelebA, and custom image datasets.
"""

import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class TimeEmbedding(nn.Module):
    """Timestep embedding for diffusion models.

    Args:
        embedding_dim: Output embedding dimension.
        mode: 'sinusoidal' (fixed, default) or 'learnable'.
        max_timesteps: Max number of timesteps (only used when mode='learnable').
    """

    def __init__(self, embedding_dim, mode="sinusoidal", max_timesteps=1000):
        super().__init__()
        self.embedding_table = nn.Embedding(max_timesteps, embedding_dim)

        if mode == "sinusoidal":
            half_dim = embedding_dim // 2
            exponent = -math.log(10000) * torch.arange(half_dim) / half_dim
            positions = torch.arange(max_timesteps).unsqueeze(1).float() * exponent.unsqueeze(0).exp()
            self.embedding_table.weight.data = torch.cat([positions.sin(), positions.cos()], dim=-1)
            self.embedding_table.weight.requires_grad_(False)
        elif mode != "learnable":
            raise ValueError(f"Unknown mode '{mode}', expected 'sinusoidal' or 'learnable'")

    def forward(self, timesteps):
        return self.embedding_table(timesteps)


class ResidualBlock(nn.Module):
    """Residual block with time embedding injection (pre-norm, matches DDPM paper).

    Follows the order: norm → activation → conv, matching the reference TF implementation.
    Supports optional dropout and conv/nin shortcut selection when channel dims differ.
    """

    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.0, conv_shortcut=False):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels),
        )

        self.norm2 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels != out_channels:
            # conv_shortcut uses 3x3, nin shortcut uses 1x1 (as in the reference)
            self.shortcut = (
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
                if conv_shortcut
                else nn.Conv2d(in_channels, out_channels, kernel_size=1)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, time_emb):
        residual = self.shortcut(x)

        hidden = self.conv1(F.silu(self.norm1(x)))
        hidden = hidden + self.time_mlp(time_emb)[:, :, None, None]
        hidden = self.conv2(self.dropout(F.silu(self.norm2(hidden))))

        return hidden + residual


class AttnBlock(nn.Module):
    """Self-attention block for U-Net (spatial attention over feature maps).

    Matches the reference TF implementation: computes Q/K/V via 1x1 convs,
    applies scaled dot-product attention, and projects output back.
    """

    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.query_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.key_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.value_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        hidden = self.norm(x)

        query = self.query_proj(hidden)   # (B, C, H, W)
        key = self.key_proj(hidden)
        value = self.value_proj(hidden)

        # Reshape to (B, H*W, C) for attention
        query = query.reshape(B, C, H * W).permute(0, 2, 1)   # (B, HW, C)
        key = key.reshape(B, C, H * W)                         # (B, C, HW)
        value = value.reshape(B, C, H * W).permute(0, 2, 1)   # (B, HW, C)

        # Scaled dot-product attention: (B, HW, HW)
        attn_weights = torch.bmm(query, key) * (C ** -0.5)
        attn_weights = F.softmax(attn_weights, dim=-1)

        # Weighted sum of values: (B, HW, C) → (B, C, H, W)
        attended = torch.bmm(attn_weights, value).permute(0, 2, 1).reshape(B, C, H, W)
        return x + self.out_proj(attended)


class Unet(nn.Module):
    """U-Net noise prediction network for DDPM.

    Matches the architecture from Ho et al. (2020) and the reference TF implementation:
    - Multiple residual blocks per resolution level
    - Self-attention at specified spatial resolutions
    - Nearest-neighbor upsampling + conv (or avg-pool downsampling as fallback)
    - Strided conv downsampling

    Args:
        in_channels: Number of input image channels (e.g. 1 for MNIST, 3 for RGB).
        image_size: Input spatial resolution (used to determine which levels get attention).
        base_channels: Base channel count (scaled by ch_mult at each level).
        ch_mult: Channel multiplier per resolution level.
        num_res_blocks: Number of residual blocks per resolution level.
        attn_resolutions: Spatial resolutions at which to insert self-attention.
        dropout: Dropout probability inside residual blocks.
        resamp_with_conv: If True, use conv for downsampling/upsampling; else avg-pool/nearest.
        time_emb_dim: Timestep embedding dim (defaults to base_channels * 4).
        time_emb_mode: 'sinusoidal' (fixed) or 'learnable'.
        max_timesteps: Max timestep value (for embedding table size).
        channel_mults: Legacy alias for (base_channels, base_channels*m2, ...) style input.
    """

    def __init__(
        self,
        in_channels=1,
        image_size=32,
        base_channels=128,
        ch_mult=(1, 2, 2, 2),
        num_res_blocks=2,
        attn_resolutions=(16,),
        dropout=0.0,
        resamp_with_conv=True,
        time_emb_dim=None,
        time_emb_mode="sinusoidal",
        max_timesteps=1000,
        channel_mults=None,  # legacy: (32, 64, 128) → base_channels=32, ch_mult=(1,2,4)
        block_size=1,        # space-to-depth block size; 1 = disabled, 2/4 = memory saving mode
    ):
        super().__init__()

        if channel_mults is not None:
            base_channels = channel_mults[0]
            ch_mult = tuple(c // base_channels for c in channel_mults)

        assert block_size >= 1 and (block_size & (block_size - 1)) == 0, \
            "block_size must be a power of 2 (e.g. 1, 2, 4)"
        self.block_size = block_size
        # After space_to_depth, each pixel absorbs block_size² spatial neighbours into channels
        # so the actual channel count seen by the UNet is in_channels * block_size²
        unet_in_channels = in_channels * (block_size ** 2)

        if time_emb_dim is None:
            time_emb_dim = base_channels * 4

        self.time_embedding = nn.Sequential(
            TimeEmbedding(base_channels, mode=time_emb_mode, max_timesteps=max_timesteps),
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        num_resolutions = len(ch_mult)
        attn_resolutions = set(attn_resolutions)

        def make_res_block(in_ch, out_ch):
            return ResidualBlock(in_ch, out_ch, time_emb_dim, dropout=dropout)

        def make_attn_or_identity(channels, resolution):
            return AttnBlock(channels) if resolution in attn_resolutions else nn.Identity()

        # ---- Initial conv ----
        # When block_size > 1, input has been space_to_depth'd: in_channels * block_size²
        self.conv_in = nn.Conv2d(unet_in_channels, base_channels, kernel_size=3, padding=1)

        # ---- Encoder (downsampling) ----
        # Each level has num_res_blocks res+attn pairs followed by an optional downsampler.
        # We track the channel count of every pushed skip so the decoder can consume them
        # in the correct order without guessing.
        self.down_blocks = nn.ModuleList()
        self.down_attns = nn.ModuleList()
        self.downsamplers = nn.ModuleList()

        in_ch = base_channels
        current_res = image_size
        skip_channels = [in_ch]  # first skip = conv_in output

        for level_idx, mult in enumerate(ch_mult):
            out_ch = base_channels * mult
            level_res_blocks = nn.ModuleList()
            level_attn_blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                level_res_blocks.append(make_res_block(in_ch, out_ch))
                level_attn_blocks.append(make_attn_or_identity(out_ch, current_res))
                in_ch = out_ch
                skip_channels.append(in_ch)
            self.down_blocks.append(level_res_blocks)
            self.down_attns.append(level_attn_blocks)

            if level_idx != num_resolutions - 1:
                if resamp_with_conv:
                    self.downsamplers.append(
                        nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=2, padding=1)
                    )
                else:
                    self.downsamplers.append(nn.AvgPool2d(2))
                current_res //= 2
                skip_channels.append(in_ch)  # downsampler output also pushed as skip
            else:
                self.downsamplers.append(nn.Identity())  # sentinel — no push

        # ---- Bottleneck ----
        self.mid_block1 = make_res_block(in_ch, in_ch)
        self.mid_attn = AttnBlock(in_ch)
        self.mid_block2 = make_res_block(in_ch, in_ch)

        # ---- Decoder (upsampling) ----
        # Mirror of encoder: consume skips in reverse order.
        # Each level has num_res_blocks+1 blocks; the extra block absorbs the downsampler skip.
        self.up_blocks = nn.ModuleList()
        self.up_attns = nn.ModuleList()
        self.upsamplers = nn.ModuleList()

        # Reverse the skip channel list so we pop from the end
        remaining_skips = list(reversed(skip_channels))

        for level_idx in reversed(range(num_resolutions)):
            out_ch = base_channels * ch_mult[level_idx]
            level_res_blocks = nn.ModuleList()
            level_attn_blocks = nn.ModuleList()
            for _ in range(num_res_blocks + 1):
                skip_ch = remaining_skips.pop(0)
                level_res_blocks.append(make_res_block(in_ch + skip_ch, out_ch))
                level_attn_blocks.append(make_attn_or_identity(out_ch, current_res))
                in_ch = out_ch
            self.up_blocks.append(level_res_blocks)
            self.up_attns.append(level_attn_blocks)

            if level_idx != 0:
                if resamp_with_conv:
                    self.upsamplers.append(nn.Sequential(
                        nn.Upsample(scale_factor=2, mode="nearest"),
                        nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1),
                    ))
                else:
                    self.upsamplers.append(nn.Upsample(scale_factor=2, mode="nearest"))
                current_res *= 2
            else:
                self.upsamplers.append(nn.Identity())  # sentinel for outermost level

        assert not remaining_skips, f"Unmatched skip channels in decoder init: {remaining_skips}"

        # ---- Output projection ----
        self.norm_out = nn.GroupNorm(min(8, in_ch), in_ch)
        # Output channels match the (possibly space_to_depth'd) input channels
        self.conv_out = nn.Conv2d(in_ch, unet_in_channels, kernel_size=3, padding=1)
        # Store original in_channels for depth_to_space in forward
        self._in_channels = in_channels

    def forward(self, x, timesteps):
        # ---- space_to_depth: reduce spatial size, expand channels ----
        # e.g. [B, 3, 256, 256] + block_size=2 → [B, 12, 128, 128]
        if self.block_size > 1:
            bs = self.block_size
            B, C, H, W = x.shape
            # Rearrange: pick every block_size pixel in H and W, fold into channel dim
            x = x.reshape(B, C, H // bs, bs, W // bs, bs)
            x = x.permute(0, 1, 3, 5, 2, 4)          # (B, C, bs, bs, H', W')
            x = x.reshape(B, C * bs * bs, H // bs, W // bs)

        time_emb = self.time_embedding(timesteps)

        # Initial conv
        hidden = self.conv_in(x)

        # Encoder — push every block output as a skip connection.
        # The downsampler output is also pushed so the decoder's extra block has a skip.
        skips = [hidden]
        for res_blocks, attn_blocks, downsampler in zip(
            self.down_blocks, self.down_attns, self.downsamplers
        ):
            for res_block, attn_block in zip(res_blocks, attn_blocks):
                hidden = res_block(hidden, time_emb)
                hidden = attn_block(hidden)
                skips.append(hidden)
            # downsamplers[-1] is Identity (last level), others actually downsample
            hidden = downsampler(hidden)
            if not isinstance(downsampler, nn.Identity):
                skips.append(hidden)

        # Bottleneck
        hidden = self.mid_block1(hidden, time_emb)
        hidden = self.mid_attn(hidden)
        hidden = self.mid_block2(hidden, time_emb)

        # Decoder — pop skips in reverse; upsamplers[-1] is Identity (innermost level)
        for res_blocks, attn_blocks, upsampler in zip(
            self.up_blocks, self.up_attns, self.upsamplers
        ):
            for res_block, attn_block in zip(res_blocks, attn_blocks):
                skip = skips.pop()
                hidden = torch.cat([hidden, skip], dim=1)
                hidden = res_block(hidden, time_emb)
                hidden = attn_block(hidden)
            if not isinstance(upsampler, nn.Identity):
                hidden = upsampler(hidden)

        assert not skips, f"{len(skips)} unused skip connections"

        hidden = F.silu(self.norm_out(hidden))
        out = self.conv_out(hidden)

        # ---- depth_to_space: restore original spatial size and channels ----
        # e.g. [B, 12, 128, 128] + block_size=2 → [B, 3, 256, 256]
        if self.block_size > 1:
            bs = self.block_size
            B, C_expanded, H_small, W_small = out.shape
            C = self._in_channels
            out = out.reshape(B, C, bs, bs, H_small, W_small)
            out = out.permute(0, 1, 4, 2, 5, 3)      # (B, C, H', bs, W', bs)
            out = out.reshape(B, C, H_small * bs, W_small * bs)

        return out


class DDPM:
    """Denoising Diffusion Probabilistic Model.

    Args:
        model: The noise prediction U-Net.
        num_timesteps: Total number of diffusion steps T.
        beta_schedule: Noise schedule type: 'linear', 'quad', 'warmup10', 'warmup50', 'const', 'jsd'.
        beta_start: Starting beta value.
        beta_end: Ending beta value.
        device: Torch device.
    """

    def __init__(self, model, num_timesteps=1000, beta_schedule="linear",
                 beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.model = model.to(device)
        self.num_timesteps = num_timesteps
        self.device = device

        betas_np = self._make_beta_schedule(beta_schedule, beta_start, beta_end, num_timesteps)
        self.betas = torch.tensor(betas_np, dtype=torch.float32, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - self.alpha_cumprod)

    @staticmethod
    def _make_beta_schedule(schedule, beta_start, beta_end, num_timesteps):
        """Build a numpy beta array for the requested schedule."""
        if schedule == "linear":
            return np.linspace(beta_start, beta_end, num_timesteps, dtype=np.float64)
        elif schedule == "quad":
            # quadratic: beta grows as a squared curve (slower start)
            return np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_timesteps, dtype=np.float64) ** 2
        elif schedule == "warmup10":
            betas = beta_end * np.ones(num_timesteps, dtype=np.float64)
            warmup = int(num_timesteps * 0.1)
            betas[:warmup] = np.linspace(beta_start, beta_end, warmup, dtype=np.float64)
            return betas
        elif schedule == "warmup50":
            betas = beta_end * np.ones(num_timesteps, dtype=np.float64)
            warmup = int(num_timesteps * 0.5)
            betas[:warmup] = np.linspace(beta_start, beta_end, warmup, dtype=np.float64)
            return betas
        elif schedule == "const":
            return beta_end * np.ones(num_timesteps, dtype=np.float64)
        elif schedule == "jsd":
            # 1/T, 1/(T-1), ..., 1  — equal signal-to-noise ratio per step
            return 1.0 / np.linspace(num_timesteps, 1, num_timesteps, dtype=np.float64)
        else:
            raise ValueError(f"Unknown beta_schedule '{schedule}'. "
                             f"Choose from: linear, quad, warmup10, warmup50, const, jsd")

    def forward_diffusion(self, x_0, timesteps, noise=None):
        """q(x_t | x_0): add noise to clean images."""
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_alpha = self.sqrt_alpha_cumprod[timesteps][:, None, None, None]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alpha_cumprod[timesteps][:, None, None, None]
        return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise, noise

    def compute_loss(self, x_0):
        """Training loss: MSE between true noise and predicted noise."""
        batch_size = x_0.shape[0]
        timesteps = torch.randint(0, self.num_timesteps, (batch_size,), device=self.device)
        noisy_images, noise = self.forward_diffusion(x_0, timesteps)
        predicted_noise = self.model(noisy_images, timesteps)
        return F.mse_loss(predicted_noise, noise)

    @torch.no_grad()
    def sample(self, image_shape, num_samples=16):
        """Generate images via reverse diffusion (DDPM sampling)."""
        self.model.eval()
        x_t = torch.randn(num_samples, *image_shape, device=self.device)

        for t in reversed(range(self.num_timesteps)):
            timesteps = torch.full((num_samples,), t, device=self.device, dtype=torch.long)
            predicted_noise = self.model(x_t, timesteps)

            alpha_t = self.alphas[t]
            alpha_cumprod_t = self.alpha_cumprod[t]
            beta_t = self.betas[t]

            # Reverse step mean
            mean = (1.0 / torch.sqrt(alpha_t)) * (
                x_t - (beta_t / torch.sqrt(1.0 - alpha_cumprod_t)) * predicted_noise
            )

            if t > 0:
                noise = torch.randn_like(x_t)
                x_t = mean + torch.sqrt(beta_t) * noise
            else:
                x_t = mean

        self.model.train()
        return x_t.clamp(-1, 1)

    @torch.no_grad()
    def ddim_sample(self, image_shape, num_samples=16, ddim_steps=50, eta=0.0):
        """Generate images via DDIM (faster, optionally deterministic) sampling.

        Args:
            image_shape: (C, H, W) of the target image.
            num_samples: Number of images to generate.
            ddim_steps: Number of denoising steps (subset of full T).
            eta: Stochasticity control. 0 = deterministic, 1 = DDPM equivalent.
        """
        self.model.eval()

        # Build a subsequence of timesteps, evenly spaced
        step_size = self.num_timesteps // ddim_steps
        timestep_sequence = list(range(0, self.num_timesteps, step_size))

        x_t = torch.randn(num_samples, *image_shape, device=self.device)

        for i in reversed(range(len(timestep_sequence))):
            t = timestep_sequence[i]
            t_prev = timestep_sequence[i - 1] if i > 0 else 0

            t_batch = torch.full((num_samples,), t, device=self.device, dtype=torch.long)
            predicted_noise = self.model(x_t, t_batch)

            alpha_cumprod_t = self.alpha_cumprod[t]
            alpha_cumprod_t_prev = self.alpha_cumprod[t_prev] if i > 0 else torch.tensor(1.0, device=self.device)

            # Predict x_0 from x_t and predicted noise
            predicted_x0 = (x_t - torch.sqrt(1.0 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_cumprod_t)
            predicted_x0 = predicted_x0.clamp(-1, 1)

            # Compute sigma (controls stochasticity)
            sigma_t = eta * torch.sqrt(
                (1.0 - alpha_cumprod_t_prev) / (1.0 - alpha_cumprod_t)
                * (1.0 - alpha_cumprod_t / alpha_cumprod_t_prev)
            )

            # Direction pointing to x_t
            direction = torch.sqrt(1.0 - alpha_cumprod_t_prev - sigma_t ** 2) * predicted_noise

            # Combine
            x_t = torch.sqrt(alpha_cumprod_t_prev) * predicted_x0 + direction

            if i > 0 and eta > 0:
                x_t = x_t + sigma_t * torch.randn_like(x_t)

        self.model.train()
        return x_t.clamp(-1, 1)


def train(
    dataset_name="mnist",
    data_root="./data",
    image_size=32,
    in_channels=1,
    # ---- UNet architecture ----
    base_channels=128,
    ch_mult=(1, 2, 2, 2),
    num_res_blocks=2,
    attn_resolutions=(16,),
    dropout=0.0,
    resamp_with_conv=True,
    time_emb_mode="sinusoidal",
    block_size=1,
    # ---- Diffusion schedule ----
    num_timesteps=1000,
    beta_schedule="linear",
    beta_start=1e-4,
    beta_end=0.02,
    # ---- Training ----
    batch_size=128,
    num_epochs=10,
    learning_rate=2e-4,
    num_workers=2,
    num_samples=16,
    device=None,
    output_dir=".",
    save_every_n_steps=500,
    checkpoint_dir="./checkpoints",
    use_ddim=False,
    ddim_steps=50,
    ddim_eta=0.0,
):
    """Train a DDPM model.

    Args:
        dataset_name: 'mnist', 'cifar10', 'celeba', or 'custom'.
        data_root: Root directory for dataset storage.
        image_size: Resize images to this resolution.
        in_channels: Number of image channels (1 for grayscale, 3 for RGB).
        base_channels: UNet base channel count (scaled by ch_mult at each level).
        ch_mult: Channel multiplier tuple per resolution level.
        num_res_blocks: Number of residual blocks per resolution level.
        attn_resolutions: Spatial resolutions at which to apply self-attention.
        dropout: Dropout probability inside residual blocks.
        resamp_with_conv: Use conv for up/downsampling instead of pool/nearest.
        time_emb_mode: 'sinusoidal' (fixed) or 'learnable'.
        block_size: Space-to-depth block size (1 = disabled, 2/4 = memory saving).
        num_timesteps: Total number of diffusion steps T.
        beta_schedule: Noise schedule type: 'linear', 'quad', 'warmup10', 'warmup50', 'const', 'jsd'.
        beta_start: Starting beta value.
        beta_end: Ending beta value.
        batch_size: Training batch size.
        num_epochs: Number of training epochs.
        learning_rate: Adam learning rate.
        num_workers: DataLoader workers.
        num_samples: Number of images to generate per checkpoint save.
        device: Torch device string (auto-detected if None).
        output_dir: Directory to save sample images.
        save_every_n_steps: Save checkpoint every N training steps.
        checkpoint_dir: Directory to save model checkpoints.
        use_ddim: Use DDIM sampling instead of DDPM when saving samples.
        ddim_steps: Number of DDIM denoising steps.
        ddim_eta: DDIM stochasticity (0 = deterministic, 1 = DDPM equivalent).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    # Build transform
    normalize_mean = [0.5] * in_channels
    normalize_std = [0.5] * in_channels
    transform_list = [transforms.Resize(image_size)]
    if dataset_name == "celeba":
        transform_list.append(transforms.CenterCrop(image_size))
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(normalize_mean, normalize_std),
    ])
    transform = transforms.Compose(transform_list)

    # Build dataset
    dataset_builders = {
        "mnist": lambda: datasets.MNIST(root=data_root, train=True, download=True, transform=transform),
        "cifar10": lambda: datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform),
        "celeba": lambda: datasets.CelebA(root=data_root, split="train", download=True, transform=transform),
        "custom": lambda: datasets.ImageFolder(root=data_root, transform=transform),
    }
    if dataset_name not in dataset_builders:
        raise ValueError(f"Unknown dataset '{dataset_name}', choose from {list(dataset_builders.keys())}")
    dataset = dataset_builders[dataset_name]()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # UNet image_size seen by the model accounts for block_size reduction
    unet_image_size = image_size // block_size

    # Build model
    model = Unet(
        in_channels=in_channels,
        image_size=unet_image_size,
        base_channels=base_channels,
        ch_mult=ch_mult,
        num_res_blocks=num_res_blocks,
        attn_resolutions=attn_resolutions,
        dropout=dropout,
        resamp_with_conv=resamp_with_conv,
        time_emb_mode=time_emb_mode,
        max_timesteps=num_timesteps,
        block_size=block_size,
    )
    num_parameters = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_parameters:,} | Device: {device} | Dataset: {dataset_name}")
    print(f"UNet: base_ch={base_channels}, ch_mult={ch_mult}, num_res_blocks={num_res_blocks}, "
          f"attn_res={attn_resolutions}, block_size={block_size}")
    print(f"Diffusion: T={num_timesteps}, schedule={beta_schedule}, β=[{beta_start}, {beta_end}]")

    ddpm = DDPM(model, num_timesteps=num_timesteps, beta_schedule=beta_schedule,
                beta_start=beta_start, beta_end=beta_end, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Training loop
    global_step = 0
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch_idx, (images, _) in enumerate(dataloader):
            images = images.to(device)
            loss = ddpm.compute_loss(images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            global_step += 1
            print(f"\r  Batch {batch_idx + 1}/{len(dataloader)} loss: {loss.item():.4f}", end="")

            # Save checkpoint and generate samples every N steps
            if global_step % save_every_n_steps == 0:
                ckpt_path = os.path.join(checkpoint_dir, f"ddpm_step{global_step}.pt")
                torch.save({
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "model_state_dict": model.state_dict(),
                    #"optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss.item(),
                }, ckpt_path)
                print(f"\n  Checkpoint saved → {ckpt_path}")

                image_shape = (in_channels, image_size, image_size)
                if use_ddim:
                    samples = ddpm.ddim_sample(image_shape, num_samples=num_samples, ddim_steps=ddim_steps, eta=ddim_eta)
                    sampler_label = f"ddim(steps={ddim_steps},eta={ddim_eta})"
                else:
                    samples = ddpm.sample(image_shape, num_samples=num_samples)
                    sampler_label = "ddpm"
                grid = torchvision.utils.make_grid(samples, nrow=4, normalize=True, value_range=(-1, 1))
                save_path = os.path.join(output_dir, f"{sampler_label}_samples_step{global_step}.png")
                torchvision.utils.save_image(grid, save_path)
                print(f"  Saved {num_samples} samples [{sampler_label}] → {save_path}")

        avg_loss = total_loss / len(dataloader)
        print(f"\nEpoch {epoch + 1}/{num_epochs} | Avg Loss: {avg_loss:.4f}")


if __name__ == "__main__":
    import argparse

    BETA_SCHEDULES = ["linear", "quad", "warmup10", "warmup50", "const", "jsd"]

    parser = argparse.ArgumentParser(description="DDPM: Train or Sample")
    subparsers = parser.add_subparsers(dest="mode", help="Run mode")

    # ---- Train subcommand ----
    train_parser = subparsers.add_parser("train", help="Train a DDPM model")
    # Dataset
    train_parser.add_argument("--dataset", type=str, default="mnist",
                              choices=["mnist", "cifar10", "celeba", "custom"])
    train_parser.add_argument("--data-root", type=str, default="./data")
    train_parser.add_argument("--image-size", type=int, default=32)
    train_parser.add_argument("--in-channels", type=int, default=1)
    # UNet architecture
    train_parser.add_argument("--base-channels", type=int, default=128,
                              help="UNet base channel count")
    train_parser.add_argument("--ch-mult", type=int, nargs="+", default=[1, 2, 2, 2],
                              help="Channel multipliers per resolution level")
    train_parser.add_argument("--num-res-blocks", type=int, default=2,
                              help="Residual blocks per resolution level")
    train_parser.add_argument("--attn-resolutions", type=int, nargs="+", default=[16],
                              help="Spatial resolutions at which to apply self-attention")
    train_parser.add_argument("--dropout", type=float, default=0.0)
    train_parser.add_argument("--no-resamp-conv", action="store_true",
                              help="Use avg-pool/nearest instead of conv for resampling")
    train_parser.add_argument("--time-emb-mode", type=str, default="sinusoidal",
                              choices=["sinusoidal", "learnable"])
    train_parser.add_argument("--block-size", type=int, default=1,
                              help="Space-to-depth block size (1=disabled, 2/4=memory saving)")
    # Diffusion schedule
    train_parser.add_argument("--num-timesteps", type=int, default=1000)
    train_parser.add_argument("--beta-schedule", type=str, default="linear", choices=BETA_SCHEDULES)
    train_parser.add_argument("--beta-start", type=float, default=1e-4)
    train_parser.add_argument("--beta-end", type=float, default=0.02)
    # Training
    train_parser.add_argument("--batch-size", type=int, default=128)
    train_parser.add_argument("--epochs", type=int, default=30)
    train_parser.add_argument("--lr", type=float, default=2e-4)
    train_parser.add_argument("--num-workers", type=int, default=2)
    train_parser.add_argument("--num-samples", type=int, default=16)
    train_parser.add_argument("--device", type=str, default=None)
    train_parser.add_argument("--output-dir", type=str, default=".")
    train_parser.add_argument("--save-every", type=int, default=500,
                              help="Save checkpoint every N steps")
    train_parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    # Sampling
    train_parser.add_argument("--ddim", action="store_true",
                              help="Use DDIM sampling when saving samples")
    train_parser.add_argument("--ddim-steps", type=int, default=50)
    train_parser.add_argument("--ddim-eta", type=float, default=0.0,
                              help="DDIM stochasticity (0=deterministic, 1=DDPM)")

    # ---- Sample subcommand ----
    sample_parser = subparsers.add_parser("sample", help="Generate samples from a checkpoint")
    sample_parser.add_argument("--checkpoint", type=str, required=True,
                               help="Path to model checkpoint (.pt)")
    sample_parser.add_argument("--image-size", type=int, default=32)
    sample_parser.add_argument("--in-channels", type=int, default=1)
    # Must match the architecture used during training
    sample_parser.add_argument("--base-channels", type=int, default=128)
    sample_parser.add_argument("--ch-mult", type=int, nargs="+", default=[1, 2, 2, 2])
    sample_parser.add_argument("--num-res-blocks", type=int, default=2)
    sample_parser.add_argument("--attn-resolutions", type=int, nargs="+", default=[16])
    sample_parser.add_argument("--dropout", type=float, default=0.0)
    sample_parser.add_argument("--no-resamp-conv", action="store_true")
    sample_parser.add_argument("--time-emb-mode", type=str, default="sinusoidal",
                               choices=["sinusoidal", "learnable"])
    sample_parser.add_argument("--block-size", type=int, default=1)
    sample_parser.add_argument("--num-timesteps", type=int, default=1000)
    sample_parser.add_argument("--beta-schedule", type=str, default="linear", choices=BETA_SCHEDULES)
    sample_parser.add_argument("--beta-start", type=float, default=1e-4)
    sample_parser.add_argument("--beta-end", type=float, default=0.02)
    sample_parser.add_argument("--num-samples", type=int, default=16)
    sample_parser.add_argument("--device", type=str, default=None)
    sample_parser.add_argument("--output", type=str, default="ddpm_generated.png")
    sample_parser.add_argument("--ddim", action="store_true")
    sample_parser.add_argument("--ddim-steps", type=int, default=50)
    sample_parser.add_argument("--ddim-eta", type=float, default=0.0)

    args = parser.parse_args()

    if args.mode == "train":
        train(
            dataset_name=args.dataset,
            data_root=args.data_root,
            image_size=args.image_size,
            in_channels=args.in_channels,
            base_channels=args.base_channels,
            ch_mult=tuple(args.ch_mult),
            num_res_blocks=args.num_res_blocks,
            attn_resolutions=tuple(args.attn_resolutions),
            dropout=args.dropout,
            resamp_with_conv=not args.no_resamp_conv,
            time_emb_mode=args.time_emb_mode,
            block_size=args.block_size,
            num_timesteps=args.num_timesteps,
            beta_schedule=args.beta_schedule,
            beta_start=args.beta_start,
            beta_end=args.beta_end,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            num_workers=args.num_workers,
            num_samples=args.num_samples,
            device=args.device,
            output_dir=args.output_dir,
            save_every_n_steps=args.save_every,
            checkpoint_dir=args.checkpoint_dir,
            use_ddim=args.ddim,
            ddim_steps=args.ddim_steps,
            ddim_eta=args.ddim_eta,
        )

    elif args.mode == "sample":
        device = args.device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

        unet_image_size = args.image_size // args.block_size
        model = Unet(
            in_channels=args.in_channels,
            image_size=unet_image_size,
            base_channels=args.base_channels,
            ch_mult=tuple(args.ch_mult),
            num_res_blocks=args.num_res_blocks,
            attn_resolutions=tuple(args.attn_resolutions),
            dropout=args.dropout,
            resamp_with_conv=not args.no_resamp_conv,
            time_emb_mode=args.time_emb_mode,
            max_timesteps=args.num_timesteps,
            block_size=args.block_size,
        )
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded checkpoint: {args.checkpoint} (step {checkpoint.get('global_step', 'N/A')})")

        ddpm = DDPM(model, num_timesteps=args.num_timesteps, beta_schedule=args.beta_schedule,
                    beta_start=args.beta_start, beta_end=args.beta_end, device=device)
        image_shape = (args.in_channels, args.image_size, args.image_size)
        if args.ddim:
            samples = ddpm.ddim_sample(image_shape, num_samples=args.num_samples,
                                       ddim_steps=args.ddim_steps, eta=args.ddim_eta)
            sampler_label = f"ddim(steps={args.ddim_steps},eta={args.ddim_eta})"
        else:
            samples = ddpm.sample(image_shape, num_samples=args.num_samples)
            sampler_label = "ddpm"
        grid = torchvision.utils.make_grid(samples, nrow=4, normalize=True, value_range=(-1, 1))
        torchvision.utils.save_image(grid, args.output)
        print(f"Saved {args.num_samples} samples [{sampler_label}] → {args.output}")

    else:
        parser.print_help()