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
    """Residual block with time embedding injection."""

    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.GroupNorm(min(8, in_channels), in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels),
        )
        self.conv2 = nn.Sequential(
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, time_emb):
        residual = self.shortcut(x)
        hidden = self.conv1(x)
        hidden = hidden + self.time_mlp(time_emb)[:, :, None, None]
        hidden = self.conv2(hidden)
        return hidden + residual


class Unet(nn.Module):
    """U-Net noise prediction network for DDPM.

    Args:
        in_channels: Number of input image channels (e.g. 1 for MNIST, 3 for RGB).
        channel_mults: Channel multipliers at each resolution level.
        time_emb_dim: Dimension of the timestep embedding.
    """

    def __init__(self, in_channels=1, channel_mults=(64, 128, 256), time_emb_dim=256,
                 time_emb_mode="sinusoidal", max_timesteps=1000):
        super().__init__()
        self.time_embedding = nn.Sequential(
            TimeEmbedding(time_emb_dim, mode=time_emb_mode, max_timesteps=max_timesteps),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # ---- Encoder (downsampling) ----
        self.encoder_blocks = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        prev_channels = in_channels
        for channels in channel_mults:
            self.encoder_blocks.append(ResidualBlock(prev_channels, channels, time_emb_dim))
            self.downsamplers.append(nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1))
            prev_channels = channels

        # ---- Bottleneck ----
        bottleneck_channels = channel_mults[-1]
        self.bottleneck = ResidualBlock(bottleneck_channels, bottleneck_channels, time_emb_dim)

        # ---- Decoder (upsampling) with skip connections ----
        self.upsamplers = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        reversed_mults = list(reversed(channel_mults))
        for i, channels in enumerate(reversed_mults):
            skip_channels = channels  # from encoder
            input_channels = reversed_mults[i - 1] if i > 0 else bottleneck_channels
            self.upsamplers.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1),
            ))
            self.decoder_blocks.append(ResidualBlock(input_channels + skip_channels, channels, time_emb_dim))

        # ---- Output projection ----
        self.output_conv = nn.Sequential(
            nn.GroupNorm(min(8, channel_mults[0]), channel_mults[0]),
            nn.SiLU(),
            nn.Conv2d(channel_mults[0], in_channels, kernel_size=1),
        )

    def forward(self, x, timesteps):
        time_emb = self.time_embedding(timesteps)

        # Encoder — store skip connections
        skips = []
        hidden = x
        for block, down in zip(self.encoder_blocks, self.downsamplers):
            hidden = block(hidden, time_emb)
            skips.append(hidden)
            hidden = down(hidden)

        # Bottleneck
        hidden = self.bottleneck(hidden, time_emb)

        # Decoder — consume skip connections in reverse
        for up, block, skip in zip(self.upsamplers, self.decoder_blocks, reversed(skips)):
            hidden = up(hidden)
            # Handle spatial size mismatch from odd dimensions
            if hidden.shape != skip.shape:
                hidden = F.interpolate(hidden, size=skip.shape[2:])
            hidden = torch.cat([hidden, skip], dim=1)
            hidden = block(hidden, time_emb)

        return self.output_conv(hidden)

class DDPM:
    """Denoising Diffusion Probabilistic Model.

    Args:
        model: The noise prediction U-Net.
        num_timesteps: Total number of diffusion steps.
        beta_start: Starting noise schedule value.
        beta_end: Ending noise schedule value.
        device: Torch device.
    """

    def __init__(self, model, num_timesteps=1000, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.model = model.to(device)
        self.num_timesteps = num_timesteps
        self.device = device

        # Linear noise schedule
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - self.alpha_cumprod)

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
    channel_mults=(32, 64, 128, 256),
    time_emb_dim=128,
    time_emb_mode="sinusoidal",
    num_timesteps=1000,
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
        dataset_name: 'mnist', 'cifar10', or 'celeba'.
        data_root: Root directory for dataset storage.
        image_size: Resize images to this resolution.
        in_channels: Number of image channels (1 for grayscale, 3 for RGB).
        channel_mults: UNet channel multipliers per resolution level.
        time_emb_dim: Timestep embedding dimension.
        time_emb_mode: 'sinusoidal' or 'learnable'.
        num_timesteps: Number of diffusion steps.
        batch_size: Training batch size.
        num_epochs: Number of training epochs.
        learning_rate: Adam learning rate.
        num_workers: DataLoader workers.
        num_samples: Number of images to generate per epoch.
        device: Torch device string (auto-detected if None).
        output_dir: Directory to save sample images.
        save_every_n_steps: Save checkpoint every N training steps.
        checkpoint_dir: Directory to save model checkpoints.
        use_ddim: Use DDIM sampling instead of DDPM during training.
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

    # Build model
    model = Unet(
        in_channels=in_channels,
        channel_mults=channel_mults,
        time_emb_dim=time_emb_dim,
        time_emb_mode=time_emb_mode,
        max_timesteps=num_timesteps,
    )
    num_parameters = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_parameters:,} | Device: {device} | Dataset: {dataset_name}")

    ddpm = DDPM(model, num_timesteps=num_timesteps, device=device)
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
                    "optimizer_state_dict": optimizer.state_dict(),
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

    parser = argparse.ArgumentParser(description="DDPM: Train or Sample")
    subparsers = parser.add_subparsers(dest="mode", help="Run mode")

    # ---- Train subcommand ----
    train_parser = subparsers.add_parser("train", help="Train a DDPM model")
    train_parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "cifar10", "celeba", "custom"])
    train_parser.add_argument("--data-root", type=str, default="./data")
    train_parser.add_argument("--image-size", type=int, default=32)
    train_parser.add_argument("--in-channels", type=int, default=1)
    train_parser.add_argument("--channel-mults", type=int, nargs="+", default=[32, 64, 128, 256])
    train_parser.add_argument("--time-emb-dim", type=int, default=128)
    train_parser.add_argument("--time-emb-mode", type=str, default="sinusoidal", choices=["sinusoidal", "learnable"])
    train_parser.add_argument("--num-timesteps", type=int, default=1000)
    train_parser.add_argument("--batch-size", type=int, default=128)
    train_parser.add_argument("--epochs", type=int, default=30)
    train_parser.add_argument("--lr", type=float, default=2e-4)
    train_parser.add_argument("--num-workers", type=int, default=2)
    train_parser.add_argument("--num-samples", type=int, default=16)
    train_parser.add_argument("--device", type=str, default=None)
    train_parser.add_argument("--output-dir", type=str, default=".")
    train_parser.add_argument("--save-every", type=int, default=500, help="Save checkpoint every N steps")
    train_parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    train_parser.add_argument("--ddim", action="store_true", help="Use DDIM sampling during training")
    train_parser.add_argument("--ddim-steps", type=int, default=50, help="Number of DDIM denoising steps")
    train_parser.add_argument("--ddim-eta", type=float, default=0.0, help="DDIM stochasticity (0=deterministic, 1=DDPM)")

    # ---- Sample subcommand ----
    sample_parser = subparsers.add_parser("sample", help="Generate samples from a checkpoint")
    sample_parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    sample_parser.add_argument("--image-size", type=int, default=32)
    sample_parser.add_argument("--in-channels", type=int, default=1)
    sample_parser.add_argument("--channel-mults", type=int, nargs="+", default=[32, 64, 128, 256])
    sample_parser.add_argument("--time-emb-dim", type=int, default=128)
    sample_parser.add_argument("--time-emb-mode", type=str, default="sinusoidal", choices=["sinusoidal", "learnable"])
    sample_parser.add_argument("--num-timesteps", type=int, default=1000)
    sample_parser.add_argument("--num-samples", type=int, default=16)
    sample_parser.add_argument("--device", type=str, default=None)
    sample_parser.add_argument("--output", type=str, default="ddpm_generated.png", help="Output image path")
    sample_parser.add_argument("--ddim", action="store_true", help="Use DDIM sampling instead of DDPM")
    sample_parser.add_argument("--ddim-steps", type=int, default=50, help="Number of DDIM denoising steps")
    sample_parser.add_argument("--ddim-eta", type=float, default=0.0, help="DDIM stochasticity (0=deterministic, 1=DDPM)")

    args = parser.parse_args()

    if args.mode == "train":
        train(
            dataset_name=args.dataset,
            data_root=args.data_root,
            image_size=args.image_size,
            in_channels=args.in_channels,
            channel_mults=tuple(args.channel_mults),
            time_emb_dim=args.time_emb_dim,
            time_emb_mode=args.time_emb_mode,
            num_timesteps=args.num_timesteps,
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

        model = Unet(
            in_channels=args.in_channels,
            channel_mults=tuple(args.channel_mults),
            time_emb_dim=args.time_emb_dim,
            time_emb_mode=args.time_emb_mode,
            max_timesteps=args.num_timesteps,
        )
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded checkpoint: {args.checkpoint} (step {checkpoint.get('global_step', 'N/A')})")

        ddpm = DDPM(model, num_timesteps=args.num_timesteps, device=device)
        image_shape = (args.in_channels, args.image_size, args.image_size)
        if args.ddim:
            samples = ddpm.ddim_sample(image_shape, num_samples=args.num_samples, ddim_steps=args.ddim_steps, eta=args.ddim_eta)
            sampler_label = f"ddim(steps={args.ddim_steps},eta={args.ddim_eta})"
        else:
            samples = ddpm.sample(image_shape, num_samples=args.num_samples)
            sampler_label = "ddpm"
        grid = torchvision.utils.make_grid(samples, nrow=4, normalize=True, value_range=(-1, 1))
        torchvision.utils.save_image(grid, args.output)
        print(f"Saved {args.num_samples} samples [{sampler_label}] → {args.output}")

    else:
        parser.print_help()