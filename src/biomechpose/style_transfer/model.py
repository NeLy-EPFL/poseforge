"""
CycleGAN model implementation for fruit fly image translation.
Handles RGB (simulated) <-> Grayscale (experimental) translation.
"""

import torch
import torch.nn as nn
import functools


class ResNetBlock(nn.Module):
    """Residual block with reflection padding."""

    def __init__(self, channels: int, use_dropout: bool = False):
        super().__init__()
        conv_block = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
            nn.ReLU(True),
        ]

        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        conv_block += [
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
        ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class Generator(nn.Module):
    """Generator network with ResNet blocks."""

    def __init__(
        self,
        input_channels: int = 3,
        output_channels: int = 1,
        base_filters: int = 64,
        n_residual_blocks: int = 9,
        use_dropout: bool = False,
    ):
        super().__init__()

        # Initial convolution block
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_channels, base_filters, 7),
            nn.InstanceNorm2d(base_filters),
            nn.ReLU(True),
        ]

        # Downsampling
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [
                nn.Conv2d(
                    base_filters * mult, base_filters * mult * 2, 3, stride=2, padding=1
                ),
                nn.InstanceNorm2d(base_filters * mult * 2),
                nn.ReLU(True),
            ]

        # ResNet blocks
        mult = 2**n_downsampling
        for i in range(n_residual_blocks):
            model += [ResNetBlock(base_filters * mult, use_dropout)]

        # Upsampling
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [
                nn.ConvTranspose2d(
                    base_filters * mult,
                    int(base_filters * mult / 2),
                    3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                nn.InstanceNorm2d(int(base_filters * mult / 2)),
                nn.ReLU(True),
            ]

        # Final convolution
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(base_filters, output_channels, 7),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    """PatchGAN discriminator."""

    def __init__(
        self, input_channels: int = 3, base_filters: int = 64, n_layers: int = 3
    ):
        super().__init__()

        # Initial layer (no normalization)
        sequence = [
            nn.Conv2d(input_channels, base_filters, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
        ]

        # Intermediate layers
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(
                    base_filters * nf_mult_prev,
                    base_filters * nf_mult,
                    4,
                    stride=2,
                    padding=1,
                ),
                nn.InstanceNorm2d(base_filters * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        # Final layers
        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(
                base_filters * nf_mult_prev,
                base_filters * nf_mult,
                4,
                stride=1,
                padding=1,
            ),
            nn.InstanceNorm2d(base_filters * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]

        # Output layer
        sequence += [nn.Conv2d(base_filters * nf_mult, 1, 4, stride=1, padding=1)]

        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)


class CycleGAN(nn.Module):
    """Complete CycleGAN model with both generators and discriminators."""

    def __init__(
        self,
        # Architecture hyperparameters
        generator_base_filters: int = 64,
        generator_n_residual_blocks: int = 9,
        generator_use_dropout: bool = False,
        discriminator_base_filters: int = 64,
        discriminator_n_layers: int = 3,
    ):
        super().__init__()

        # Generators: sim_to_exp (RGB->Gray), exp_to_sim (Gray->RGB)
        self.G_sim_to_exp = Generator(
            input_channels=3,
            output_channels=1,
            base_filters=generator_base_filters,
            n_residual_blocks=generator_n_residual_blocks,
            use_dropout=generator_use_dropout,
        )

        self.G_exp_to_sim = Generator(
            input_channels=1,
            output_channels=3,
            base_filters=generator_base_filters,
            n_residual_blocks=generator_n_residual_blocks,
            use_dropout=generator_use_dropout,
        )

        # Discriminators: for simulated (RGB) and experimental (Gray)
        self.D_sim = Discriminator(
            input_channels=3,
            base_filters=discriminator_base_filters,
            n_layers=discriminator_n_layers,
        )

        self.D_exp = Discriminator(
            input_channels=1,
            base_filters=discriminator_base_filters,
            n_layers=discriminator_n_layers,
        )

    def forward(self, sim_images=None, exp_images=None):
        """Forward pass for training or inference."""
        results = {}

        if sim_images is not None:
            # Simulate -> Experimental -> Simulate (cycle)
            fake_exp = self.G_sim_to_exp(sim_images)
            reconstructed_sim = self.G_exp_to_sim(fake_exp)

            results.update(
                {
                    "fake_exp": fake_exp,
                    "reconstructed_sim": reconstructed_sim,
                    "D_exp_fake": self.D_exp(fake_exp),
                }
            )

        if exp_images is not None:
            # Experimental -> Simulate -> Experimental (cycle)
            fake_sim = self.G_exp_to_sim(exp_images)
            reconstructed_exp = self.G_sim_to_exp(fake_sim)

            results.update(
                {
                    "fake_sim": fake_sim,
                    "reconstructed_exp": reconstructed_exp,
                    "D_sim_fake": self.D_sim(fake_sim),
                }
            )

        # Discriminator outputs for real images
        if sim_images is not None:
            results["D_sim_real"] = self.D_sim(sim_images)
        if exp_images is not None:
            results["D_exp_real"] = self.D_exp(exp_images)

        return results
