import torch
import torch.nn as nn
import torch.nn.functional as F


def weights_init(m):

    classname = m.__class__.__name__

    if hasattr(m, 'weight') and classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)

    elif classname.find('BatchNorm2d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


# ------
# ResNet
# ------
class ResidualBlock(nn.Module):

    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, 3),
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, 3),
            nn.InstanceNorm2d(in_channels)
        )

    def forward(self, x):

        return x + self.block(x)


# ---------
# Generator
# ---------
class ResnetGenerator(nn.Module):

    def __init__(self, channels, num_residual_blocks):
        super(ResnetGenerator, self).__init__()

        # Initial convolution block
        out_channels = 64
        layers = [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(channels, out_channels, 7),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        in_channels = out_channels

        # Downsampling
        for _ in range(2):

            out_channels *= 2
            layers += [
                nn.Conv2d(in_channels, out_channels, 3, 2, 1),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ]
            in_channels = out_channels

        # Residual blocks
        for _ in range(num_residual_blocks):

            layers += [ResidualBlock(out_channels)]

        # Upsampling
        for _ in range(2):

            out_channels //= 2
            layers += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_channels, out_channels, 3, 1, 1),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ]
            in_channels = out_channels

        # Output_layer
        layers += [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(out_channels, channels, 7),
            nn.Tanh()
        ]

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


# -------------
# Discriminator
# -------------
class Discriminator(nn.Module):

    def __init__(self, in_channels):
        super(Discriminator, self).__init__()

        def discriminator_block(in_chs, out_chs, IN=True):

            layers = [nn.Conv2d(in_chs, out_chs, 4, 2, 1)]
            if IN:
                layers.append(nn.InstanceNorm2d(out_chs))
            layers.append(nn.LeakyReLU(0.2, inplace=True))

            return layers

        self.blocks = nn.Sequential(
            *discriminator_block(in_channels, 64, IN=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img):

        return self.blocks(img)


