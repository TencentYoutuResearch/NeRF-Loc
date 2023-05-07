import torch
import torch.nn as nn
import torch.nn.functional as F

class RayUnet(nn.Module):
    def __init__(self, in_channels, n_samples):
        super().__init__()

        out_channels = in_channels

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, 64, 3, 1, padding=1),
            nn.LayerNorm([64,n_samples]),
            # nn.InstanceNorm1d(64),
            nn.ELU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, 3, 1, padding=1),
            nn.LayerNorm([128,n_samples//2]),
            # nn.InstanceNorm1d(128),
            nn.ELU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 128, 3, 1, padding=1),
            nn.LayerNorm([128,n_samples//4]),
            # nn.InstanceNorm1d(128),
            nn.ELU(inplace=True)
        )
        self.maxpool = nn.MaxPool1d(2)
        self.trans_conv3 = nn.Sequential(
            nn.ConvTranspose1d(128, 128, 3, 2, padding=1, output_padding=1),
            nn.LayerNorm([128,n_samples//4]),
            # nn.InstanceNorm1d(128),
            nn.ELU(inplace=True)
        )
        self.trans_conv2 = nn.Sequential(
            nn.ConvTranspose1d(256, 64, 3, 2, padding=1, output_padding=1),
            nn.LayerNorm([64,n_samples//2]),
            # nn.InstanceNorm1d(64),
            nn.ELU(inplace=True)
        )
        self.trans_conv1 = nn.Sequential(
            nn.ConvTranspose1d(128, 32, 3, 2, padding=1, output_padding=1),
            nn.LayerNorm([32,n_samples]),
            # nn.InstanceNorm1d(32),
            nn.ELU(inplace=True)
        )
        self.conv_out = nn.Sequential(
            nn.Conv1d(in_channels+32, out_channels, 3, 1, padding=1),
            nn.LayerNorm([out_channels, n_samples]),
            # nn.InstanceNorm1d(out_channels),
            nn.ELU(inplace=True)
        )

    def forward(self, x):
        """
            x: B,C,N
        """
        conv1_0 = self.conv1(x)
        conv1 = self.maxpool(conv1_0)
        conv2_0 = self.conv2(conv1)
        conv2 = self.maxpool(conv2_0)
        conv3_0 = self.conv3(conv2)
        conv3 = self.maxpool(conv3_0)
        x_0 = self.trans_conv3(conv3)
        x_1 = self.trans_conv2(torch.cat([conv2, x_0], dim=1))
        x_2 = self.trans_conv1(torch.cat([conv1, x_1], dim=1))
        x_out = self.conv_out(torch.cat([x, x_2], dim=1))
        return x_out
