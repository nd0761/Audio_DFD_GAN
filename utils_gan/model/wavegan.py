import sys
sys.path.insert(0, '/tank/local/ndf3868/GODDS/GAN/utils_gan/model/helpers')

from wave_gan_helper import PhaseShuffle, Transposed1DConv

import torch
import torch.nn as nn

import math

class DEPR_WaveGANGenerator(nn.Module):
    def __init__(self, input_dim=100, audio_dim=190_000, model_size=64,
                 initial_depth=5, final_depth=8, num_channels=1):
        super(WaveGANGenerator, self).__init__()
        
        # Fully connected layer to transform noise vector
        self.fc_noise = nn.Sequential(
            nn.Linear(input_dim, audio_dim),
            nn.ReLU()
        )

        self.conv = nn.Sequential(
            nn.Conv1d(num_channels + 1, 64, kernel_size=18, stride=4, padding=7),  # [190_000 / 4 = 47500]
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=18, stride=4, padding=7),  # [47500 / 4 = 11875]
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=13, stride=4, padding=7),  # [11875 / 4 = 2970]
            nn.ReLU(),
# nn.
            nn.ConvTranspose1d(256, 128, kernel_size=13, stride=4, padding=7, output_padding=0),  # [2970 -> 11875]
            nn.ReLU(),
            nn.ConvTranspose1d(128, 64, kernel_size=18, stride=4, padding=7, output_padding=0),  # [11875 * 4 = 47500]
            nn.ReLU(),
            nn.ConvTranspose1d(64, num_channels, kernel_size=18, stride=4, padding=7, output_padding=0),  # [47500 * 4 = 190000]
            nn.Tanh()
        )
        
    def forward(self, audio, z):
        z = self.fc_noise(z)              # Shape: [batch_size, audio_dim]
        z = z.unsqueeze(1)                # Shape: [batch_size, 1, audio_dim]
        
        x = torch.cat([audio, z], dim=1)  # Shape: [batch_size, num_channels + 1, audio_dim]

        output = self.conv(x)
        return output

class DEPR_WaveGANDiscriminator(nn.Module):
    def __init__(self, audio_dim=190_000, num_channels=1, 
                 shift_factor=2, model_size=64,
                 initial_depth=5, final_depth=8):
        super(WaveGANDiscriminator, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(num_channels, 16, kernel_size=18, stride=4, padding=7),  # [190_000 / 4 = 47500]
            nn.LeakyReLU(0.2),
            nn.Conv1d(16, 32, kernel_size=18, stride=4, padding=7),  # [47500 / 4 = 11875]
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, 64, kernel_size=13, stride=4, padding=7),  # [11875 / 4 = 2970]
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, kernel_size=18, stride=4, padding=8),  # 2970 -> 743
            nn.LeakyReLU(0.2),
        )
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 743, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

class WaveGANGenerator(nn.Module):
    def __init__(self, noise_dim=100, audio_dim=131072, model_size=64,
                 initial_depth=2, final_depth=4, num_channels=1):
        super(WaveGANGenerator, self).__init__()
        
        self.audio_conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=4, kernel_size=20, stride=4, padding=8), #[1, 131072 -> 4, 32768]
            # nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2),
            nn.Conv1d(in_channels=4, out_channels=8, kernel_size=20, stride=4, padding=8), #[4, 32768 -> 8, 8192]
            # nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=20, stride=4, padding=8), #[8, 8192 -> 16, 2048]
            # nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
        )

        self.noise_fc = nn.Sequential(
            nn.Linear(noise_dim, 512),
            nn.ReLU(),
            nn.Linear(512, audio_dim // 64),  # Match shape with the audio feature map after conv layers
            nn.ReLU()
        )
        
        # Combine audio and noise features
        self.comb_conv = nn.Sequential(
            # nn.functional.interpolate(scale_factor=4, mode='nearest'),
            Transposed1DConv(in_channels=16 + 1, out_channels=8, kernel_size=25, stride=1, upsample=4),
            # nn.BatchNorm1d(32),
            # nn.Dropout(0.2),
            nn.ReLU(),
            # nn.ConvTranspose1d(in_channels=8, out_channels=4, kernel_size=18, stride=1, padding=7),
            Transposed1DConv(in_channels=8, out_channels=4, kernel_size=25, stride=1, upsample=4),
            # nn.BatchNorm1d(16),
            nn.ReLU(),
            # nn.ConvTranspose1d(in_channels=4, out_channels=1, kernel_size=18, stride=1, padding=7),
            Transposed1DConv(in_channels=4, out_channels=1, kernel_size=25, stride=1, upsample=4),
            nn.Tanh()  # Output range between -1 and 1
        )
        
    def forward(self, audio, z):
        z       = z.unsqueeze(1)                # Shape: [batch_size, 1, audio_dim]
        z       = self.noise_fc(z)              # Shape: [batch_size, 1, audio_dim//64]
        # print('print noise', z.shape)
        audio   = self.audio_conv(audio)        # Shape: [batch_size, 1, audio_dim//64]
        # print('print audio', audio.shape)
        
        x = torch.cat([audio, z], dim=1)        # Shape: [batch_size, num_channels + 1, audio_dim]
        output = self.comb_conv(x)
        return output

class WaveGANDiscriminator(nn.Module):
    def __init__(self, audio_dim=131072, num_channels=1, 
                 shift_factor=2, model_size=64,
                 initial_depth=5, final_depth=8):
        super(WaveGANDiscriminator, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(num_channels, 4, kernel_size=25, stride=4, padding=11),  # [131072 / 4 = 32768]
            nn.LeakyReLU(0.2),
            nn.Conv1d(4, 8, kernel_size=25, stride=4, padding=11),  # [47500 / 4 = 11875]
            # nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Conv1d(8, 16, kernel_size=25, stride=4, padding=11),  # [11875 / 4 = 2970]
            # nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Conv1d(16, 32, kernel_size=25, stride=4, padding=11),  # 2970 -> 743
            # nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            # nn.Conv1d(num_channels, )
        )
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * audio_dim // (16*16), 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # print('DISC', x.shape)
        x = self.conv(x)
        x = self.fc(x)
        return x