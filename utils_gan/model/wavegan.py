
        # self.conv = nn.Sequential(
        #     nn.Conv1d(num_channels + 1, 64, kernel_size=18, stride=4, padding=7),  # [190_000 / 4 = 47500]
        #     nn.ReLU(),
        #     nn.Conv1d(64, 128, kernel_size=18, stride=4, padding=7),  # [47500 / 4 = 11875]
        #     nn.ReLU(),
        #     nn.Conv1d(128, 256, kernel_size=13, stride=4, padding=7),  # [11875 / 4 = 2970]
        #     nn.ReLU(),
# nn.
        #     nn.ConvTranspose1d(256, 128, kernel_size=13, stride=4, padding=7, output_padding=0),  # [2970 -> 11875]
        #     nn.ReLU(),
        #     nn.ConvTranspose1d(128, 64, kernel_size=18, stride=4, padding=7, output_padding=0),  # [11875 * 4 = 47500]
        #     nn.ReLU(),
        #     nn.ConvTranspose1d(64, num_channels, kernel_size=18, stride=4, padding=7, output_padding=0),  # [47500 * 4 = 190000]
        #     nn.Tanh()
        # )
        
import sys
sys.path.insert(0, '/tank/local/ndf3868/GODDS/GAN/utils_gan/model/helpers')

from wave_gan_helper import PhaseShuffle

import torch
import torch.nn as nn

class WaveGANGenerator(nn.Module):
    def __init__(self, input_dim=100, audio_dim=190_000, model_size=64,
                 initial_depth=5, final_depth=8, num_channels=1):
        super(WaveGANGenerator, self).__init__()
        
        # Fully connected layer to transform noise vector
        self.fc_noise = nn.Sequential(
            nn.Linear(input_dim, audio_dim),
            nn.ReLU()
        )

        # self.fc_noise_audio = nn.Sequential(
        #     nn.Linear(audio_dim, model_size*256),
        #     nn.ReLU()
        # )

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
            # nn.Conv1d(num_channels + 1, 64, kernel_size=18, stride=4, padding=7),  # [256 * 64 -> 4096]
            # nn.ReLU(),
            # nn.Conv1d(64, 128, kernel_size=18, stride=4, padding=7),  # [4096 -> 1024]
            # nn.ReLU(),
            # nn.Conv1d(128, 256, kernel_size=18, stride=4, padding=7),  # [1024 -> 256]
            # nn.ReLU(),
            # nn.Linear(256, 2970),
            # nn.ReLU(),
            # nn.ConvTranspose1d(256, 128, kernel_size=13, stride=4, padding=7, output_padding=0),  # -> 11875]
            # nn.ReLU(),
            # nn.ConvTranspose1d(128, 64, kernel_size=18, stride=4, padding=7, output_padding=0),  # -> 47500]
            # nn.ReLU(),
            # nn.ConvTranspose1d(64, num_channels, kernel_size=18, stride=4, padding=7, output_padding=0),  # -> 190000]
            # nn.Tanh()
            # nn.ConvTranspose1d(256, 128,         kernel_size=18, stride=4, padding=7, output_padding=0),  # -> 1024
            # nn.ReLU(),
            # nn.ConvTranspose1d(128, 64,          kernel_size=18, stride=4, padding=7, output_padding=0),  # -> 4096
            # nn.ReLU(),
            # nn.ConvTranspose1d(64, num_channels, kernel_size=18, stride=4, padding=7, output_padding=0),  # -> 16384
            # nn.ReLU(),
            # nn.ConvTranspose1d(64, num_channels, kernel_size=18, stride=4, padding=7, output_padding=0),  # -> 65536
            # nn.ReLU(),
            # nn.ConvTranspose1d(64, num_channels, kernel_size=18, stride=2, padding=7, output_padding=0),  # -> 131074
            # nn.Tanh()
        )
        
    def forward(self, audio, z):
        z = self.fc_noise(z)              # Shape: [batch_size, audio_dim]
        z = z.unsqueeze(1)                # Shape: [batch_size, 1, audio_dim]
        
        x = torch.cat([audio, z], dim=1)  # Shape: [batch_size, num_channels + 1, audio_dim]
        # x = self.fc_noise_audio(x)

        # t = x.clone()
        # for l in self.conv:
        #     print(t.shape)
        #     t = l(t)
        
        output = self.conv(x)
        return output

class WaveGANDiscriminator(nn.Module):
    def __init__(self, audio_dim=190_000, num_channels=1, 
                 shift_factor=2, model_size=64,
                 initial_depth=5, final_depth=8):
        super(WaveGANDiscriminator, self).__init__()

        self.fc_noise_audio = nn.Sequential(
            nn.Linear(audio_dim, model_size*256),
            nn.ReLU()
        )

        self.conv = nn.Sequential(
            nn.Conv1d(num_channels, 16, kernel_size=18, stride=4, padding=7),  # [190_000 / 4 = 47500]
            nn.LeakyReLU(0.2),
            nn.Conv1d(16, 32, kernel_size=18, stride=4, padding=7),  # [47500 / 4 = 11875]
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, 64, kernel_size=13, stride=4, padding=7),  # [11875 / 4 = 2970]
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, kernel_size=18, stride=4, padding=8),  # 2970 -> 743
            nn.LeakyReLU(0.2),
            # nn.Conv1d(128, 256, kernel_size=18, stride=4, padding=8),  # 743 -> 186

            # nn.Conv1d(num_channels, 32, kernel_size=18, stride=4, padding=7),  # [256 * 64 -> 4096]
            # nn.LeakyReLU(0.2),
            # PhaseShuffle(shift_factor),
            # nn.Conv1d(32, 64,           kernel_size=18, stride=4, padding=7),  # [4096 -> 1024]
            # nn.LeakyReLU(0.2),
            # PhaseShuffle(shift_factor),
            # nn.Conv1d(64, 128,          kernel_size=18, stride=4, padding=7),  # [1024 -> 256]
            # nn.LeakyReLU(0.2),
        )
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 743, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x = self.fc_noise_audio(x)
        # t = x.clone()
        # for l in self.conv:
        #     print(t.shape)
        #     t = l(t)
        # print(t.shape)

        x = self.conv(x)
        x = self.fc(x)
        return x