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
                 initial_depth=2, final_depth=4, num_channels=1, noise_chan=128):
        super(WaveGANGenerator, self).__init__()

        self.num_channels = num_channels
        self.audio_dim = audio_dim
        self.noise_chan = noise_chan

        self.audio_conv_last_dim = 32
        
        # self.audio_conv = nn.Sequential(
        #     nn.Conv1d(in_channels=1, out_channels=8, 
        #               kernel_size=20, stride=4, padding=8), #[1, 131072 -> *, 32768]
        #     nn.LeakyReLU(0.2),
        #     nn.Conv1d(in_channels=8, out_channels=16, 
        #               kernel_size=20, stride=4, padding=8), #[*, 32768 -> *, 8192]
        #     nn.LeakyReLU(0.2),
        #     nn.Conv1d(in_channels=16, out_channels=self.audio_conv_last_dim, 
        #               kernel_size=20, stride=4, padding=8), #[*, 8192 -> 32, 2048]
        #     nn.LeakyReLU(0.2),
        # )

        # self.audio_af_conv_dim = audio_dim // (4 * 4 * 4)
        self.audio_af_conv_dim = self.audio_dim // self.noise_chan

        self.noise_fc = nn.Sequential(
            nn.Linear(noise_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, noise_chan * self.audio_af_conv_dim),  # Match shape with the audio feature map after conv layers
            nn.ReLU()
        )

        # two dimensions will be transposed after that point noise_chan + last_out_channels = 64
        
        # Combine audio and noise features
        self.conv = nn.Sequential(
            Transposed1DConv(in_channels=self.audio_af_conv_dim, out_channels=self.audio_af_conv_dim // 2, 
                             kernel_size=25, stride=1, upsample=4), # 64 - 256
            nn.LeakyReLU(0.2),
            # nn.ReLU(),
            nn.BatchNorm1d(self.audio_af_conv_dim // 2),

            Transposed1DConv(in_channels=self.audio_af_conv_dim // 2, out_channels=self.audio_af_conv_dim // 4, 
                             kernel_size=25, stride=1, upsample=4), # 256 - 1024
            nn.LeakyReLU(0.2),
            # nn.ReLU(),
            nn.BatchNorm1d(self.audio_af_conv_dim // 4),

            Transposed1DConv(in_channels=self.audio_af_conv_dim // 4, out_channels=self.audio_af_conv_dim // 8, 
                             kernel_size=25, stride=1, upsample=4), # 1025 - 4096
            nn.LeakyReLU(0.2), 
            # nn.ReLU(),
            nn.BatchNorm1d(self.audio_af_conv_dim // 8),

            Transposed1DConv(in_channels=self.audio_af_conv_dim // 8, out_channels=self.audio_af_conv_dim // 16, 
                             kernel_size=25, stride=1, upsample=4), #4096 - 16...
            nn.LeakyReLU(0.2),  # Output range between -1 and 1
            # nn.ReLU(),
            nn.BatchNorm1d(self.audio_af_conv_dim // 16),

            # Transposed1DConv(in_channels=self.audio_af_conv_dim // 16, out_channels=self.audio_af_conv_dim // 32, 
            #                  kernel_size=25, stride=1, upsample=4), #16... - 65...
            # nn.LeakyReLU(0.2),  # Output range between -1 and 1
            # nn.ReLU(),

            Transposed1DConv(in_channels=self.audio_af_conv_dim // 16, out_channels=num_channels, 
                             kernel_size=25, stride=1, upsample=4), #65... - 131072
            nn.Tanh()  # Output range between -1 and 1
        )
        print('Initialize GEN')
        init_weights(self.conv, 'leaky_relu', 0.2)

        
    def forward(self, audio, z):
        z       = z.unsqueeze(1)                # Shape: [batch_size, 1, audio_dim]
        z       = self.noise_fc(z)              # Shape: [batch_size, noise_chan, audio_dim//64]
        z       = z.view(-1, self.noise_chan, self.audio_af_conv_dim)
        
        audio_after_conv = audio.view(-1, self.noise_chan, self.audio_af_conv_dim) # 32 - 4096
        
        x = audio_after_conv * 0.7 + z * 0.3 # combine noise and processed audio
        
        x = x.view(-1, self.audio_af_conv_dim, self.noise_chan)
        output = self.conv(x)
        return output

class WaveGANDiscriminator(nn.Module):
    def __init__(self, audio_dim=131072, num_channels=1, 
                 shift_factor=2, model_size=2,
                 initial_depth=5, final_depth=8):
        super(WaveGANDiscriminator, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(num_channels, out_channels=model_size, kernel_size=25, stride=4, padding=11),  # [131072 / 4 = 32768]
            # PhaseShuffle(shift_factor),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(model_size),
            # nn.ReLU(),

            nn.Conv1d(in_channels=model_size, out_channels=model_size*2, kernel_size=25, stride=4, padding=11),  # [32768 / 4 = 8192]
            # PhaseShuffle(shift_factor),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(model_size*2),
            # nn.ReLU(),

            nn.Conv1d(in_channels=model_size*2, out_channels=model_size*4, kernel_size=25, stride=4, padding=11),  # [8192 / 4 = 2048]
            # PhaseShuffle(shift_factor),
            nn.LeakyReLU(0.2),
            # nn.BatchNorm1d(model_size*4),
            # nn.ReLU(),
        )
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear((model_size*4) * audio_dim // (4**3), 1),
            nn.Sigmoid()
        )
        print('Initialize DISC')
        init_weights(self.conv, 'leaky_relu', 0.2)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x
    
def init_weights(sequence, activation, slope): #activation - 'relu' 'leaky_relu'
    for l_i, l in enumerate(sequence):
        if isinstance(l, nn.BatchNorm1d):
            continue
        
        if isinstance(l, Transposed1DConv):
            print(l.transpose_ops[1])
            if activation == 'leaky_relu':
                nn.init.kaiming_uniform_(sequence[l_i].transpose_ops[1].weight, mode='fan_in', nonlinearity=activation, a=slope)
            else:
                nn.init.kaiming_uniform_(sequence[l_i].transpose_ops[1].weight, mode='fan_in', nonlinearity=activation)
        elif hasattr(l, "weight"):
            print(l)
            if activation == 'leaky_relu':
                nn.init.kaiming_uniform_(sequence[l_i].weight, mode='fan_in', nonlinearity=activation, a=slope)
            else:
                nn.init.kaiming_uniform_(sequence[l_i].weight, mode='fan_in', nonlinearity=activation)
