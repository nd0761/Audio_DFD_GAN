import torch
import torch.nn as nn

class WaveGANGenerator(nn.Module):
    def __init__(self, input_dim=100, audio_dim=190_000, 
                 initial_depth=5, final_depth=8, num_channels=1):
        super(WaveGANGenerator, self).__init__()
        
        # Fully connected layer to transform noise vector
        self.fc_noise = nn.Sequential(
            nn.Linear(input_dim, audio_dim),
            nn.ReLU()
        )

        self.dims_incr = [num_channels+1]+[pow(2, t) for t in range(initial_depth, final_depth+1)]
        self.dims_decr = [pow(2, t) for t in range(final_depth, initial_depth-1,-1)] + [num_channels]
        
        layers = []
        for i in range(len(self.dims_incr)-1):
            layers += [nn.Conv1d(self.dims_incr[i], self.dims_incr[i+1], 
                                kernel_size=25, stride=4, padding=11)]
            layers += [nn.ReLU()]
        for i in range(len(self.dims_decr)-1):
            out_pad = 1
            if i == len(self.dims_decr) - 2: out_pad = 2
            layers += [nn.ConvTranspose1d(self.dims_decr[i], self.dims_decr[i+1], 
                                kernel_size=25, stride=4, padding=11, output_padding=out_pad)]
            if i == len(self.dims_decr) - 2:
                layers += [nn.Tanh()]
            else:
                layers += [nn.ReLU()]
        # for l in layers:
        #     print(l)

        # Convolutional layers to process concatenated noise and audio
        # self.conv = nn.Sequential(
        #     *layers
        # )

        self.conv = nn.Sequential(
            nn.Conv1d(num_channels + 1, 64, kernel_size=18, stride=4, padding=7),  # [190_000 / 4 = 47500]
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=18, stride=4, padding=7),  # [47500 / 4 = 11875]
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=13, stride=4, padding=7),  # [11875 / 4 = 2970]
            nn.ReLU(),
            nn.ConvTranspose1d(256, 128, kernel_size=13, stride=4, padding=7, output_padding=0),  # [2375 * 5 = 11875]
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
        # t = x.clone()
        # print(t.shape)
        # for l in self.conv:
        #     t = l(t)
        #     print(t.shape)
        # print('---')
        output = self.conv(x)
        return output

class WaveGANDiscriminator(nn.Module):
    def __init__(self, input_dim=190_000, num_channels=1, initial_depth=5, final_depth=8):
        super(WaveGANDiscriminator, self).__init__()

        self.dims_incr = [num_channels]+[pow(2, t) for t in range(initial_depth, final_depth+1)]
        
        layers = []
        for i in range(len(self.dims_incr)-1):
            layers += [nn.Conv1d(self.dims_incr[i], self.dims_incr[i+1], 
                                kernel_size=25, stride=4, padding=11)]
            layers += [nn.LeakyReLU(0.2)]
        
        self.conv = nn.Sequential(
            *layers
        )

        self.conv = nn.Sequential(
            nn.Conv1d(num_channels, 32, kernel_size=18, stride=4, padding=7),
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, 64, kernel_size=18, stride=4, padding=7),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, kernel_size=13, stride=4, padding=7),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 256, kernel_size=20, stride=4, padding=7),
            nn.LeakyReLU(0.2)
        )
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 742, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):

        # t = x.clone()
        # print(t.shape)
        # for l in self.conv:
        #     t = l(t)
        #     print(t.shape)
        # print('---')

        # print(x.shape)
        x = self.conv(x)
        x = self.fc(x)
        return x