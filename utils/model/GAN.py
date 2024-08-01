import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, audio_dim, hidden_dim):
        super(Generator, self).__init__()

        self.aud_dim = audio_dim
        self.hid_dim = hidden_dim

        self.augm = nn.Sequential(
            self.generator_block(1, 16), # [1, 190_000]
            self.generator_block(16, 32),
            self.generator_block(32, 64),
            self.generator_block(64, 32),
            self.generator_block(32, 16),
            nn.Conv1d(16, 1, kernel_size=15, stride=1, padding=7), # [1, 190_000]
            nn.Tanh()
        )

    def forward(self, x):
        return self.augm(x)
    
    def generator_block(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Conv1d(in_dim, out_dim, kernel_size=15, stride=1, padding=7),  # Output: [16, 190_000]
            nn.ReLU(inplace=True),
        )

class Discriminator(nn.Module):
    def __init__(self, audio_dim, hidden_dim, output_dim):
        super(Discriminator, self).__init__()

        self.aud_dim = audio_dim
        self.hid_dim = hidden_dim
        self.out_dim = output_dim

        self.disc = nn.Sequential(
            self.discriminator_block(1, 16),
            self.discriminator_block(16, 32),
            self.discriminator_block(32, 64),
            self.discriminator_block(64, 128),
            nn.Flatten(),
            nn.Linear(128 * audio_dim, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.disc(x).view(-1)
    
    def discriminator_block(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Conv1d(in_dim, out_dim, kernel_size=15, stride=1, padding=7),  # Output: [16, 190_000]
            nn.LeakyReLU(0.2, inplace=True),
        )