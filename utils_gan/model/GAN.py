import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, audio_dim, hidden_dim):
        super(Generator, self).__init__()

        self.aud_dim = audio_dim
        self.hid_dim = hidden_dim

        dims = [1] + [16 * pow(2, i) for i in range(0, hidden_dim+1)] # 1 16 32 64 32 16
        dims += [16 * pow(2, i) for i in range(hidden_dim-1, -1, -1)]

        list_of_layers = [
            self.generator_block(dims[i], dims[i+1]) 
            for i in range(len(dims)-1)
        ] + [
            nn.Conv1d(dims[-1], 1, kernel_size=15, stride=1, padding=7), # [1, 190_000]
            nn.Tanh()
        ]

        self.augm = nn.Sequential(
            *list_of_layers
            
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

        dims = [1] + [16 * pow(2, i) for i in range(0, hidden_dim+1)] # 1 16 32 64 128

        list_of_layers = [
            self.discriminator_block(dims[i], dims[i+1])
            for i in range(len(dims)-1)
        ]+[
            nn.Flatten(),
            nn.Linear(dims[-1] * audio_dim, output_dim),
            nn.Sigmoid()
        ]

        self.disc = nn.Sequential(
            *list_of_layers
        )

    def forward(self, x):
        return self.disc(x).view(-1)
    
    def discriminator_block(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Conv1d(in_dim, out_dim, kernel_size=15, stride=1, padding=7),  # Output: [16, 190_000]
            nn.LeakyReLU(0.2, inplace=True),
        )