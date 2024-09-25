import sys
sys.path.insert(0, '/tank/local/ndf3868/GODDS/GAN/utils_gan/model/helpers')

# from wave_gan_helper import PhaseShuffle, Transposed1DConv
#https://wandb.ai/shambhavicodes/vae-gan/reports/An-Introduction-to-VAE-GANs--VmlldzoxMTcxMjM5
#https://dsp.stackexchange.com/questions/55577/generate-audio-data-using-vaegan

import torch
import torch.nn as nn

import math

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
    def forward(self, audio):
        pass

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
    def forward(self, x):
        pass

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
    def forward(self, x):
        pass