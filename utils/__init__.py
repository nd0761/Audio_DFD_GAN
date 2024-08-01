import sys 
import os
sys.path.append(os.path.abspath('/tank/local/ndf3868/GODDS/GAN/utils'))

from dataset.asvdataset import ASV_DATASET

from model.GAN import Generator, Discriminator
from model.whisper_models import Whispers

from training.loss import generator_loss, discriminator_loss
from training.metrics import test_metrics, test_data
from training.train import train