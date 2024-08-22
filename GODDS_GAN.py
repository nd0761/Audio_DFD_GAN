import numpy as np

import random
import json

import torch
from torch import nn
from torch.utils.data import DataLoader, SubsetRandomSampler

import datetime
import sys 
import os
sys.path.append(os.path.abspath('/tank/local/ndf3868/GODDS/GAN'))

# file_path = os.path.realpath(__file__)

import utils_gan

from utils_gan import ASV_DATASET, Generator, Discriminator, Whispers, train
# from utils_gan import  train, test_metrics, set_up_metrics_list, \
#     visualize, visualize_separate


def get_current_timestamp():
    now = datetime.datetime.now()
    timestamp = now.strftime("%d.%m.%y-%H:%M:%S")
    return timestamp

def set_seed(seed: int = 42) -> None:
    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

def get_balanced_indeces(dataset, n_samples_per_class, shuffle):
    class_indeces = []

    for label, indices in dataset.class_indeces.items():
        class_indeces.extend(np.random.choice(indices, n_samples_per_class, replace=False))
    
    if shuffle: np.random.shuffle(class_indeces)
    return class_indeces

# set_seed(3407)
device = 'cuda:2' if torch.cuda.is_available() else 'cpu'

if __name__ == "__main__":
    set_seed(3407)

    bonafide_class      = 0

    train_with_wisper   = True

    data_sample_size    = 2500  # 100 - Debugging| 2500 - full training
    gen_fake            = False # True - Debug   | False - full training

    input_size          = 190_000
    hidden_dim_gen      = 3
    hidden_dim_disc     = 1
    output_size         = 1

    batch_size          = 8

    lr                  = 1e-4

    bootstrap_iterations= 1 #5
    n_epochs            = 50 #15


    # Generator & Optimizer for Generator
    gen = Generator(input_size, hidden_dim=hidden_dim_gen).to(device)
    gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)

    # Discriminator & Optimizer for Discriminator
    disc = Discriminator(input_size, hidden_dim=hidden_dim_disc, output_dim=output_size).to(device)
    disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)

    whisper_model_config_directory = '/tank/local/ndf3868/GODDS/GAN/whisper_config/finetuned_ITW'

    whisper_models_direstories = [
        # models trained on Marco's test split (from pre-trained checkpoint)
        '/tank/local/ndf3868/GODDS/GAN/whisper_config/finetuned_ITW/lfcc_lcnn_20240615_160043',
        '/tank/local/ndf3868/GODDS/GAN/whisper_config/finetuned_ITW/lfcc_specrnet_20240615_172529',
        '/tank/local/ndf3868/GODDS/GAN/whisper_config/finetuned_ITW/mfcc_lcnn_20240615_160143',
        '/tank/local/ndf3868/GODDS/GAN/whisper_config/finetuned_ITW/mfcc_mesonet_20240615_165114',
        '/tank/local/ndf3868/GODDS/GAN/whisper_config/finetuned_ITW/rawnet3_20240615_224507',
        # '/tank/local/ndf3868/GODDS/GAN/whisper_config/finetuned_ITW/whisper_lcnn_20240616_060610',
        # '/tank/local/ndf3868/GODDS/GAN/whisper_config/finetuned_ITW/whisper_lfcc_mesonet_20240615_165928',
        # '/tank/local/ndf3868/GODDS/GAN/whisper_config/finetuned_ITW/whisper_mesonet_20240616_070134',

        # '/tank/local/ndf3868/GODDS/deepfake-whisper/src/models/whisper_lcnn/20240512_000000',
        # '/tank/local/ndf3868/GODDS/deepfake-whisper/src/models/whisper_mesonet/20240512_000000',
        # '/tank/local/ndf3868/GODDS/deepfake-whisper/src/models/whisper_mfcc_lcnn/20240516_005202',
        # '/tank/local/ndf3868/GODDS/deepfake-whisper/src/models/whisper_mfcc_mesonet/20240513_131017',
        # '/tank/local/ndf3868/GODDS/deepfake-whisper/src/models/whisper_mfcc_specrnet/20240516_030405',
        # '/tank/local/ndf3868/GODDS/deepfake-whisper/src/models/whisper_specrnet/20240512_000000',
    ]
    whisp = None
    whisp_opt = None
    whisp     = Whispers(whisper_models_direstories, output_size, device).to(device) # for whisper bonafide class == 1
    whisp_opt = torch.optim.Adam(whisp.process.parameters(), lr=lr)
    set_seed(3407)

    # audio_samples = torch.rand(size=(8, input_size), dtype=torch.float32) - 0.5
    # targets = torch.randint(low=0, high=2, size=(8, 1))

    # rand_pred = whisp.detectors_prediction(audio_samples.to(device))
    # print(rand_pred)

    criterion = nn.BCELoss()

    asv_directory = '/tank/local/ndf3868/GODDS/datasets/ASV'
    print("reading TRAIN dataset")
    train_dataset = ASV_DATASET(asv_directory, 'train', 'LA', class_balance=None, gen_fake=gen_fake) #oversample undersample undersample_all
    print("reading TEST  dataset")
    test_dataset  = ASV_DATASET(asv_directory, 'dev',   'LA', class_balance=None, gen_fake=gen_fake)

    print("sampling TEST dataset")
    sampler = SubsetRandomSampler(get_balanced_indeces(dataset=test_dataset, n_samples_per_class=data_sample_size, shuffle=False))
    test_dataloader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, sampler=sampler)

    for _ in range(bootstrap_iterations):

        print("sampling TRAIN dataset for bootstrap iteration", _)
        sampler = SubsetRandomSampler(get_balanced_indeces(dataset=train_dataset, n_samples_per_class=data_sample_size, shuffle=True))
        # dataset = CustomAudioDataset(audio_samples, targets)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)

        logs_dir = '/tank/local/ndf3868/GODDS/GAN/logs'
        ckpt_dir = '/tank/local/ndf3868/GODDS/GAN/checkpoints/models'

        timestamp = f'{get_current_timestamp()}'
        if gen_fake: timestamp = 'DEBUG_'+timestamp

        os.makedirs(os.path.join(logs_dir, timestamp))
        os.makedirs(os.path.join(ckpt_dir, timestamp))

        os.makedirs(os.path.join(logs_dir, timestamp, 'distr'))
        os.makedirs(os.path.join(logs_dir, timestamp, 'metrics'))

        train(train_with_wisper, 
            train_dataloader, test_dataloader, 
            train_dataset, test_dataset,
            gen,      disc,     whisp,
            gen_opt,  disc_opt, whisp_opt,
            criterion,
            n_epochs, device,
            logs_dir=os.path.join('/tank/local/ndf3868/GODDS/GAN/logs', timestamp),
            ckpt_dir=os.path.join(ckpt_dir, timestamp))