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

file_path = os.path.realpath(__file__)

from utils import ASV_DATASET, Generator, Discriminator, Whispers, train, test_metrics, test_data, set_up_metrics_list, visualize


def get_current_timestamp():
    now = datetime.datetime.now()
    timestamp = now.strftime("%d%m%y-%H:%M:%S")
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
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == "__main__":
    set_seed(3407)

    bonafide_class      = 0

    bootstrap_iterations= 5
    n_epochs            = 15

    input_size  = 190_000
    hidden_size = 200
    output_size = 1

    lr = 1e-8

    batch_size = 32

    # audio_samples = torch.rand(size=(8, input_size), dtype=torch.float32) - 0.5
    # targets = torch.randint(low=0, high=2, size=(8, 1))

    # Generator & Optimizer for Generator
    gen = Generator(input_size, hidden_size).to(device)
    gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)

    # Discriminator & Optimizer for Discriminator
    disc = Discriminator(input_size, hidden_size, output_size).to(device)
    disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)

    # whisper_model_config_directory = '/tank/local/ndf3868/GODDS/GAN/whisper_config/finetuned_ITW'
    # whiper_model_names = [
        
    # ]

    # whisper_models_direstories = [
    #     # models trained on Marco's test split (from pre-trained checkpoint)
    #     '/tank/local/ndf3868/GODDS/deepfake-whisper/src/models/lfcc_lcnn/20240516_050615',
    #     '/tank/local/ndf3868/GODDS/deepfake-whisper/src/models/rawnet3/20240519_202326',
    #     # '/tank/local/ndf3868/GODDS/deepfake-whisper/src/models/whisper_lcnn/20240512_000000',
    #     # '/tank/local/ndf3868/GODDS/deepfake-whisper/src/models/whisper_mesonet/20240512_000000',
    #     # '/tank/local/ndf3868/GODDS/deepfake-whisper/src/models/whisper_mfcc_lcnn/20240516_005202',
    #     # '/tank/local/ndf3868/GODDS/deepfake-whisper/src/models/whisper_mfcc_mesonet/20240513_131017',
    #     # '/tank/local/ndf3868/GODDS/deepfake-whisper/src/models/whisper_mfcc_specrnet/20240516_030405',
    #     # '/tank/local/ndf3868/GODDS/deepfake-whisper/src/models/whisper_specrnet/20240512_000000',
    # ]
    # whisp = Whispers(whisper_models_direstories, output_size, device).to(device)
    # whisp_opt = torch.optim.Adam(disc.parameters(), lr=lr)
    # audio_samples = torch.rand(size=(8, 1, input_size), dtype=torch.float32) - 0.5
    # whisp(audio_samples)

    criterion = nn.MSELoss()

    asv_directory = '/tank/local/ndf3868/GODDS/datasets/ASV'
    print("reading TRAIN dataset")
    train_dataset = ASV_DATASET(asv_directory, 'train', 'LA', class_balance=None) #oversample undersample undersample_all
    print("reading TEST  dataset")
    test_dataset  = ASV_DATASET(asv_directory, 'dev', 'LA', class_balance=None)

    print("sampling TEST dataset")
    sampler = SubsetRandomSampler(get_balanced_indeces(dataset=test_dataset, n_samples_per_class=2500, shuffle=False))
    test_dataloader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, sampler=sampler)

    for _ in range(bootstrap_iterations):

        print("sampling TRAIN dataset for bootstrap iteration", _)
        sampler = SubsetRandomSampler(get_balanced_indeces(dataset=train_dataset, n_samples_per_class=2500, shuffle=True))
        # dataset = CustomAudioDataset(audio_samples, targets)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)


        # print(get_current_timestamp())
        train(train_dataloader, gen, disc, 
            criterion, disc_opt, gen_opt, 
            n_epochs, device)
        
        logs_dir = '/tank/local/ndf3868/GODDS/GAN/logs'

        full_real, full_noised, full_nonnoised = test_data(gen, disc, test_dataloader, device)

        metrics_list = set_up_metrics_list(train_dataset.bonafide_class)
        metrics      = test_metrics(metrics_list, full_real, full_noised, full_nonnoised)

        visualize(full_noised, full_nonnoised, os.path.join(logs_dir, f'density_distribution_{_}.png'))

        timestamp = f'{get_current_timestamp()}'
        timestamp = 'None'
        with open(os.path.join(logs_dir, f"{timestamp}_sample_iteration_{_}.json"), "w") as outfile: 
            json.dump(metrics, outfile)
        print(metrics)