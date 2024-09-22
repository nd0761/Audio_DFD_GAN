import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", "is_categorical_dtype")
warnings.filterwarnings("ignore", "use_inf_as_na")

import numpy as np

import random
import json

import torch
from torch import nn
from torch.utils.data import DataLoader, SubsetRandomSampler

import datetime
import sys 
import os
# sys.path.append(os.path.abspath('/tank/local/ndf3868/GODDS/GAN'))

# file_path = os.path.realpath(__file__)

import config
import utils_gan

from utils_gan import ASV_DATASET, ITW_DATSET, \
        Generator, Discriminator, \
        WaveGANGenerator, WaveGANDiscriminator, \
            Whispers, train
# from utils_gan import  train, test_metrics, set_up_metrics_list, \
#     visualize, visualize_separate

import sys
from IPython.core.ultratb import ColorTB

sys.excepthook = ColorTB()

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



def initialize_whisper():
     # Whisper models & Optimizer for Whisper models

    # IN THE WILD IN vss5
    # InTheWild: /tank/local/hgi0312/data/release_in_the_wild
    # metadata in csv, /tank/local/ndf3868/GODDS/GAN/checkpoints/datasets/inthewild_test_ids.pickle has ids for TEST

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
    ]
    whisp     = Whispers(whisper_models_direstories, 
                         config.output_size, config.device).to(config.device) # for whisper bonafide class == 1
    whisp_opt = torch.optim.Adam(whisp.process.parameters(), lr=config.lr)
    return whisp, whisp_opt

def initialize_gan():
    if config.train_with_wavegan:
        gen = WaveGANGenerator(noise_dim=config.noise_size, 
                               audio_dim=config.input_size, num_channels=1, 
                               initial_depth=config.wave_gen_initial_depth, 
                                 final_depth=config.wave_gen_final_depth).to(config.device)
        gen_opt = torch.optim.Adam(gen.parameters(),    lr=config.lr_gen, 
                                   betas=(config.beta1, config.beta2))

        disc = WaveGANDiscriminator(audio_dim=config.input_size, 
                                    num_channels=1, 
                                    initial_depth=config.wave_disc_initial_depth,
                                    final_depth=config.wave_disc_final_depth).to(config.device)
        disc_opt = torch.optim.Adam(disc.parameters(),  lr=config.lr_dis, betas=(config.beta1, config.beta2))
    else:
        gen = Generator(config.input_size, hidden_dim=config.hidden_dim_gen).to(config.device)
        gen_opt = torch.optim.Adam(gen.parameters(),    lr=config.lr_gen, betas=(config.beta1, config.beta2))

        # Discriminator & Optimizer for Discriminator
        disc = Discriminator(config.input_size, 
                            hidden_dim=config.hidden_dim_disc, output_dim=config.output_size).to(config.device)
        disc_opt = torch.optim.Adam(disc.parameters(),  lr=config.lr_dis, betas=(config.beta1, config.beta2))
    return gen, gen_opt, disc, disc_opt

def initialize_dataset():
    tr_da_path = os.path.join(config.dapt_dir, f'{config.dataset_type}_train.pt')
    te_da_path = os.path.join(config.dapt_dir, f'{config.dataset_type}_test.pt')
    if config.dataset_type == 'asv':
        # Loading up datasets
        if os.path.exists(tr_da_path):
            print("loading TRAIN dataset")
            train_dataset = torch.load(tr_da_path)
        else:
            print("reading TRAIN dataset")
            train_dataset = ASV_DATASET(config.asv_directory, 'train', 'LA', 
                                        class_balance=None, gen_fake=config.gen_fake) #it supports oversample, undersample, and undersample_all but preferably use SubsetRandomSampler
            print('saving  TRAIN dataset')
            torch.save(train_dataset, tr_da_path)
        print('----')
        if os.path.exists(te_da_path):
            print("loading TRAIN dataset")
            test_dataset = torch.load(te_da_path)
        else:
            print("reading TEST  dataset")
            test_dataset  = ASV_DATASET(config.asv_directory, 'dev',   'LA', 
                                        class_balance=None, gen_fake=config.gen_fake)
            print('saving  TEST  dataset')
            torch.save(train_dataset, tr_da_path)

        print("sampling TEST dataset")
        sampler = SubsetRandomSampler(get_balanced_indeces(dataset=test_dataset, 
                                                           n_samples_per_class=config.data_sample_size, shuffle=False))
        test_dataloader  = DataLoader(test_dataset,  batch_size=config.batch_size, shuffle=False, sampler=sampler)
        # test_dataset = None
        train_dataloader = None # it is sampled in bootstrap iterations ecery time to make training unique every time
    elif config.dataset_type == 'wild':
        if os.path.exists(tr_da_path):
            print("loading TRAIN dataset")
            train_dataset = torch.load(tr_da_path)
        else:
            print("reading TRAIN dataset")
            train_dataset = ITW_DATSET('train')
            print('saving  TRAIN dataset')
            torch.save(train_dataset, tr_da_path)
        print('----')
        if os.path.exists(te_da_path):
            print("loading TEST dataset")
            test_dataset = torch.load(te_da_path)
        else:
            print("reading TEST dataset")
            test_dataset = ITW_DATSET('test')
            print('saving  TEST dataset')
            torch.save(train_dataset, te_da_path)

        train_dataloader = DataLoader(train_dataset,  batch_size=config.batch_size, shuffle=True)
        test_dataloader  = DataLoader(test_dataset,  batch_size=config.batch_size, shuffle=False)
        # test_dataset = None
    else:
        print("Unsupported dataset type:", config.dataset_type)
        raise ValueError()
    print("finished setting up datasets")
    return train_dataset, train_dataloader, test_dataset, test_dataloader 



def set_up_logs_dir():
    timestamp = f'{get_current_timestamp()}'
    if config.gen_fake or config.DEBUG: timestamp = 'DEBUG_'+timestamp

    os.makedirs(os.path.join(config.logs_dir, timestamp))
    os.makedirs(os.path.join(config.ckpt_dir, timestamp))

    os.makedirs(os.path.join(config.logs_dir, timestamp, 'distr'))
    os.makedirs(os.path.join(config.logs_dir, timestamp, 'metrics'))
    os.makedirs(os.path.join(config.logs_dir, timestamp, 'audio'))
    os.makedirs(os.path.join(config.logs_dir, timestamp, 'spectrograms'))
    return timestamp

def bootstrap_iteration(
        gen,      disc,     whisp,
        gen_opt,  disc_opt, whisp_opt,
        criterion,
        train_dataloader, test_dataloader, 
        train_dataset, test_dataset):
    
    print("sampling TRAIN dataset for bootstrap iteration")
    if config.dataset_type == 'asv':
        sampler = SubsetRandomSampler(get_balanced_indeces(dataset=train_dataset, 
                                                        n_samples_per_class=config.data_sample_size, shuffle=True))
        train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, sampler=sampler)
    
    timestamp = 'none'
    if config.save_logs: timestamp = set_up_logs_dir()

    train(config.train_with_wisper, 
        train_dataloader, test_dataloader, 
        train_dataset, test_dataset,
        gen,      disc,     whisp,
        gen_opt,  disc_opt, whisp_opt,
        criterion,
        config.n_epochs,
        logs_dir=os.path.join(config.logs_dir, timestamp),
        ckpt_dir=os.path.join(config.ckpt_dir, timestamp))


def initialize_all():
    set_seed(3407)
    
    gen, gen_opt, disc, disc_opt = initialize_gan()  
    if config.train_with_wisper: whisp, whisp_opt = initialize_whisper()
    else: whisp, whisp_opt = None, None
    criterion = nn.BCELoss()

    set_seed(3407)
    train_dataset, train_dataloader, test_dataset, test_dataloader  = initialize_dataset()

    return criterion, \
        gen, disc, whisp, \
        gen_opt, disc_opt, whisp_opt, \
            train_dataset, train_dataloader, \
                test_dataset, test_dataloader

def run_experiment(config_dict):
    os.environ["RAY_memory_usage_threshold"] = '0.98'
    config.n_epochs = 10
    config.n_test = config.n_epochs + 1
    config.n_epochs_no_whisp = config.n_epochs + 1
    config.wandb_log = False
    config.ray_tune  = True
    config.save_logs = False

    config.lr_gen  = config_dict['lr_gen']
    config.lr_dis  = config_dict['lr_dis']
    config.penalty = config_dict['penalty']
    config.beta1   = config_dict['beta1']
    config.beta2   = config_dict['beta2']
    config.noise_size = config_dict['noise_size']

    criterion, gen, disc, whisp, gen_opt, disc_opt, whisp_opt, \
        train_dataset, train_dataloader, test_dataset, test_dataloader = initialize_all()

    bootstrap_iteration(
        gen,      disc,     whisp,
        gen_opt,  disc_opt, whisp_opt,
        criterion,
        train_dataloader, test_dataloader, 
        train_dataset, test_dataset)



if __name__ == "__main__":
    #---- Run normal training without tuning---

    criterion, gen, disc, whisp, gen_opt, disc_opt, whisp_opt, \
        train_dataset, train_dataloader, test_dataset, test_dataloader = initialize_all()

    for _ in range(config.bootstrap_iterations):
        bootstrap_iteration(
        gen,      disc,     whisp,
        gen_opt,  disc_opt, whisp_opt,
        criterion,
        train_dataloader, test_dataloader, 
        train_dataset, test_dataset)