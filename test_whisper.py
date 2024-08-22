import numpy as np

import random
import json
import yaml

import torch
from torch import nn
from torch.utils.data import DataLoader, SubsetRandomSampler

import datetime
import sys 
import os
sys.path.append(os.path.abspath('/tank/local/ndf3868/GODDS/GAN'))
sys.path.append('/tank/local/ndf3868/GODDS/deepfake-whisper')

from src.models.models import get_model #/tank/local/ndf3868/GODDS/deepfake-whisper/src/models
from src.commons import set_seed


import torch.nn as nn 
import torch

from tqdm.auto import tqdm

# file_path = os.path.realpath(__file__)

import utils_gan

from utils_gan import ASV_DATASET, Generator, Discriminator, Whispers, train
from utils_gan import set_up_metrics_list, test_metrics
# from utils_gan import  train, test_metrics, set_up_metrics_list, \
#     visualize, visualize_separate


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

def load_model(model_directory, device):
    with open(f"{model_directory}/config.yaml", 'r') as f:
        model_config = yaml.safe_load(f)
        model_config["checkpoint"]["path"] = os.path.join(model_directory, 'ckpt.pth')
    
    model = get_model(
        model_name=model_config["model"]["name"],
        config=model_config["model"]["parameters"],
        device=device,
    )
    model.load_state_dict(torch.load(model_config["checkpoint"]["path"]))

    model = model.to(device)
    return model

def test_model(gen, whisp, dataloader, bonafide_class=0, whisp_bonafide=1):

    predictor_type  = ['whisp']
    prediction_type = ['label', 'pred']
    data_type       = ['noised', 'non-noised']
    predictions = {pt:{f"{a} {b}": np.array([]) for a in prediction_type for b in data_type} for pt in predictor_type}
    
    pbar = tqdm(dataloader, desc='Evaluation in progress')
    for (data, sr, label, gen_type) in pbar:
        data = data.float()

        cur_batch_size = len(data)
        data           = data.to(device)
        label          = label.float().cpu().detach().numpy()

        noised = gen(data)

        noised_data = torch.squeeze(noised,   1)
        real_data = torch.squeeze(data,   1)

        nois_pred = torch.sigmoid(whisp(noised_data).squeeze(1))
        real_pred = torch.sigmoid(whisp(real_data).squeeze(1))

        if bonafide_class != whisp_bonafide: 
            nois_pred = torch.ones_like(nois_pred) - nois_pred
            real_pred = torch.ones_like(real_pred) - real_pred
        
        nois_pred = nois_pred.cpu().detach().numpy()
        real_pred = real_pred.cpu().detach().numpy()

        lab_no  = [label] # for type = 0 disc| for type = 1 disc | for whisp
        lab_non = [label] # for type 1
        labs = [lab_no, lab_non]

        
        pred_no  = [nois_pred]
        pred_non = [real_pred]
        preds = [pred_no, pred_non]
        
        for prt_id, prt in enumerate(predictor_type):
            for pnt, vals in zip(prediction_type, [labs, preds]):
                for dt, val in zip(data_type, vals):
                    predictions[prt][pnt+' '+dt] = val[prt_id]
    for prt in predictor_type:
        for pt in prediction_type:
            predictions[prt][f'{pt} all'] = np.concatenate((predictions[prt][f'{pt} non-noised'], predictions[prt][f'{pt} noised']), axis=0)
    
    metrics_list = set_up_metrics_list(bonafide_class)
    metrics      = test_metrics(metrics_list, predictions)
    return metrics

# set_seed(3407)
device = 'cuda:2' if torch.cuda.is_available() else 'cpu'

if __name__ == "__main__":
    set_seed(3407)
    bonafide_class      = 0

    train_with_wisper   = True

    data_sample_size    = 2500  # 100 - Debugging| 2500 - full training
    gen_fake            = False # True - Debug   | False - full training

    input_size          = 190_000
    hidden_dim_gen      = 2
    hidden_dim_disc     = 2
    output_size         = 1

    batch_size          = 8

    lr                  = 1e-4

    bootstrap_iterations= 1 #5
    # n_epochs            = 50 #15


    # Generator & Optimizer for Generator
    gen = Generator(input_size, hidden_dim=hidden_dim_gen).to(device)
    gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)

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

    epoch_pt_path = '/tank/local/ndf3868/GODDS/GAN/checkpoints/models/180824-07:12:19/epoch49.pt'

    checkpoint = torch.load(epoch_pt_path)

    gen.load_state_dict(checkpoint['gen_state_dict'])
    gen.eval()
    set_seed(3407)

    asv_directory = '/tank/local/ndf3868/GODDS/datasets/ASV'
    print("reading TEST  dataset")
    test_dataset    = ASV_DATASET(asv_directory, 'dev',   'LA', class_balance=None, gen_fake=gen_fake)
    print("sampling TEST dataset")
    sampler         = SubsetRandomSampler(get_balanced_indeces(dataset=test_dataset, n_samples_per_class=data_sample_size, shuffle=False))
    test_dataloader = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, sampler=sampler)

    
    logs_dir = '/tank/local/ndf3868/GODDS/GAN/logs'
    timestamp = f'{get_current_timestamp()}'
    timestamp = 'TESTING_WHISPER_'+timestamp
    if gen_fake: timestamp = 'DEBUG_'+timestamp

    os.makedirs(os.path.join(logs_dir, timestamp))
    logs_dir = os.path.join(logs_dir, timestamp)

    full_result = {}

    for t in tqdm(whisper_models_direstories, desc='testing whisper'):
        mod_name = t.split('/')[-1]
        whisp_model = load_model(t, device)
        whisp_model.eval()

        set_seed(3407)
        metrics = test_model(gen, whisp_model, test_dataloader)
        full_result[mod_name] = metrics
    with open(os.path.join(logs_dir, 'ALL_PREDICTIONS.json'), 'w') as f:
        json.dump(full_result, f)

