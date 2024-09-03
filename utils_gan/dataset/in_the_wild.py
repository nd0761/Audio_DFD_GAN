import os
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

import pickle
import pandas as pd
import numpy as np
from scipy.io import wavfile

import soundfile as sf

from torch.utils.data import Dataset
import torchaudio
import torch

import sys
sys.path.append(os.path.abspath('/tank/local/ndf3868/GODDS/GAN')) # IMPORTANT
import config

class ITW_DATSET(Dataset):
    def __init__(self, data_type):          # data_type either train or test
        self.bonafide_class = config.bonafide_class
        self.data_type = data_type

        self.audio_dir      = config.in_the_wild_dir
        self.test_ids_pkl   = config.in_the_wild_pkl
        self.meta_csv_path  = os.path.join(config.in_the_wild_dir, 'meta.csv')
        with open(self.test_ids_pkl, 'rb') as f:
            self.test_ids  = pickle.load(f)
        self.meta_csv       = pd.read_csv(self.meta_csv_path)

        self.get_files_for_type()
    
    def get_files_for_type(self):
        self.targets = pd.DataFrame(columns=['file_name', 'audio', 'is_original', 'generator_type', 'sample_rate'])

        def process_file(filepath):
            samplerate, data = wavfile.read(filepath)
            data = torch.from_numpy(data)
            data = preproc_audio(data)

            return data, samplerate

        def update_target(data):
            data_df = pd.DataFrame(data)
            if data_df is not None: self.targets = pd.concat([self.targets, data_df], ignore_index=True)
            for k in data: data[k] = []
            return data
        
        data = {
            'file_name': [],
            'audio': [],
            'is_original': [],
            'generator_type': [], # speaker
            'sample_rate': []
        }

        audio_filenames = []
        is_orig = 0
        for audio_file in os.listdir(self.audio_dir):
            if not audio_file.endswith('wav'): continue
            basename = audio_file.split('.')[0]
            if (float(basename) in self.test_ids      and self.data_type=='test') or \
               (float(basename) not in self.test_ids  and self.data_type=='train'):
                audio_filenames.append(audio_file)
                is_orig += int(self.meta_csv[self.meta_csv['file'] == audio_file]['label'].values[0] == 'bona-fide')

        for audio_file in tqdm(audio_filenames, desc='Processing InTheWild data'):
            audio, sr = process_file(os.path.join(self.audio_dir, audio_file)) 
            # audio data shape is [190_000] we make it into [1, 190_000]
            is_original = (self.bonafide_class if 
                            (self.meta_csv[self.meta_csv['file'] == audio_file]['label'].values[0] == 'bona-fide') 
                            else 1-self.bonafide_class)
            speaker = self.meta_csv[self.meta_csv['file'] == audio_file]['speaker'].values[0]

            data_new = {
                'file_name': [audio_file],
                'audio': [audio[None, :]],
                'is_original': [is_original],
                'generator_type': [speaker],
                'sample_rate': [sr]
            }
            for t in data_new: data[t].extend(data_new[t])
            if len(data['file_name']) % 2000 == 0: data = update_target(data)
        
        if len(data['file_name']) > 0:      update_target(data)
        print('Finished reading, number of audio in dataset', len(self.targets),
              '\nCount of original audio', len(self.targets[self.targets['is_original'] == self.bonafide_class]),
              '\nCount of spoofed  audio', len(self.targets[self.targets['is_original'] == 1-self.bonafide_class]))
        # pass
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        data = self.targets.iloc[idx]
        return data['audio'], data['sample_rate'], data['is_original'], data['generator_type']


def preproc_audio(data):
    data = apply_pad(data, config.input_size)
    return data

def apply_trim(waveform, sample_rate):
    (
        waveform_trimmed,
        sample_rate_trimmed,
    ) = torchaudio.sox_effects.apply_effects_tensor(waveform, sample_rate, config.SOX_SILENCE)

    if waveform_trimmed.size()[1] > 0:
        waveform = waveform_trimmed
        sample_rate = sample_rate_trimmed

    return waveform, sample_rate


def apply_pad(waveform, cut):
    """Pad wave by repeating signal until `cut` length is achieved."""
    if len(waveform.shape) > 1: waveform = waveform.squeeze(0)
    waveform_len = waveform.shape[0]

    if waveform_len >= cut:
        return waveform[:cut]

    # need to pad
    num_repeats = int(cut / waveform_len) + 1
    padded_waveform = torch.tile(waveform, (1, num_repeats))[:, :cut][0]

    return padded_waveform