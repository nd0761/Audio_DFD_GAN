import os
from tqdm.auto import tqdm

import pandas as pd

import soundfile as sf

from torch.utils.data import Dataset
import torchaudio
import torch

SOX_SILENCE = [
    # trim all silence that is longer than 0.2s and louder than 1% volume (relative to the file)
    # from beginning and middle/end
    ["silence", "1", "0.2", "1%", "-1", "0.2", "1%"],
]
FRAMES_NUMBER = 190_000  # <- originally 64_600

# bonafide     - label 0 (original)
# not bonafide - label 1 (generated)

subfolder_prefix = "ASVspoof2019_""_cm_protocols"

class ASV_DATASET(Dataset):
    def __init__(self, asv_directory, category='train', type='LA', class_balance=None, class_amount=500): # type LA or PA | category eval dev train
        self.bonafide_class = 0
        self.prefix = os.path.join(type, type, f"ASVspoof2019_{type}_cm_protocols")

        suffix = 'trl'
        if category == 'train': suffix = 'trn'

        self.targets_loc = os.path.join(asv_directory, self.prefix, f"ASVspoof2019.{type}.cm.{category}.{suffix}.txt")
        self.files_loc   = os.path.join(asv_directory, type, type,  f"ASVspoof2019_{type}_{category}", 'flac')

        self.read_target()
        self.read_data()
        print("Finished reading")

        # if category == 'train':
        if class_balance == 'undersample_all':
            self.undersample_all(class_amount)
        if class_balance == 'undersample':
            self.undersample()
        if class_balance == 'oversample':
            self.oversample()
    
    def read_target(self):
        self.targets = {}
        self.indexing = {}

        last_idx = 0

        # g, b = 0, 0

        with open(self.targets_loc, 'r') as f:
            lines = f.readlines()
            for line in tqdm(lines, desc='Reading target'):
                split_line  = line.split(' ')

                file_name   = split_line[1]
                is_original = (self.bonafide_class if (split_line[-1].rstrip() == 'bonafide') else 1-self.bonafide_class)
                # g += (is_original == self.bonafide_class)
                # b += (1 - is_original)
                gen_type    = ' '.join([split_line[-2], split_line[-3]])
                
                if file_name not in self.targets:
                    self.targets[file_name] = {'audio': None, 'sample_rate':None, 'is_original': is_original, 'generator_type': gen_type}
                    self.indexing[last_idx] = file_name
                    last_idx += 1
                else:
                    self.targets[file_name] = {'audio': None, 'sample_rate':None, 'is_original': is_original, 'generator_type': gen_type}

    def preproc_audio(self, data):
        data = apply_pad(data, FRAMES_NUMBER)
        return data

    def read_data(self):
        bad_countet = 0
        max_len = 0
        for file in tqdm(os.listdir(self.files_loc), desc='Reading audio'):
            if not file.endswith('flac'): continue
            file_name = file.split('.')[0]
            path = os.path.join(self.files_loc, file)
            
            if file_name not in self.targets: 
                bad_countet += 1
                continue
            data, samplerate = sf.read(path)  
            data = torch.from_numpy(data)
            data = self.preproc_audio(data)
            # if bad_countet == 0:
            #     print('AUDIO SHAPE', data.shape)
            #     bad_countet += 1
            self.targets[file_name]['audio']        = data[None, :]
            self.targets[file_name]['sample_rate']  = samplerate
            max_len = max(max_len, data.shape[0])
        # print("CNT of Files without a label", bad_countet)
        # print("MAX len", max_len)
        data = {
            'file_name': [],
            'audio': [],
            'is_original': [],
            'generator_type': [],
            'sample_rate': []
        }
        
        for file_name, content in self.targets.items():
            data['file_name'].append(file_name)
            data['audio'].append(content['audio'])
            data['is_original'].append(content['is_original'])
            data['generator_type'].append(content['generator_type'])
            data['sample_rate'].append(content['sample_rate'])
        
        self.targets = pd.DataFrame(data)
        self.targets.reset_index()
    
    def undersample_all(self, class_amount):
        minority_class = self.targets[self.targets['is_original'] != self.targets['is_original'].mode()[0]]
        majority_class = self.targets[self.targets['is_original'] == self.targets['is_original'].mode()[0]]

        print('UNDERSAMPLING from majority', len(majority_class), "and", len(minority_class), "to", class_amount)

        majority_class = majority_class.sample(class_amount)
        minority_class = minority_class.sample(class_amount)

        self.targets = pd.concat([minority_class, majority_class])
        self.targets.reset_index()
        print('\n------\nBONAFIDE  samples', len(self.targets[self.targets['is_original'] == self.bonafide_class]), 
              '\nGENERATED samples', len(self.targets[self.targets['is_original'] != self.bonafide_class]))
    
    def undersample(self):
        minority_class = self.targets[self.targets['is_original'] != self.targets['is_original'].mode()[0]]
        majority_class = self.targets[self.targets['is_original'] == self.targets['is_original'].mode()[0]]

        print('UNDERSAMPLING from majority', len(majority_class), "to", len(minority_class))

        majority_class = majority_class.sample(len(minority_class))
        self.targets = pd.concat([minority_class, majority_class])
        self.targets.reset_index()
        print('BONAFIDE samples', len(self.targets[self.targets['is_original'] == self.bonafide_class]), 
              'GENERATED samples', len(self.targets[self.targets['is_original'] != self.bonafide_class]))
    
    def oversample(self):
        minority_class = self.targets[self.targets['is_original'] != self.targets['is_original'].mode()[0]]
        majority_class = self.targets[self.targets['is_original'] == self.targets['is_original'].mode()[0]]

        print('OVERSAMPLING from miority', len(minority_class), "to", len(majority_class))
        num_samples_needed = len(majority_class) - len(minority_class)
    
        # If no oversampling is needed, return the original DataFrame
        if num_samples_needed >= 0:
            # Iteratively append minority class samples to avoid memory issues
            minority_oversampled = minority_class.sample(n=num_samples_needed, replace=True)
            self.targets = pd.concat([majority_class, minority_class, minority_oversampled])
        # minority_class = minority_class.sample(len(majority_class), replace=True)
        # self.targets = pd.concat([minority_class, majority_class])
        self.targets.reset_index()
        print('\n------\nBONAFIDE  samples', len(self.targets[self.targets['is_original'] == self.bonafide_class]), 
              '\nGENERATED samples', len(self.targets[self.targets['is_original'] != self.bonafide_class]))
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        data = self.targets.iloc[idx]
        return data['audio'], data['sample_rate'], data['is_original'], data['generator_type']



def apply_trim(waveform, sample_rate):
    (
        waveform_trimmed,
        sample_rate_trimmed,
    ) = torchaudio.sox_effects.apply_effects_tensor(waveform, sample_rate, SOX_SILENCE)

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