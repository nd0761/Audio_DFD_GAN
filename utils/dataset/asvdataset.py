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
        # self.read_data()
        print("Finished reading")

        # if category == 'train':
        if class_balance == 'undersample_all':
            self.undersample_all(class_amount)
        if class_balance == 'undersample':
            self.undersample()
        if class_balance == 'oversample':
            self.oversample()

        self.set_up_indexes()

    def read_audio_file(self, file_name):
        path = os.path.join(self.files_loc, file_name+'.flac')

        if not os.path.exists(path): return None, None

        data, samplerate = sf.read(path)  
        data = torch.from_numpy(data)
        data = self.preproc_audio(data)

        return data, samplerate
    
    def read_target(self):
        self.targets = pd.DataFrame(columns=['file_name', 'audio', 'is_original', 'generator_type', 'sample_rate'])
        updated_filenames = set()

        with open(self.targets_loc, 'r') as f:
            lines = f.readlines()
            for line in tqdm(lines, desc='Reading target'):
                split_line  = line.split(' ')

                file_name   = split_line[1]
                if file_name in updated_filenames: continue
                updated_filenames.add(file_name)

                is_original = (self.bonafide_class if (split_line[-1].rstrip() == 'bonafide') else 1-self.bonafide_class)
                gen_type    = ' '.join([split_line[-2], split_line[-3]])

                audio, sr = self.read_audio_file(file_name)
                if audio is None: continue

                data = {
                    'file_name': [file_name],
                    'audio': [audio[None, :]],
                    'is_original': [is_original],
                    'generator_type': [gen_type],
                    'sample_rate': [sr]
                }
                data_df = pd.DataFrame(data)
                self.targets = pd.concat([self.targets, data_df])
        self.targets = self.targets.reset_index()
        print("Size of dataset",len(self.targets))

    def preproc_audio(self, data):
        data = apply_pad(data, FRAMES_NUMBER)
        return data

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
    
    def set_up_indexes(self):
        self.class_indeces = {  self.bonafide_class: self.targets.index[self.targets['is_original'] ==   self.bonafide_class].to_numpy(),
                              1-self.bonafide_class: self.targets.index[self.targets['is_original'] == 1-self.bonafide_class].to_numpy()}

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