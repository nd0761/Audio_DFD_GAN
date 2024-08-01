
import yaml

import sys 
import os
# sys.path.append(os.path.abspath('/tank/local/ndf3868/GODDS/deepfake-whisper/src'))
sys.path.append('/tank/local/ndf3868/GODDS/deepfake-whisper/src/models')
sys.path.append('/tank/local/ndf3868/GODDS/deepfake-whisper/src')
sys.path.append('/tank/local/ndf3868/GODDS/deepfake-whisper')

import models #/tank/local/ndf3868/GODDS/deepfake-whisper/src/models

import torch.nn as nn 
import torch

class Whispers(nn.Module):
    def __init__(self, whisper_models_directories, output_dim, device):
        self.device = device
        self.whisper_models = [self.load_model(t) for t in whisper_models_directories]

        self.process = nn.Sequential(
            nn.Linear(len(self.whisper_models), output_dim),
            nn.Sigmoid()
        )
    
    def load_model(self, model_directory):
        with open(f"{model_directory}/training/config.yaml", 'r') as f:
            model_config = yaml.safe_load(f)
        model = models.get_model(
            model_name=model_config["name"],
            config=model_config["parameters"],
            device=self.device,
        )
        model.load_state_dict(torch.load(model_config["checkpoint"]["path"]))
        model = model.to(self.device)
        model.eval()
        return model
    
    def forward(self, x):
        model_predictions = torch.FloatTensor([model(x) for model in self.whisper_models])
        print(model_predictions)
        return self.process(model_predictions)

