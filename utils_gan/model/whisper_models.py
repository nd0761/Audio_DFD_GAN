
import yaml

import sys 
import os
sys.path.append('/tank/local/ndf3868/GODDS/deepfake-whisper')

from src.models.models import get_model #/tank/local/ndf3868/GODDS/deepfake-whisper/src/models
from src.commons import set_seed

import torch.nn as nn 
import torch

class Whispers(nn.Module):
    def __init__(self, whisper_models_directories, output_dim, device):
        set_seed(3407)
        super(Whispers, self).__init__()

        self.bonafide_class = 1

        self.device = device
        self.whisper_models = []
        for t in whisper_models_directories:
            self.whisper_models.append(self.load_model(t))

        self.process = nn.Sequential(
            nn.Linear(len(self.whisper_models) + 1, output_dim),
            nn.Sigmoid()
        )
    
    def freeze_model(self, model):
        for param in model.parameters():
            param.requires_grad = False
        model.eval()
        return model
    
    def load_model(self, model_directory):
        with open(f"{model_directory}/config.yaml", 'r') as f:
            model_config = yaml.safe_load(f)
            model_config["checkpoint"]["path"] = os.path.join(model_directory, 'ckpt.pth')
        
        model = get_model(
            model_name=model_config["model"]["name"],
            config=model_config["model"]["parameters"],
            device=self.device,
        )
        model.load_state_dict(torch.load(model_config["checkpoint"]["path"]))

        model = model.to(self.device)
        model = self.freeze_model(model)
        
        return model

    def detectors_prediction(self, batch):

        def run_pred(model, batch):
            model.eval()
            return model(batch).squeeze(1)
    
        with torch.no_grad():

            batch_pred = [run_pred(model, batch) for model in self.whisper_models]
            batch_pred = [torch.sigmoid(t) for t in batch_pred]
            batch_pred_label = [(t + 0.5).int() for t in batch_pred]
            
        return torch.stack(batch_pred, dim=1), torch.stack(batch_pred_label, dim=1)
    
    def predict_on_data(self, data, disc_pred, dataset_bonafide_class):
        whisp_pred, _   = self.detectors_prediction(torch.squeeze(data,   1))

        if self.bonafide_class != dataset_bonafide_class: whisp_pred = torch.ones_like(whisp_pred) - whisp_pred
        # print(whisp_pred.shape, disc_pred.shape)
        whisp_with_disc = torch.cat((whisp_pred, disc_pred), dim=1)
        return whisp_with_disc
    
    def forward(self, x):
        return self.process(x)