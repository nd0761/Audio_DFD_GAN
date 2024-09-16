
import torch
wandb_log           = False
device              = 'cuda:2' if torch.cuda.is_available() else 'cpu'

in_the_wild_dir     = "/tank/local/hgi0312/data/release_in_the_wild"
in_the_wild_pkl     = "/tank/local/ndf3868/GODDS/GAN/checkpoints/datasets/inthewild_test_ids.pickle"

asv_directory = '/tank/local/ndf3868/GODDS/datasets/ASV'
    
bonafide_class      = 0     
    # Bonafide class in the chosen dataset, notice that 'whisper' models used for training instead use bonafide = 1, therefore, during training, their predictions are reversed

train_with_wisper   = True  # Toggle whether whisper models are used in trining or only GAN is trained
train_with_wavegan  = True  # WaveGan whould be processed with additional noise

data_sample_size    = 2500  # 100 - Debugging| 2500 - full training     Number of samples chosen from both training and test datasets
gen_fake            = False # True - Debug   | False - full training    Toggle used in debugging that creates random audio instead of reading ASV dataset

DEBUG               = False  # flag that would change a number of behaviours (purely cosmetic)

SOX_SILENCE = [
    # trim all silence that is longer than 0.2s and louder than 1% volume (relative to the file)
    # from beginning and middle/end
    ["silence", "1", "0.2", "1%", "-1", "0.2", "1%"],
]

# FRAMES_NUMBER       = 190_000
noise_size          = 100       # dimensionality of noise for wavegan

input_size          = 190_000   # Dimensionality of audio 

hidden_dim_gen      = 6         # The depth of Generator model
hidden_dim_disc     = 2         # The depth of Discriminator model

wave_gen_initial_depth  = 5
wave_gen_final_depth    = 8
wave_disc_initial_depth  = 5
wave_disc_final_depth    = 8

output_size         = 1         # Output size, prediction of wether the discriminator predicts an audio as noised or non noised

batch_size          = 8         # Batch size used in training

lr                  = 1e-4*2    # Learning rate used in training
lr_gen              = lr*2    # Learning rate used in training
lr_dis              = lr/1.5    # Learning rate used in training

penalty             = 10

beta1               = 0.5

bootstrap_iterations= 1 #5      Number of iterations for bootstrapping 
n_epochs            = 5 if DEBUG == False else 1 #15    Number of epochs 


dataset_type = 'wild'           # supports ASV2019 'asv' and InTheWild 'wild'

logs_dir = '/tank/local/ndf3868/GODDS/GAN/logs'
ckpt_dir = '/tank/local/ndf3868/GODDS/GAN/checkpoints/models'
dapt_dir = '/tank/local/ndf3868/GODDS/GAN/checkpoints/datasets'