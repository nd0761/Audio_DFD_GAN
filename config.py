import torch
import sys
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(color_scheme='Linux', call_pdb=False)

batch_size          = 32        # Batch size used in training

DEBUG               = False  # flag that would change a number of behaviours (purely cosmetic)

wandb_log           = True
ray_tune            = False
ray_tune_step       = 20
save_logs           = True

device              = 'cuda' if torch.cuda.is_available() else 'cpu'

in_the_wild_dir     = "/tank/local/hgi0312/data/release_in_the_wild"
in_the_wild_pkl     = "/tank/local/ndf3868/GODDS/GAN/checkpoints/datasets/inthewild_test_ids.pickle"

asv_directory = '/tank/local/ndf3868/GODDS/datasets/ASV'
    
bonafide_class      = 0     
    # Bonafide class in the chosen dataset, notice that 'whisper' models used for training instead use bonafide = 1, therefore, during training, their predictions are reversed

train_with_wisper   = True  # Toggle whether whisper models are used in trining or only GAN is trained
train_with_wavegan  = True  # WaveGan whould be processed with additional noise

data_sample_size    = 2500  # 100 - Debugging| 2500 - full training     Number of samples chosen from both training and test datasets
gen_fake            = False # True - Debug   | False - full training    Toggle used in debugging that creates random audio instead of reading ASV dataset


SOX_SILENCE = [
    # trim all silence that is longer than 0.2s and louder than 1% volume (relative to the file)
    # from beginning and middle/end
    ["silence", "1", "0.2", "1%", "-1", "0.2", "1%"],
]

# FRAMES_NUMBER       = 190_000
noise_size          = 128       # dimensionality of noise for wavegan

# input_size          = 190_000   # Dimensionality of audio 
input_size          = 131_072   # Dimensionality of audio 

hidden_dim_gen      = 6         # The depth of Generator model
hidden_dim_disc     = 2         # The depth of Discriminator model

wave_gen_initial_depth   = 5
wave_gen_final_depth     = 8
wave_disc_initial_depth  = 5
wave_disc_final_depth    = 8

output_size         = 1         # Output size, prediction of wether the discriminator predicts an audio as noised or non noised


lr                  = 1e-4    # Learning rate used in training
lr_gen              = lr / 5#0.009685600748667533 #0.000100778#0.000215011    #0.009685600748667533      # Learning rate used in training
lr_dis              = lr * 3 #0.00013827007668818942 #6.95911e-05#8.6315e-05 #0.00013827007668818942    # Learning rate used in training

penalty             = 1 #6 #9

beta1               = 0.6427777761161522    #beta 1 for adam optimizer
beta2               = 0.997289150566182     #beta 2 for adam optimizer

bootstrap_iterations= 1 #5      Number of iterations for bootstrapping 
n_epochs            = 70 if DEBUG == False else 1 #15    Number of epochs 
n_epochs_no_whisp   = int(n_epochs * 0.2)
n_test              = n_epochs+1# frequency to test model on test data

w_trainin_step      = 30        # every w_trainin_step we train whisper models after first n_epochs_no_whisp epochs
d_trainin_step      = 1         # first d_trainin_step we train discr then, every g_trainin_step we train generator
g_trainin_step      = 1

display_step        = 3         # frequency to update loss in tqdm visualization
log_au_spec         = 300       # frequency to store audio and spectrograms 


dataset_type = 'wild'           # supports ASV2019 'asv' and InTheWild 'wild'

logs_dir = '/tank/local/ndf3868/GODDS/GAN/logs'
ckpt_dir = '/tank/local/ndf3868/GODDS/GAN/checkpoints/models'
dapt_dir = '/tank/local/ndf3868/GODDS/GAN/checkpoints/datasets'

# ╭─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮                                              
# │ Trial name                status            lr_gen        lr_dis     penalty     g_trainin_step     iter     total time (s)       D_full        G_full     W_full      combined │                                              
# ├─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤                                              
# │ tune_params_21d11_00000   TERMINATED   4.99707e-05   3.1141e-05            9                 10        3           3832.03    0.0590888    0.0384954        10000   0.0467328   │                                              
# │ tune_params_21d11_00001   TERMINATED   0.000131656   2.9049e-05            9                 10        3           3861.46    0.0176164    0.0337878        10000   0.0273193   │                                              
# │ tune_params_21d11_00002   TERMINATED   7.83818e-05   3.38925e-05           9                 30        3           4452.49    0.0307415    0.00142737       10000   0.013153    │                                              
# │ tune_params_21d11_00004   TERMINATED   4.99174e-05   0.000247405           3                 10        1            973.817   0.0268532    0.26602          10000   0.170353    │                                              
# │ tune_params_21d11_00005   TERMINATED   0.000215011   8.6315e-05            6                 30        3           4283.86    0.0010826    0.00052389       10000   0.000747373 │ -                                             
# │ tune_params_21d11_00006   TERMINATED   0.00026351    4.07165e-05           9                 90        1            856.321   1.86307      0.000219297      10000   0.745358    │                                              
# │ tune_params_21d11_00007   TERMINATED   0.000123401   4.59635e-05           9                 90        1            952.409   1.19829      0.000307881      10000   0.479501    │                                              
# │ tune_params_21d11_00008   TERMINATED   2.79968e-05   2.57658e-05           3                 90        1            975.478   1.84937      0.00933307       10000   0.745348    │                                              
# │ tune_params_21d11_00009   TERMINATED   6.1485e-05    4.557e-05             0                 30        3           4505.48    0.00438518   0.0169027        10000   0.0118957   │                                              
# │ tune_params_21d11_00010   TERMINATED   3.11526e-05   0.000141467           6                 60        2           1952.26    0.0568788    0.00313337       10000   0.0246315   │                                              
# │ tune_params_21d11_00012   TERMINATED   4.06275e-05   0.000175429           9                 90        1            854.357   0.338431     0.0014827        10000   0.136262    │                                              
# │ tune_params_21d11_00013   TERMINATED   5.51226e-05   3.89419e-05           6                 60        2           1617.57    0.0583099    0.00397224       10000   0.0257073   │                                              
# │ tune_params_21d11_00014   TERMINATED   8.07805e-05   4.41329e-05           9                 60        3           3906.98    0.0341473    0.00287604       10000   0.0153845   │                                              
# │ tune_params_21d11_00015   TERMINATED   0.000161362   3.15663e-05           0                 90        1            829.32    0.569488     0.0011027        10000   0.228457    │                                              
# │ tune_params_21d11_00016   TERMINATED   2.6815e-05    3.29875e-05           3                 10        1           1006.21    0.313722     2.29378          10000   1.50176     │                                              
# │ tune_params_21d11_00017   TERMINATED   0.000288286   7.09521e-05           3                 60        3           3738.61    0.00399714   0.00249938       10000   0.00309848  │                                              
# │ tune_params_21d11_00018   TERMINATED   0.000231928   9.66825e-05           6                 90        2           1579.96    0.501839     0.00344472       10000   0.202802    │                                              
# │ tune_params_21d11_00019   TERMINATED   3.25012e-05   0.000199616           3                 90        1            745.468   0.29491      0.00192635       10000   0.11912     │                                              
# │ tune_params_21d11_00021   TERMINATED   0.000112245   0.000120192           0                 90        2           1557.08    0.131108     0.0178111        10000   0.06313     │                                              
# │ tune_params_21d11_00022   TERMINATED   3.15884e-05   0.000104961           3                 30        3           4063.22    0.0137009    0.0295461        10000   0.023208    │                                              
# │ tune_params_21d11_00024   TERMINATED   0.000100778   6.95911e-05           6                 30        3           3135.55    0.00401182   0.000854743      10000   0.00211757  │ - (with 50/50?)                                            
# │ tune_params_21d11_00003   ERROR        9.07155e-05   0.000146625           0                 10                                                                                 │                                              
# │ tune_params_21d11_00011   ERROR        0.000172042   0.00024715            0                 60                                                                                 │
# │ tune_params_21d11_00020   ERROR        5.67279e-05   0.000146612           0                 60                                                                                 │
# │ tune_params_21d11_00023   ERROR        0.000166568   0.000135491           9                 10        1            984.741   0.0364678    0.000133414      10000   0.0146672   │