# Audio_DFD_GAN
Audio deepfake detection robust to noise

This repository holds implementation for audio deepfake detection trained for robustness to noise. 
Before running the main script ([GODDS_GAN.py](https://github.com/nd0761/Audio_DFD_GAN/blob/main/GODDS_GAN.py)) make sure that weights_config folder holds all required weights and corresponding [repository ](https://github.com/icemoon97/deepfake-whisper) is cloned in the directory neighboring current repository.

To start training on the ASV dataset, ensure that the file_path set in asv_directory in ([GODDS_GAN.py](https://github.com/nd0761/Audio_DFD_GAN/blob/main/GODDS_GAN.py)) has the structure:
ASV
↳LA
  ↳LA
    ↳ASVspoof2019_***
↳PA
  ↳PA
    ↳ASVspoof2019_***

In the current repo version, LA train is used for training and LA dev for testing. 2500 bonafide and the same number of spoofed audio were chosen from both of them. To select a different number of elements, change the number corresponding to data_sample_size in ([GODDS_GAN.py](https://github.com/nd0761/Audio_DFD_GAN/blob/main/GODDS_GAN.py))

During training, checkpoints, as well as logs are collected and stored in the corresponding directories. As a quality metric, all measures described in ([metrics.py](https://github.com/nd0761/Audio_DFD_GAN/blob/main/utils_gan/training/metrics.py)) and distribution visualized collected for both discriminator and whisper models. 


