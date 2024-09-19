from tqdm.auto import tqdm
import torch

import numpy as np

# import json
import sys 
import os
sys.path.append(os.path.abspath('/tank/local/ndf3868/GODDS/GAN/utils_gan')) # IMPORTANT

from training.loss import full_cycle
from training.metrics import set_up_metrics_list, test_metrics
# from training.distribution_visualizer import visualize_separate
from training.log_data import log_metrics, save_checkpoint, log_audio, log_spectrogram

import wandb

import config

# import tempfile
# from pathlib import Path

# from ray.train import Checkpoint, get_checkpoint
from ray import train as r_train
# import ray.cloudpickle as pickle

TRAINING_TYPE = 1 # 1 - noised vs non noised| 0 - generator vs bonafide

def train(train_with_wisper, 
            train_dataloader, test_dataloader, 
            train_dataset, test_dataset,
            gen,        disc,       whisp,
            gen_opt,    disc_opt,   whisp_opt, 
            criterion,
            n_epochs, 
            logs_dir, ckpt_dir):
    bonafide_class = train_dataset.bonafide_class
    # print(config.batch_size)
    # return None
    if config.wandb_log: 
        wandb_proj = wandb.init(
        # set the wandb project where this run will be logged
        project="AGAN",

        # track hyperparameters and run metadata
        config={
            "learning_rate_G": config.lr_gen,
            "learning_rate_D": config.lr_dis,
            "architecture": "WAVE_WES",
            "dataset": config.dataset_type,
            "epochs": config.n_epochs,
            "g_trainin_step": config.g_trainin_step,
            "w_trainin_step": config.w_trainin_step,
            "n_epochs_no_whisp": config.n_epochs_no_whisp,
            "penalty_amount": config.penalty,
        }
    )
    else: wandb_proj = None
    
    for epoch in range(n_epochs): #tqdm(range(n_epochs), desc='Training', leave):
        print('\n✧\n')
        mdl, mgl, mwl = train_epoch(train_with_wisper, 
            train_dataloader, bonafide_class,
            gen,        disc,       whisp,
            gen_opt,    disc_opt,   whisp_opt, 
            criterion,
            epoch, wandb_proj,
            test_dataloader, logs_dir)
        
        if epoch % 3 == 0 or epoch == n_epochs-1:
            predictions = produce_prediction_dict(train_with_wisper,
                                                test_dataloader, bonafide_class,
                                                gen, disc, whisp)
            metrics      = test_metrics(set_up_metrics_list(bonafide_class), predictions)

            if config.save_logs: log_metrics(predictions, metrics, 
                logs_dir, epoch)
            
        ckpt_path = os.path.join(ckpt_dir, f'last_ckpt.pt')
        _ = save_checkpoint(ckpt_dir, epoch,
            gen, disc, whisp,
            gen_opt, disc_opt, whisp_opt, ckpt_path)
        if epoch % 10 == 0 or epoch == n_epochs-1:
            ckpt_path = os.path.join(ckpt_dir, f'epoch{epoch}.pt')
            _ = save_checkpoint(ckpt_dir, epoch,
                gen, disc, whisp,
                gen_opt, disc_opt, whisp_opt, ckpt_path)
        
        if config.ray_tune:
            r_train.report(
                {"D_full": mdl, "G_full": mgl, "W_full":mwl, "combined": mdl * 0.4 + mgl * 0.6},
            )
            # --- OUT OF MEMORY WHEN SAVING CKPT ---
            # with tempfile.TemporaryDirectory() as checkpoint_dir:
            #     data_path = Path(checkpoint_dir) / "data.pkl"
            #     with open(data_path, "wb") as fp:
            #         pickle.dump(ckpt_dict, fp)

            #     checkpoint = Checkpoint.from_directory(checkpoint_dir)
            #     train.report(
            #         {"D_full": mdl, "G_full": mgl, "W_full":mwl, "combined": mdl * 0.4 + mgl * 0.6},
            #         checkpoint=checkpoint,
            #     )

        if config.save_logs: 
            orig_wav, nois_wav = log_audio(gen, test_dataloader, f'epoch_{epoch}', logs_dir)
            audio_files = [
                orig_wav,
                nois_wav
            ]
            spec_files = [
                os.path.join(logs_dir, 'spectrograms', f'epoch_{epoch}_orig.png'),
                os.path.join(logs_dir, 'spectrograms', f'epoch_{epoch}_nois.png')]
            for a_f, s_f in zip(audio_files, spec_files):
                log_spectrogram(a_f, s_f)
    if wandb_proj is not None: wandb_proj.finish()

def train_epoch(train_with_wisper, dataloader, bonafide_class,
                gen,        disc,       whisp,
                gen_opt,    disc_opt,   whisp_opt, 
                criterion,
                epoch, wandb_proj, 
                test_dataloader, logs_dir):
    
    # # Sample
    # original_audio = torch.randn(16, 1, 190_000)  # Batch size of 16, input audio
    # z = torch.randn(16, 100)  # Batch size of 16, noise vector of 100
    # generator = WaveGANGenerator()
    # fake_audio = generator(original_audio, z)  # Output shape will be [16, 1, 190_000]

    # real_audio = torch.randn(16, 1, 190_000)  # Batch size of 16, input dim [1, 190_000]
    # discriminator = WaveGANDiscriminator()
    # real_or_fake = discriminator(real_audio)  # Output shape will be [16, 1]
    
    gen.train()
    disc.train()
    if config.train_with_wisper: whisp.train()

    cur_step                = 0
    display_step            = 4
    mean_discriminator_loss = []
    mean_generator_loss     = []
    mean_whisp_loss         = []

    if config.save_logs: pbar = tqdm(dataloader, desc='description', dynamic_ncols=True, position=0)
    else: pbar = dataloader
    if config.save_logs: pbar_loss = tqdm(range(len(dataloader)), dynamic_ncols=True, position=1)
    else: pbar_loss = None
    t = 25

    for (data, sr, label, gen_type) in pbar:
        data = data.float()

        cur_batch_size = len(data)
        data           = data.to(config.device)
        label          = label.float().to(config.device)
        reverse_label  = torch.sub(torch.ones_like(label),label) # 1 initially REAL now 1 FAKE

        if config.train_with_wavegan: z = torch.randn(data.shape[0], config.noise_size).to(config.device)
        else: z = None

        t_cur = cur_step
        if cur_step != 0 and epoch <= config.n_epochs_no_whisp: t_cur = None

        D_R, D_N, W_R, W_N, G_D, G_W = full_cycle(criterion,
            gen, disc, whisp,
            gen_opt, disc_opt, whisp_opt,
            z, data, label, reverse_label,
            bonafide_class, cur_step=t_cur, cur_step_g=cur_step)

        mean_discriminator_loss.append((D_R + D_N))
        
        if t_cur is not None and t_cur % config.w_trainin_step == 0:
            mean_generator_loss.append((G_D * 0.8 + G_W * 0.2))
            mean_whisp_loss.append((W_N + W_R))
        else:
            if cur_step % config.g_trainin_step == 0:
                mean_generator_loss.append(G_D)

        if cur_step % display_step == 0 or cur_step == len(dataloader) -1:
            mdl = sum(mean_discriminator_loss)  / len(mean_discriminator_loss)
            mgl = sum(mean_generator_loss)      / len(mean_generator_loss)
            mwl = sum(mean_whisp_loss)          / len(mean_whisp_loss)
            
            pbar_desc = f"Epoch {epoch} ✧Step {cur_step}✧"
            
            pbar_desc_loss = f"Gen L: {mgl:.3f} || "+ \
                    f"Dis L: {mdl:.3f} || "+ \
                    f"Whi L: {mwl:.3f}"
            if config.save_logs: pbar.set_description(pbar_desc)
            if config.save_logs: pbar_loss.set_description(pbar_desc_loss)
            
            if wandb_proj is not None: 
                if t_cur is not None and t_cur % config.w_trainin_step == 0:
                    wandb_proj.log({
                    "Discriminator full": D_R + D_N, 
                    "Discriminator real": D_R,  
                    "Discriminator noised": D_N, 
                    "Whisper full": W_R + W_N, 
                    "Whisper real": W_R,  
                    "Whisper noised": W_N, 
                    "Generator full": G_D * 0.8 + G_W * 0.2, 
                    "Generator disc": G_D,  
                    "Generator whis": G_W,
                    })
                else:
                    wandb_proj.log({
                    "Discriminator full": D_R + D_N, 
                    "Discriminator real": D_R,  
                    "Discriminator noised": D_N,
                    "Generator full": G_D, 
                    "Generator disc": G_D,
                    })

        if config.save_logs and cur_step%500 == 0: 
            orig_wav, nois_wav = log_audio(gen, test_dataloader, cur_step, logs_dir)
            audio_files = [
                orig_wav,
                nois_wav
            ]
            spec_files = [
                os.path.join(logs_dir, 'spectrograms', f'orig.png'),
                os.path.join(logs_dir, 'spectrograms', f'{cur_step}_nois.png')]
            for a_f, s_f in zip(audio_files, spec_files):
                log_spectrogram(a_f, s_f)
        
        if config.save_logs: pbar_loss.update(1)
        cur_step += 1
        if config.DEBUG and t < 0: break
        t -= 1
    if config.save_logs: pbar_loss.close()
    return sum(mean_discriminator_loss)  / len(mean_discriminator_loss), \
        sum(mean_generator_loss)      / len(mean_generator_loss), \
        sum(mean_whisp_loss)          / len(mean_whisp_loss)

def produce_prediction_dict(train_with_wisper,
              dataloader, dataset_bonafide_class,
              gen, disc, whisp):
    gen.eval()
    disc.eval()
    if train_with_wisper: whisp.eval()

    total_batches = len(dataloader)
    limit_batches = int(total_batches * 0.15)


    if config.save_logs: pbar = tqdm(dataloader, desc='Evaluation in progress', dynamic_ncols=True, total=limit_batches)
    else: pbar = dataloader

    predictor_type  = ['disc', 'whisp']
    prediction_type = ['label', 'pred']
    data_type       = ['noised', 'non-noised']

    '''
    The predictor structure is:
        disc:
            label noised        TRAINING_TYPE = 0 -> this corresponds to label in dataloader | TRAINING_TYPE = 1 -> nosied    = 0
            pred  noised
            --
            label non-noised    TRAINING_TYPE = 0 -> this corresponds to label in dataloader | TRAINING_TYPE = 1 -> nonnosied = 1
            pred  non-noised
            --
            label all
            pred all
        whisp:
            label noised        TRAINING_TYPE always 0!
            pred  noised
            --
            label non-noised    TRAINING_TYPE always 0!
            pred  non-noised
            --
            label all
            pred all
    '''

    predictions = {pt:{f"{a} {b}": np.array([]) for a in prediction_type for b in data_type} for pt in predictor_type}
    t = 10
    for (data, sr, label, gen_type) in pbar:
        data = data.float()

        cur_batch_size = len(data)
        data           = data.to(config.device)
        label          = label.float().cpu().detach().numpy()

        if config.train_with_wavegan:
            z = torch.randn(data.shape[0], config.noise_size).to(config.device)
            noised = gen(data, z)
        else:
            noised = gen(data)

        nois_disc_pred  = disc(noised)
        real_disc_pred  = disc(data)

        real_whisp_with_disc = whisp.predict_on_data(data,   real_disc_pred, dataset_bonafide_class)
        nois_whisp_with_disc = whisp.predict_on_data(noised, nois_disc_pred, dataset_bonafide_class)

        real_whisp_pred = torch.squeeze(whisp(real_whisp_with_disc), 1).cpu().detach().numpy()
        nois_whisp_pred = torch.squeeze(whisp(nois_whisp_with_disc), 1).cpu().detach().numpy()
        real_disc_pred  = real_disc_pred.cpu().detach().numpy()
        nois_disc_pred  = nois_disc_pred.cpu().detach().numpy()
        # print(nois_disc_pred.shape)

        lab_no  = [label, np.zeros_like(label), label] # for type = 0 disc| for type = 1 disc | for whisp
        lab_non = [label, np.ones_like(label), label] # for type 1
        labs    = [lab_no, lab_non]

        pred_no  = [np.squeeze(nois_disc_pred), np.squeeze(nois_disc_pred), nois_whisp_pred]
        pred_non = [np.squeeze(real_disc_pred), np.squeeze(real_disc_pred), real_whisp_pred]
        preds = [pred_no, pred_non]
        
        for prt_id, prt in enumerate(predictor_type):
            prt_id *= 2
            if prt_id == 0: prt_id = TRAINING_TYPE
            for pnt, vals in zip(prediction_type, [labs, preds]):
                for dt, val in zip(data_type, vals):
                    predictions[prt][pnt+' '+dt] = val[prt_id]
        if config.DEBUG and t < 0: break
        t -= 1
        if limit_batches <= 0: break
        limit_batches -= 1
        # # break #DEBUG
        
    for prt in predictor_type:
        for pt in prediction_type:
            predictions[prt][f'{pt} all'] = np.concatenate((predictions[prt][f'{pt} non-noised'], predictions[prt][f'{pt} noised']), axis=0)
    # print(predictions)
    # assert 1==0
    return predictions