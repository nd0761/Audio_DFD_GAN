from tqdm.auto import tqdm
import torch

import numpy as np

# import json
import sys 
import os
sys.path.append(os.path.abspath('/tank/local/ndf3868/GODDS/GAN/utils_gan')) # IMPORTANT

from training.loss import full_cycle, cycle_dis, cycle_gen, cycle_gen_whisp, cycle_whisp
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
    if config.save_logs:
        log_audio_data, log_audio_sr, _, _ = next(iter(test_dataloader))
        if config.train_with_wavegan: log_audio_noise = torch.randn(1, config.noise_size).to(config.device)
        else: log_audio_noise=None
    bonafide_class = train_dataset.bonafide_class
    if config.wandb_log: 
        wandb_proj = wandb.init(
        project="AGAN",
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
    
    for epoch in range(n_epochs):
        print('\n✧\n')

        mdl, mgl, mwl = train_epoch(train_dataloader, bonafide_class,
            gen,        disc,       whisp,
            gen_opt,    disc_opt,   whisp_opt, 
            criterion,
            epoch, wandb_proj,
            test_dataloader, logs_dir)
        
        if epoch != 0 and epoch % config.n_test == 0 or epoch == n_epochs-1:
            predictions = produce_prediction_dict(train_with_wisper,
                                                test_dataloader, bonafide_class,
                                                gen, disc, whisp)
            metrics      = test_metrics(set_up_metrics_list(bonafide_class), predictions)

            if config.save_logs: log_metrics(predictions, metrics, 
                logs_dir, epoch)
            
        ckpt_path = os.path.join(ckpt_dir, f'last_ckpt.pt')
        save_checkpoint(ckpt_dir, epoch,
            gen, disc, whisp,
            gen_opt, disc_opt, whisp_opt, ckpt_path)
        if epoch % 10 == 0 or epoch == n_epochs-1:
            ckpt_path = os.path.join(ckpt_dir, f'epoch{epoch}.pt')
            save_checkpoint(ckpt_dir, epoch,
                gen, disc, whisp,
                gen_opt, disc_opt, whisp_opt, ckpt_path)
        
        if config.ray_tune:
            r_train.report(
                {"D_full": mdl, "G_full": mgl, "W_full":mwl, "combined": mdl * 0.4 + mgl * 0.6},
            )

        if config.save_logs: 
            orig_wav, nois_wav = log_audio(gen, log_audio_data, log_audio_sr, 
                                           f'epoch_{epoch}', logs_dir, log_audio_noise)
            audio_files = [orig_wav, nois_wav]
            spec_files = [
                os.path.join(logs_dir, 'spectrograms', f'orig.png'),
                os.path.join(logs_dir, 'spectrograms', f'epoch_{epoch}_nois.png')]
            for a_f, s_f in zip(audio_files, spec_files): log_spectrogram(a_f, s_f)
    if wandb_proj is not None: wandb_proj.finish()


def train_epoch(dataloader, bonafide_class,
                gen,        disc,       whisp,
                gen_opt,    disc_opt,   whisp_opt, 
                criterion,
                epoch, wandb_proj, 
                test_dataloader, logs_dir):

    if config.save_logs: # Generate new data using same noise + audio
        log_audio_data, log_audio_sr, _, _ = next(iter(test_dataloader))
        if config.train_with_wavegan: log_audio_noise = torch.randn(1, config.noise_size).to(config.device)
        else: log_audio_noise=None
    
    gen.train()
    disc.train()
    if config.train_with_wisper: whisp.train()

    cur_step                = 0
    mean_discriminator_loss, mean_generator_loss, mean_whisp_loss = [], [], []

    if config.save_logs: pbar = tqdm(dataloader, desc='description', dynamic_ncols=True, position=0)
    else: pbar = dataloader
    if config.save_logs: pbar_loss = tqdm(range(len(dataloader)), dynamic_ncols=True, position=1)
    else: pbar_loss = None
    t = 25 # iterations to stop after in case of DEBUG

    cur_stage = 'd' # currently updating stages of disciriminator 'd' or generator 'g'
    stage_counter = config.d_trainin_step # updating for config.*_trainin_step's

    mdl, mgl, mwl = np.nan, np.nan, np.nan

    for (data, sr, label, gen_type) in pbar:
        data = data.float()

        data           = data.to(config.device)
        label          = label.float().to(config.device)
        reverse_label  = torch.sub(torch.ones_like(label),label)

        t_cur = cur_step
        if cur_step != 0 and epoch <= config.n_epochs_no_whisp: t_cur = None

        wandb_data = {}

        while stage_counter <= 0:
            if cur_stage == 'd': 
                cur_stage = 'g'
                stage_counter = config.g_trainin_step
            else: 
                cur_stage = 'd'
                stage_counter = config.d_trainin_step
        
        if cur_stage == 'd':
            D_N, D_R = cycle_dis(criterion, gen, disc, disc_opt, data)
            wandb_data["Discriminator full"] = D_N + D_R
            wandb_data["Discriminator noised"] = D_N
            wandb_data["Discriminator real"] = D_R
            mean_discriminator_loss.append((D_N + D_R))
        # if cur_step % config.w_trainin_step == 0: 
        #     W_R, W_N = cycle_whisp(criterion,
        #         gen, disc, whisp,
        #         whisp_opt,
        #         data, label, bonafide_class)
        #     wandb_data["Whisper full"] = W_N + W_R
        #     wandb_data["Whisper noised"] = W_N
        #     wandb_data["Whisper real"] = W_R
        #     mean_whisp_loss.append((W_R + W_N))
        if cur_stage == 'g':
            # if cur_step % config.w_trainin_step == 0: 
            #     G_D, G_W = cycle_gen_whisp(criterion,
            #         gen, disc, whisp,
            #         gen_opt, data, 
            #         reverse_label, bonafide_class)
            #     wandb_data["Generator full"] = G_D * 0.8 + G_W * 0.2
            #     wandb_data["Generator disc"] = G_D
            #     wandb_data["Generator whis"] = G_W
            #     mean_generator_loss.append((G_D * 0.8 + G_W * 0.2))
            G_D = cycle_gen(criterion, gen, disc, gen_opt, data)
            wandb_data["Generator full"] = G_D
            wandb_data["Generator disc"] = G_D
            mean_generator_loss.append(G_D)
        stage_counter -= 1

        if cur_step % config.display_step == 0 or cur_step == len(dataloader) -1:
            if len(mean_discriminator_loss) > 0:
                mean_discriminator_loss = mean_discriminator_loss[-config.display_step*2:]
                mdl = sum(mean_discriminator_loss)  / len(mean_discriminator_loss)
            else: mdl = np.nan
            if len(mean_generator_loss) > 0:
                mean_generator_loss = mean_generator_loss[-config.display_step*2:]
                mgl = sum(mean_generator_loss)      / len(mean_generator_loss)
            else: mgl = np.nan
            if len(mean_whisp_loss) > 0:
                mean_whisp_loss = mean_whisp_loss[-config.display_step*2:]
                mwl = sum(mean_whisp_loss)          / len(mean_whisp_loss)
            else: mwl = np.nan
            grad_g = gen.conv[0].transpose_ops[1].weight.grad.clone().mean()
            grad_d = disc.conv[0].weight.grad.clone().mean()
            pbar_desc = f"Epoch {epoch} ✧Step {cur_step}✧ Updating:{cur_stage} Gen grad {grad_g:.6e} Dis grad {grad_d:.6e}"
            
            pbar_desc_loss = f"Gen L: {mgl:.3f} || "+ \
                    f"Dis L: {mdl:.3f} || "+ \
                    f"Whi L: {mwl:.3f}"
            if ((mgl >= 20 or mdl >= 20) and (mgl != np.nan and mdl != np.nan)):# or (cur_step != 0 and (grad_g < 1e-9 or grad_d < 1e-9)):
                print("GAN is unstable!!! Failed epoch", epoch, "on step", cur_step)
                print(grad_g < 1e-7, grad_d < 1e-7)
                print('Gradients: Gen', f'{grad_g:7.6e}  Dis {grad_d:7.6f}')
                print('Loss:      Gen',      f'{mgl:7.6e}  Dis {mdl:7.6f}')
                raise ValueError()
            if config.save_logs: pbar.set_description(pbar_desc)
            if config.save_logs: pbar_loss.set_description(pbar_desc_loss)
            
            if wandb_proj is not None: 
                if t_cur is not None and t_cur % config.w_trainin_step == 0:
                    # grad_g = gen.conv[0].transpose_ops[1].weight.grad.clone().mean()
                    # grad_d = disc.conv[0].weight.grad.clone().mean()
                    wandb_data['Generator grad']     = grad_g
                    wandb_data['Discriminator grad'] = grad_d
                    wandb_proj.log(wandb_data)

        if config.save_logs and cur_step%config.log_au_spec == 0: 
            orig_wav, nois_wav = log_audio(gen, log_audio_data, log_audio_sr, cur_step, logs_dir, log_audio_noise)
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
    return mdl, mgl, mwl

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