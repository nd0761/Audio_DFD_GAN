import torch
import config

import sys
sys.path.insert(0, "/tank/local/ndf3868/GODDS/GAN/utils_gan/training/helpers")

from W_gan_helper import calculate_gradient_penalty

def cycle_dis(criterion,
              gen, disc, disc_opt,
              data):
    disc_opt.zero_grad()

    if config.train_with_wavegan: z = torch.randn(data.shape[0], config.noise_size).to(config.device)
    else: z = None

    if config.train_with_wavegan: noised_audio = gen(data, z)
    else: noised_audio = gen(data)

    real_disc_pred = disc(data)
    # errD_real = torch.mean(real_disc_pred)         #(should predict all as 1)
    errD_real = criterion(real_disc_pred, torch.ones_like(real_disc_pred))
    D_R = errD_real.item()

    nois_disc_pred = disc(noised_audio)
    # errD_noised = torch.mean(nois_disc_pred)     #(should predict all as 0)
    errD_noised = criterion(nois_disc_pred, torch.zeros_like(nois_disc_pred))
    D_N = errD_noised.item()

    grad_penalty = calculate_gradient_penalty(disc, data, noised_audio, config.device) #Wasserstein GAN

    errD = errD_noised + errD_real + grad_penalty * config.penalty
    errD.backward()
    disc_opt.step()
    return D_N, D_R

def cycle_gen(criterion, 
              gen, disc, gen_opt,
              data):
    gen_opt.zero_grad()

    if config.train_with_wavegan: z = torch.randn(data.shape[0], config.noise_size).to(config.device)
    else: z = None

    if config.train_with_wavegan: noised_audio = gen(data, z)
    else: noised_audio = gen(data)
    
    nois_disc_pred = disc(noised_audio)

    d_nois_loss = criterion(nois_disc_pred, torch.ones_like(nois_disc_pred)) # make disc predict as real(1)

    errG = d_nois_loss
    G_D = d_nois_loss.item()
    errG.backward()
    
    gen_opt.step()
    return G_D

def cycle_gen_whisp(criterion,
                    gen, disc, whisp,
                    gen_opt, data, 
                    reverse_label, bf_class):
    gen_opt.zero_grad()

    if config.train_with_wavegan: z = torch.randn(data.shape[0], config.noise_size).to(config.device)
    else: z = None

    if config.train_with_wavegan: noised_audio = gen(data, z)
    else: noised_audio = gen(data)
    
    nois_disc_pred = disc(noised_audio)

    nois_whisp_with_disc = whisp.predict_on_data(noised_audio, nois_disc_pred, bf_class)

    nois_whisp_pred = torch.squeeze(whisp(nois_whisp_with_disc), 1)

    w_nois_loss = criterion(nois_whisp_pred, reverse_label) # make whisp predict nosied data as reverse to what it should
    d_nois_loss = criterion(nois_disc_pred, torch.ones_like(nois_disc_pred)) # make disc predict as real(1)

    errG = d_nois_loss * 0.8 + w_nois_loss * 0.2
    G_D = d_nois_loss.item()
    G_W = w_nois_loss.item()
    errG.backward()
    gen_opt.step()
    return G_D, G_W

def cycle_whisp(criterion,
                gen, disc, whisp,
                whisp_opt,
                data, label, bf_class):
    whisp_opt.zero_grad()

    if config.train_with_wavegan: z = torch.randn(data.shape[0], config.noise_size).to(config.device)
    else: z = None

    if config.train_with_wavegan: noised_audio = gen(data, z)
    else: noised_audio = gen(data)
    
    real_disc_pred = disc(data)
    nois_disc_pred = disc(noised_audio)

    real_whisp_with_disc = whisp.predict_on_data(data,         real_disc_pred, bf_class)
    nois_whisp_with_disc = whisp.predict_on_data(noised_audio, nois_disc_pred, bf_class)

    real_whisp_pred = torch.squeeze(whisp(real_whisp_with_disc), 1)
    nois_whisp_pred = torch.squeeze(whisp(nois_whisp_with_disc), 1)

    nois_loss = criterion(nois_whisp_pred, label) # make whisp predict nosied data as real label(label)
    real_loss = criterion(real_whisp_pred, label) # make whisp predict real   data as real label(label)
    W_R = real_loss.item()
    W_N = nois_loss.item()

    whisD = nois_loss + real_loss
    whisD.backward()
    whisp_opt.step()
    return W_R, W_N

def full_cycle(cur_stage, criterion,
               gen, disc, whisp,
               gen_opt, disc_opt, whisp_opt,
               z, data, label, reverse_label,
               dataset_bonafide_class, 
               cur_step, cur_step_g):
    grad_d, grad_g = None, None
    ### --- UPDATE DISCRIMINATOR ---
    if cur_stage == 'd': #or cur_step_g % config.d_trainin_step == 0:
        disc_opt.zero_grad()

        if config.train_with_wavegan: z = torch.randn(data.shape[0], config.noise_size).to(config.device)
        else: z = None

        if config.train_with_wavegan: noised_audio = gen(data, z)
        else: noised_audio = gen(data)

        real_disc_pred = disc(data)
        # errD_real = torch.mean(real_disc_pred)         #(should predict all as 1)
        errD_real = criterion(real_disc_pred, torch.ones_like(real_disc_pred))
        D_R = errD_real.item()

        nois_disc_pred = disc(noised_audio)
        # errD_noised = torch.mean(nois_disc_pred)     #(should predict all as 0)
        errD_noised = criterion(nois_disc_pred, torch.zeros_like(nois_disc_pred))
        D_N = errD_noised.item()

        grad_penalty = calculate_gradient_penalty(disc, data, noised_audio, config.device) #Wasserstein GAN

        errD = errD_noised + errD_real + grad_penalty * config.penalty
        errD.backward()
        grad_d = errD.grad
        # .item()
        disc_opt.step()
    else:
        D_R = None
        D_N = None

    ### --- UPDATE WHISPER MODELS ---
    if cur_step is not None and cur_step % config.w_trainin_step == 0:
        whisp_opt.zero_grad()

        if config.train_with_wavegan: z = torch.randn(data.shape[0], config.noise_size).to(config.device)
        else: z = None

        if config.train_with_wavegan: noised_audio = gen(data, z)
        else: noised_audio = gen(data)
        
        real_disc_pred = disc(data)
        nois_disc_pred = disc(noised_audio)

        real_whisp_with_disc = whisp.predict_on_data(data,         real_disc_pred, dataset_bonafide_class)
        nois_whisp_with_disc = whisp.predict_on_data(noised_audio, nois_disc_pred, dataset_bonafide_class)

        real_whisp_pred = torch.squeeze(whisp(real_whisp_with_disc), 1)
        nois_whisp_pred = torch.squeeze(whisp(nois_whisp_with_disc), 1)

        nois_loss = criterion(nois_whisp_pred, label) # make whisp predict nosied data as real label(label)
        real_loss = criterion(real_whisp_pred, label) # make whisp predict real   data as real label(label)
        W_R = real_loss.item()
        W_N = nois_loss.item()

        whisD = nois_loss + real_loss
        whisD.backward()
        whisp_opt.step()
    else: 
        W_R = None
        W_N = None

    ### --- UPDATE GENERATOR ---
    if cur_step is not None and cur_step % config.w_trainin_step == 0:
        gen_opt.zero_grad()

        if config.train_with_wavegan: z = torch.randn(data.shape[0], config.noise_size).to(config.device)
        else: z = None

        if config.train_with_wavegan: noised_audio = gen(data, z)
        else: noised_audio = gen(data)
        
        nois_disc_pred = disc(noised_audio)

        nois_whisp_with_disc = whisp.predict_on_data(noised_audio, nois_disc_pred, dataset_bonafide_class)

        nois_whisp_pred = torch.squeeze(whisp(nois_whisp_with_disc), 1)

        w_nois_loss = criterion(nois_whisp_pred, reverse_label) # make whisp predict nosied data as reverse to what it should
        d_nois_loss = criterion(nois_disc_pred, torch.ones_like(nois_disc_pred)) # make disc predict as real(1)

        errG = d_nois_loss * 0.8 + w_nois_loss * 0.2
        G_D = d_nois_loss.item()
        G_W = w_nois_loss.item()
        errG.backward()
        # grad_g = errG.grad.item()
        gen_opt.step()
    else:
        if cur_stage == 'g': #or cur_step_g % config.g_trainin_step == 0:
            gen_opt.zero_grad()

            if config.train_with_wavegan: z = torch.randn(data.shape[0], config.noise_size).to(config.device)
            else: z = None

            if config.train_with_wavegan: noised_audio = gen(data, z)
            else: noised_audio = gen(data)
            
            nois_disc_pred = disc(noised_audio)

            d_nois_loss = criterion(nois_disc_pred, torch.ones_like(nois_disc_pred)) # make disc predict as real(1)

            errG = d_nois_loss
            G_D = d_nois_loss.item()
            G_W = None
            errG.backward()
            grad_g = errG.grad
            # .item()
            gen_opt.step()
        else:
            G_D = None
            G_W = None
    return D_R, D_N, W_R, W_N, G_D, G_W, grad_d, grad_g