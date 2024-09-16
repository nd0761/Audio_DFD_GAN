import torch
import config

import sys
sys.path.insert(0, "/tank/local/ndf3868/GODDS/GAN/utils_gan/training/helpers")

from W_gan_helper import calculate_gradient_penalty

def full_cycle(criterion,
               gen, disc, whisp,
               gen_opt, disc_opt, whisp_opt,
               z, data, label, reverse_label,
               dataset_bonafide_class):
    ### --- UPDATE DISCRIMINATOR ---
    disc_opt.zero_grad()

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

    ### --- UPDATE WHISPER MODELS ---
    whisp_opt.zero_grad()

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

    ### --- UPDATE GENERATOR ---
    gen_opt.zero_grad()

    if config.train_with_wavegan: noised_audio = gen(data, z)
    else: noised_audio = gen(data)
    
    nois_disc_pred = disc(noised_audio)

    nois_whisp_with_disc = whisp.predict_on_data(noised_audio, nois_disc_pred, dataset_bonafide_class)

    nois_whisp_pred = torch.squeeze(whisp(nois_whisp_with_disc), 1)

    w_nois_loss = criterion(nois_whisp_pred, reverse_label) # make whisp predict nosied data as real label(label)
    d_nois_loss = criterion(nois_disc_pred, torch.ones_like(nois_disc_pred))

    errG = d_nois_loss * 0.8 + w_nois_loss * 0.2
    G_D = d_nois_loss.item()
    G_W = w_nois_loss.item()
    errG.backward()
    gen_opt.step()
    return D_R, D_N, W_R, W_N, G_D, G_W

def generator_loss(criterion,
               gen, disc, whisp,
               data, dataset_bonafide_class,
               label, reverse_label, 
               type=0, training_with_whisp=False): # type = 1 - noised vs non noised  |   0 - fake vs real
    if config.train_with_wavegan:
        z = torch.randn(data.shape[0], config.noise_size).to(config.device)
        noised = gen(data, z)
    else:
        noised = gen(data)
    
    nois_disc_pred  = disc(noised)
    # disc_real_pred    = disc(data)

    if not training_with_whisp:
        gen_loss            = criterion(nois_disc_pred, torch.ones_like(nois_disc_pred)) # make disc predict data as non-noised(1)
    elif training_with_whisp:
        nois_whisp_with_disc= whisp.predict_on_data(noised, nois_disc_pred, dataset_bonafide_class)
        nois_whisp_pred     = torch.squeeze(whisp(nois_whisp_with_disc), 1)

        gen_loss   = criterion(nois_whisp_pred, reverse_label) # make whisp+disc predict data as fake when it's not and vice versa
        gen_loss_2 = criterion(nois_disc_pred,  torch.ones_like(nois_disc_pred)) # make disc predict data as non-noised(1) when it's not
        gen_loss = gen_loss * 0.2 + gen_loss_2 * 0.8
    
    return gen_loss

def discriminator_loss(criterion, 
                       gen, disc,
                       data, 
                       label, reverse_label, type=1): # type = 1 - noised vs non noised  |   0 - fake vs real
    if config.train_with_wavegan:
        z = torch.randn(data.shape[0], config.noise_size).to(config.device)
        noised = gen(data, z)
    else:
        noised = gen(data)

    nois_disc_pred = disc(noised)
    real_disc_pred = disc(data)

    fake_loss = criterion(nois_disc_pred, torch.zeros_like(nois_disc_pred))   # make disc predict     noised as     noised(0)
    real_loss = criterion(real_disc_pred, torch.ones_like(real_disc_pred))    # make disc predict nom-noised as non-noised(1)

    loss = (fake_loss + real_loss) / 2
    return loss, fake_loss, real_loss

def whisp_loss(criterion,
               gen, disc, whisp,
               data, dataset_bonafide_class,
               label, reverse_label, type=0): # for whisp it is advised to keep type as 0
    if config.train_with_wavegan:
        z = torch.randn(data.shape[0], config.noise_size).to(config.device)
        noised = gen(data, z)
    else:
        noised = gen(data)

    real_disc_pred  = disc(data)
    nois_disc_pred  = disc(noised)

    real_whisp_with_disc = whisp.predict_on_data(data, real_disc_pred, dataset_bonafide_class)
    nois_whisp_with_disc = whisp.predict_on_data(noised, nois_disc_pred, dataset_bonafide_class)

    real_whisp_pred = torch.squeeze(whisp(real_whisp_with_disc), 1)
    nois_whisp_pred = torch.squeeze(whisp(nois_whisp_with_disc), 1)

    fake_loss = criterion(nois_whisp_pred, label) # make whisp predict nosied data as real label(label)
    real_loss = criterion(real_whisp_pred, label) # make whisp predict real   data as real label(label)

    loss = (fake_loss + real_loss) / 2
    return loss, fake_loss, real_loss