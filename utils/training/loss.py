import torch

def generator_loss(gen, disc, criterion, non_noised, label, reverse_label, type=1): # type = 1 - noised vs non noised  |   0 - fake vs real
    noised            = gen(non_noised)
    disc_noised_pred  = disc(noised)
    
    if type == 1:
        gen_loss = criterion(disc_noised_pred, torch.ones_like(disc_noised_pred)) # make disc predict data as not noised(1)
    else:
        gen_loss = criterion(disc_noised_pred, reverse_label) # make disc predict data as fake when it's not
    return gen_loss

def discriminator_loss(gen, disc, criterion, non_noised, label, reverse_label, type=1): # type = 1 - noised vs non noised  |   0 - fake vs real
    noised            = gen(non_noised)

    disc_noised_pred  = disc(noised)
    disc_real_pred    = disc(non_noised)

    if type == 1:
        disc_fake_loss = criterion(disc_noised_pred, torch.zeros_like(disc_noised_pred)) # make disc predict     noised as     noised(0)
        disc_real_loss = criterion(disc_real_pred,   torch.ones_like(disc_real_pred))    # make disc predict not noised as not noised(1)
    else:
        disc_fake_loss = criterion(disc_noised_pred, label) # make disc predict nosied data as real label(label)
        disc_real_loss = criterion(disc_real_pred,   label) # make disc predict real   data as real label(label)

    disc_loss = (disc_fake_loss + disc_real_loss) / 2
    return disc_loss, disc_fake_loss, disc_real_loss