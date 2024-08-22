import torch

def generator_loss(criterion,
               gen, disc, whisp,
               data, dataset_bonafide_class,
               label, reverse_label, type=0, training_with_whisp=False): # type = 1 - noised vs non noised  |   0 - fake vs real
    noised            = gen(data)
    nois_disc_pred  = disc(noised)
    # disc_real_pred    = disc(data)

    if not training_with_whisp: 
        if type == 1:
            gen_loss = criterion(nois_disc_pred, torch.ones_like(nois_disc_pred)) # make disc predict data as not noised(1)
        else:
            gen_loss = criterion(nois_disc_pred, reverse_label) # make disc predict data as fake when it's not
    elif training_with_whisp:
        nois_whisp_with_disc= whisp.predict_on_data(noised, nois_disc_pred, dataset_bonafide_class)
        nois_whisp_pred     = torch.squeeze(whisp(nois_whisp_with_disc), 1)

        if type == 1:
            gen_loss = criterion(nois_whisp_pred, torch.ones_like(nois_disc_pred)) # make whisp+disc predict data as not noised(1)
        else:
            gen_loss = criterion(nois_whisp_pred, reverse_label) # make whisp+disc predict data as fake when it's not
    
    return gen_loss

def discriminator_loss(criterion, 
                       gen, disc,
                       data, 
                       label, reverse_label, type=1): # type = 1 - noised vs non noised  |   0 - fake vs real
    noised            = gen(data)

    nois_disc_pred = disc(noised)
    real_disc_pred = disc(data)

    if type == 1:
        fake_loss = criterion(nois_disc_pred, torch.zeros_like(nois_disc_pred)) # make disc predict     noised as     noised(0)
        real_loss = criterion(real_disc_pred, torch.ones_like(real_disc_pred))    # make disc predict not noised as not noised(1)
    else:
        fake_loss = criterion(nois_disc_pred, label) # make disc predict nosied data as real label(label)
        real_loss = criterion(real_disc_pred, label) # make disc predict real   data as real label(label)

    loss = (fake_loss + real_loss) / 2
    return loss, fake_loss, real_loss

def whisp_loss(criterion,
               gen, disc, whisp,
               data, dataset_bonafide_class,
               label, reverse_label, type=0): # for whisp it is advised to keep type as 0
    noised = gen(data)

    real_disc_pred  = disc(data)
    nois_disc_pred  = disc(noised)

    real_whisp_with_disc = whisp.predict_on_data(data, real_disc_pred, dataset_bonafide_class)
    nois_whisp_with_disc = whisp.predict_on_data(noised, nois_disc_pred, dataset_bonafide_class)

    real_whisp_pred = torch.squeeze(whisp(real_whisp_with_disc), 1)
    nois_whisp_pred = torch.squeeze(whisp(nois_whisp_with_disc), 1)

    if type == 1:
        fake_loss = criterion(nois_whisp_pred, torch.zeros_like(nois_whisp_pred)) # make whisp predict     noised as     noised(0)
        real_loss = criterion(real_whisp_pred, torch.ones_like( real_whisp_pred)) # make whisp predict not noised as not noised(1)
    else:
        fake_loss = criterion(nois_whisp_pred, label) # make whisp predict nosied data as real label(label)
        real_loss = criterion(real_whisp_pred, label) # make whisp predict real   data as real label(label)

    loss = (fake_loss + real_loss) / 2
    return loss, fake_loss, real_loss