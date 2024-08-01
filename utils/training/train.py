from tqdm.auto import tqdm
import torch

import sys 
import os
sys.path.append(os.path.abspath('/tank/local/ndf3868/GODDS/GAN/utils'))

from utils import generator_loss, discriminator_loss


def train(dataloader, gen, disc, 
          criterion, disc_opt, gen_opt, 
          n_epochs, device):
    for epoch in range(n_epochs): #tqdm(range(n_epochs), desc='Training', leave):
        train_epoch(dataloader, gen, disc, 
                criterion, disc_opt, gen_opt, 
                epoch, device)

def train_epoch(dataloader, gen, disc, 
                criterion, disc_opt, gen_opt, 
                epoch, device):
    
    gen.train()
    disc.train()

    cur_step                = 0
    display_step            = 3
    mean_discriminator_loss = 0
    mean_generator_loss     = 0

    pbar = tqdm(dataloader, desc='description')

    for (data, sr, label, gen_type) in pbar:
        data = data.float()

        cur_batch_size = len(data)
        data           = data.to(device)
        label          = label.float().to(device)
        reverse_label  = torch.sub(torch.ones_like(label),label) # 1 initially REAL now 1 FAKE
        
        disc_opt.zero_grad()
        disc_loss, disc_fake_loss, disc_real_loss = discriminator_loss(gen, disc, criterion, data, label, reverse_label, type=0)    # type = 1 - noised vs non noised  |   0 - fake vs real
        disc_loss.backward(retain_graph=True)
        disc_opt.step()

        gen_opt.zero_grad()
        gen_loss = generator_loss(gen, disc, criterion, data, label, reverse_label, type=0)         # type = 1 - noised vs non noised  |   0 - fake vs real
        gen_loss.backward()
        gen_opt.step()

        mean_discriminator_loss += disc_loss.item() / display_step
        mean_generator_loss     +=  gen_loss.item() / display_step

        if cur_step % display_step == 0 and cur_step > 0:
            # print(f"Step {cur_step}: Generator loss: {mean_generator_loss}, discriminator loss: {mean_discriminator_loss}")
            pbar.set_description(f"Epoch {epoch} Step {cur_step}: Generator loss: {mean_generator_loss:.4f}, discriminator loss: {mean_discriminator_loss:.4f}  Last Beatch: Fake {disc_fake_loss:.4f}   Real{disc_real_loss:.4f}")
            mean_generator_loss = 0
            mean_discriminator_loss = 0

        cur_step += 1
