from tqdm.auto import tqdm
import torch

import numpy as np

import json
import sys 
import os
sys.path.append(os.path.abspath('/tank/local/ndf3868/GODDS/GAN/utils_gan'))

from training.loss import generator_loss, discriminator_loss, whisp_loss
from training.metrics import set_up_metrics_list, test_metrics
from training.distribution_visualizer import visualize_separate

# from utils_gan import generator_loss, discriminator_loss, whisp_loss, test_data, set_up_metrics_list, visualize_separate


TRAINING_TYPE = 1 # 1 - noised vs non noised| 0 - generator vs bonafide

def produce_prediction_dict(train_with_wisper,
              dataloader, dataset_bonafide_class,
              gen, disc, whisp, 
              device):
    gen.eval()
    disc.eval()
    if train_with_wisper: whisp.eval()

    pbar = tqdm(dataloader, desc='Evaluation in progress')

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

    for (data, sr, label, gen_type) in pbar:
        data = data.float()

        cur_batch_size = len(data)
        data           = data.to(device)
        label          = label.float().cpu().detach().numpy()

        noised = gen(data)

        nois_disc_pred  = disc(noised)
        real_disc_pred  = disc(data)

        real_whisp_with_disc = whisp.predict_on_data(data,   real_disc_pred, dataset_bonafide_class)
        nois_whisp_with_disc = whisp.predict_on_data(noised, nois_disc_pred, dataset_bonafide_class)

        real_whisp_pred = torch.squeeze(whisp(real_whisp_with_disc), 1).cpu().detach().numpy()
        nois_whisp_pred = torch.squeeze(whisp(nois_whisp_with_disc), 1).cpu().detach().numpy()
        real_disc_pred  = real_disc_pred.cpu().detach().numpy()
        nois_disc_pred  = nois_disc_pred.cpu().detach().numpy()

        lab_no  = [label, np.zeros_like(label), label] # for type = 0 disc| for type = 1 disc | for whisp
        lab_non = [label, np.ones_like(label), label] # for type 1
        labs = [lab_no, lab_non]

        pred_no  = [nois_disc_pred, nois_disc_pred, nois_whisp_pred]
        pred_non = [real_disc_pred, real_disc_pred, real_whisp_pred]
        preds = [pred_no, pred_non]
        
        for prt_id, prt in enumerate(predictor_type):
            prt_id *= 2
            if prt_id == 0: prt_id = TRAINING_TYPE
            for pnt, vals in zip(prediction_type, [labs, preds]):
                for dt, val in zip(data_type, vals):
                    predictions[prt][pnt+' '+dt] = val[prt_id]
        
    for prt in predictor_type:
        for pt in prediction_type:
            predictions[prt][f'{pt} all'] = np.concatenate((predictions[prt][f'{pt} non-noised'], predictions[prt][f'{pt} noised']), axis=0)
    return predictions
            

def train(train_with_wisper, 
            train_dataloader, test_dataloader, 
            train_dataset, test_dataset,
            gen,        disc,       whisp,
            gen_opt,    disc_opt,   whisp_opt, 
            criterion,
            n_epochs, device,
            logs_dir, ckpt_dir):

    #----DEBUG----

    # audio_samples = torch.rand(size=(4, 190_000), dtype=torch.float32) - 0.5
    # targets = torch.randint(low=0, high=2, size=(8, 1))

    # audio_samples = audio_samples.to(device)
        
    # rand_pred, _ = whisp.detectors_prediction(audio_samples)
    # print(rand_pred)
    # rand_pred, _ = whisp.detectors_prediction(audio_samples)
    # print(rand_pred)
    # rand_pred, _ = whisp.detectors_prediction(audio_samples)
    # print(rand_pred)
    
    for epoch in range(n_epochs): #tqdm(range(n_epochs), desc='Training', leave):
        train_epoch(train_with_wisper, 
            train_dataloader, train_dataset,
            gen,        disc,       whisp,
            gen_opt,    disc_opt,   whisp_opt, 
            criterion,
            epoch, device)
        
        if epoch % 3 == 0 or epoch == n_epochs-1:
            predictions = produce_prediction_dict(train_with_wisper,
                                                test_dataloader, train_dataset.bonafide_class,
                                                gen, disc, whisp, 
                                                device)

            metrics_list = set_up_metrics_list(train_dataset.bonafide_class)

            metrics      = test_metrics(metrics_list, predictions)

            visualize_separate(predictions['disc']['pred noised'],  predictions['disc']['pred non-noised'],  
                               os.path.join(logs_dir, 'distr', f'epoch_{epoch}_density_distribution_DISC.png'))
            visualize_separate(predictions['whisp']['pred noised'], predictions['whisp']['pred non-noised'], 
                               os.path.join(logs_dir, 'distr', f'epoch_{epoch}_density_distribution_WHISP.png'))

            with open(os.path.join(logs_dir, 'metrics', f"sample_iteration_{epoch}.json"), "w") as outfile: 
                json.dump(metrics, outfile, indent=2)
            
            ckpt_path = os.path.join(ckpt_dir, f'epoch{epoch}.pt')
            torch.save({
                "epoch":epoch,

                "gen_state_dict":   gen.state_dict(),
                "disc_state_dict":  disc.state_dict(),
                "whisp_state_dict": whisp.state_dict(),

                "gen_opt_state_dict":   gen_opt.state_dict(),
                "disc_opt_state_dict":  disc_opt.state_dict(),
                "whisp_opt_state_dict": whisp_opt.state_dict(),
            }, ckpt_path)

def train_epoch(train_with_wisper, dataloader, dataset,
                gen,        disc,       whisp,
                gen_opt,    disc_opt,   whisp_opt, 
                criterion,
                epoch, device):
    
    gen.train()
    disc.train()
    if train_with_wisper: whisp.train()

    cur_step                = 0
    display_step            = 20
    mean_discriminator_loss = []
    mean_generator_loss     = []
    mean_whisp_loss         = []

    pbar = tqdm(dataloader, desc='description')

    for (data, sr, label, gen_type) in pbar:
        data = data.float()

        cur_batch_size = len(data)
        data           = data.to(device)
        label          = label.float().to(device)
        reverse_label  = torch.sub(torch.ones_like(label),label) # 1 initially REAL now 1 FAKE

        noised = gen(data)

        disc_noised_pred  = disc(noised)
        disc_real_pred    = disc(data)
            
        disc_opt.zero_grad()
        disc_loss, disc_fake_loss, disc_real_loss = discriminator_loss(criterion, 
                                                                       gen, disc,
                                                                       data, 
                                                                       label, reverse_label, type=TRAINING_TYPE)    # type = 1 - noised vs non noised  |   0 - fake vs real
        disc_loss.backward(retain_graph=True)
        disc_opt.step()
        
        if not train_with_wisper:
            gen_opt.zero_grad()
            gen_loss = generator_loss(criterion, 
                                      gen, disc, None,
                                      data, dataset.bonafide_class,
                                      label, reverse_label, type=TRAINING_TYPE, training_with_whisp=False)         # type = 1 - noised vs non noised  |   0 - fake vs real
            gen_loss.backward()
            gen_opt.step()
        else:
            whisp_opt.zero_grad()
            whisp_loss_val, _ , _ = whisp_loss(criterion,
                                            gen, disc, whisp,
                                            data, dataset.bonafide_class,
                                            label, reverse_label, type=0)
            whisp_loss_val.backward(retain_graph=True)
            whisp_opt.step()
            
            gen_opt.zero_grad()
            gen_loss = generator_loss(criterion, 
                                      gen, disc, whisp,
                                      data, dataset.bonafide_class,
                                      label, reverse_label, type=TRAINING_TYPE, training_with_whisp=True)         # type = 1 - noised vs non noised  |   0 - fake vs real
            gen_loss.backward()
            gen_opt.step()


        mean_discriminator_loss.append(disc_loss.item())
        mean_generator_loss.append(gen_loss.item())
        mean_whisp_loss.append(whisp_loss_val.item())

        if cur_step % display_step == 0 or cur_step == len(dataloader) -1:
            mean_discriminator_loss = sum(mean_discriminator_loss) / len(mean_discriminator_loss)
            mean_generator_loss     = sum(mean_generator_loss) / len(mean_generator_loss)
            mean_whisp_loss         = sum(mean_whisp_loss) / len(mean_whisp_loss)
            # print(f"Step {cur_step}: Generator loss: {mean_generator_loss}, discriminator loss: {mean_discriminator_loss}")
            pbar.set_description(f"Epoch {epoch} ✧Step {cur_step}✧  "+
f"Gen L: {mean_generator_loss:.3f} || "+ 
f"Dis L: {mean_discriminator_loss:.3f} || "+
f"Whi L: {mean_whisp_loss:.3f}")
# f"Last Batch    loss: Fake {disc_fake_loss:.4f}   Real{disc_real_loss:.4f}")
            mean_generator_loss     = []
            mean_discriminator_loss = []
            mean_whisp_loss         = []

        cur_step += 1
