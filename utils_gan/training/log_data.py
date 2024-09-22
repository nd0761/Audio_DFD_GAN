import json
import sys 
import os
sys.path.append(os.path.abspath('/tank/local/ndf3868/GODDS/GAN/utils_gan')) # IMPORTANT

from training.distribution_visualizer import visualize_separate
import torch
import config
import numpy as np

def log_metrics(predictions, metrics, 
                logs_dir, epoch):
    visualize_separate(predictions['disc']['pred noised'],  predictions['disc']['pred non-noised'],  
                               os.path.join(logs_dir, 'distr', f'epoch_{epoch}_density_distribution_DISC.png'))
    visualize_separate(predictions['whisp']['pred noised'], predictions['whisp']['pred non-noised'], 
                        os.path.join(logs_dir, 'distr', f'epoch_{epoch}_density_distribution_WHISP.png'))

    with open(os.path.join(logs_dir, 'metrics', f"sample_iteration_{epoch}.json"), "w") as outfile: 
        json.dump(metrics, outfile, indent=2)
    
def save_checkpoint(ckpt_dir, epoch,
                gen, disc, whisp,
                gen_opt, disc_opt, whisp_opt, 
                ckpt_path):
    ckpt_dict = {
        "epoch":epoch,

        "gen_state_dict":   gen.state_dict(),
        "disc_state_dict":  disc.state_dict(),
        "whisp_state_dict": whisp.state_dict(),

        "gen_opt_state_dict":   gen_opt.state_dict(),
        "disc_opt_state_dict":  disc_opt.state_dict(),
        "whisp_opt_state_dict": whisp_opt.state_dict(),
    }
    if config.save_logs: torch.save(ckpt_dict, ckpt_path)
    return ckpt_dict

def log_audio(gen, data, sr, epoch, logs_dir, z):
    from scipy.io.wavfile import write
    import soundfile as sf
    gen.eval()

    sr = sr[0]
    data = data[0][None, :].to(config.device)
    # print(data[0, :30, :30])

    # z = torch.randn(1, config.noise_size).to(config.device)
    noised = gen(data, z).cpu().detach().numpy()
    data = data.cpu().detach().numpy()

    # print(data.squeeze(0).squeeze(0).shape, noised.squeeze(0).shape)

    orig_wav = os.path.join(logs_dir, 'audio', f'orig.wav')
    nois_wav = os.path.join(logs_dir, 'audio', f'{epoch}_nois.wav')

    sf.write(orig_wav, data.squeeze(0).squeeze(0), sr)
    sf.write(nois_wav, noised.squeeze(0).squeeze(0), sr)
    return orig_wav, nois_wav

def log_spectrogram(audio_file, spec_file):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

    import librosa.display
    import librosa
    # spec_file = os.path.join(logs_dir, 'spectrograms', f'{epoch}_orig.png')

    y, sr = librosa.load(audio_file)
    D = librosa.stft(y)  
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    fig, ax = plt.subplots(ncols=2, figsize=(12, 4))

    # Plot the spectrogram on the first subplot
    img = librosa.display.specshow(S_db, x_axis='time', y_axis='linear', ax=ax[0])
    ax[0].set(title='Spectrogram', xlabel='Time (s)', ylabel='Frequency (Hz)')
    fig.colorbar(img, ax=ax[0], format="%+2.f dB")

    # Plot the waveform on the second subplot
    librosa.display.waveshow(y, sr=sr, ax=ax[1])
    ax[1].set(title='Waveform', xlabel='Time (s)', ylabel='Amplitude')

    # Save the figure to file
    fig.savefig(spec_file)

    # D = librosa.stft(y)  # STFT of y
    # S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    # fig, ax = plt.subplots()
    # img = librosa.display.specshow(S_db, x_axis='time', y_axis='linear', ax=ax)
    # ax.set(title='Now with labeled axes!')
    # fig.colorbar(img, ax=ax, format="%+2.f dB")
    # fig.savefig(spec_file)

    # window_size = 1024
    # window = np.hanning(window_size)
    # stft  = librosa.core.spectrum.stft(y, n_fft=window_size, hop_length=512, window=window)
    # out = 2 * np.abs(stft) / np.sum(window)

    # fig = plt.Figure()
    # canvas = FigureCanvas(fig)
    # ax = fig.add_subplot(111)
    # p = librosa.display.specshow(librosa.amplitude_to_db(out, ref=np.max), ax=ax, y_axis='log', x_axis='time')
    # fig.savefig(spec_file)
