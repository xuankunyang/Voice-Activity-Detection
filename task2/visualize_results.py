import matplotlib.pyplot as plt
from pathlib import Path
from audio_file_process import process_audio_file_traning
from vad_utils import read_label_from_file
import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler
import torch


def deframe(x_framed, win_len, hop_len):
    '''
        interpolates 1D data with framed alignments into persample values.
        This function helps as a visual aid and can also be used to change 
        frame-rate for features, e.g. energy, zero-crossing, etc.
        '''
    n_frames = len(x_framed)
    n_samples = n_frames*hop_len + win_len
    x_samples = np.zeros((n_samples,1))
    for i in range(n_frames):
        x_samples[i*hop_len : i*hop_len + win_len] = x_framed[i]
    return x_samples

def visulize_results(model, frame_size:float, frame_shift:float):
    num_subbands = int(6)
    smooth = bool(1)
    smooth_type = str('medfilt')
    pre_emphasis = bool(1)
    alpha = float(0.97)
    sampling_rate = int(16000)
    window = 'hanning'
    overlap = float(1.0)
    num_filters = int(23)
    num_mfcc = int(12)
    delta = int(1)

    audio_dir = Path('vad/wavs/dev')
    label_path = 'vad/data/dev_label.txt'

    labels_dict = read_label_from_file(label_path, frame_size, frame_shift)

    audio_path = 'vad/wavs/dev/54-121080-0009.wav'
    audio_key = Path(audio_path).stem

    frame_length = int(frame_size * sampling_rate)
    frame_step = int(frame_shift * sampling_rate)

    features, labels = process_audio_file_traning(label_dict=labels_dict,
                                                audio_path='vad/wavs/dev/54-121080-0009.wav', 
                                                frame_length=frame_length, frame_step=frame_step, 
                                                num_subbands=num_subbands, alpha=alpha, pre_emphasis=pre_emphasis, 
                                                smooth=smooth, sampling_rate=sampling_rate, 
                                                smooth_type=smooth_type, window=window, overlap=overlap,
                                                num_filters=num_filters, num_mfcc=num_mfcc, delta=delta)

    all_energies = features['energies']
    all_spectral_means = features['spectral_means']
    all_zcr = features['zcrs']
    all_subband_energies = [features['subband_energies'][i] for i in range(num_subbands)]
    all_fbanks = [features['f_banks'][:, i] for i in range(num_filters)]
    all_mfcc = [features['mfccs'][:, i] for i in range(num_mfcc)]
    all_delta_mfcc = [features['delta_mfccs'][:, i] for i in range(num_mfcc)]
    all_delta_delta_mfcc = [features['delta_delta_mfccs'][:, i] for i in range(num_mfcc)]
    all_f0 = features['f0']

    X = [all_energies, all_zcr, all_spectral_means, all_f0] + \
        all_subband_energies + all_fbanks + all_mfcc + all_delta_mfcc + all_delta_delta_mfcc

    X = np.array(X).T

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on device: {device}")

    inputs = torch.tensor(X, dtype=torch.float32).to(device) 
    outputs = model(inputs) 

    pred = (outputs > 0.5).float()
    pred = pred.cpu().detach().numpy()

    preditions = deframe(pred, frame_length, frame_step)
    truth = deframe(labels, frame_length, frame_step)
    
    return preditions, truth


