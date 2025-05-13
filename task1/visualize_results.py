import matplotlib.pyplot as plt
from pathlib import Path
from audio_file_process import process_audio_file_traning
from vad_utils import read_label_from_file
import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler


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

    audio_dir = Path('vad/wavs/dev')
    label_path = 'vad/data/dev_label.txt'

    labels_dict = read_label_from_file(label_path, frame_size, frame_shift)

    audio_path = 'vad/wavs/dev/54-121080-0009.wav'
    audio_key = Path(audio_path).stem

    frame_length = int(frame_size * sampling_rate)
    frame_step = int(frame_shift * sampling_rate)

    feature, label = process_audio_file_traning(label_dict=labels_dict, audio_path=str(audio_path), 
                                            num_subbands=num_subbands, alpha=alpha, 
                                            frame_length=frame_length, frame_step=frame_step, 
                                            smooth=smooth, smooth_type=smooth_type, window=window)

    energies = []
    spectral_means = []
    zcr = []
    f0 = []
    subband_energies = np.empty((num_subbands, ), dtype=object)

    for i in range(num_subbands):
        subband_energies[i] = []


    if len(label) > 0:
        energies.extend(feature['energies'])
        spectral_means.extend(feature['spectral_means'])
        zcr.extend(feature['zcrs'])
        for i in range(num_subbands):
            subband_energies[i].extend(feature['subband_energies'][i])
        f0.extend(feature['f0'])

    x = [energies, zcr, spectral_means, f0]
    for i in range(num_subbands):
        x.append(subband_energies[i])

    X = np.array(x).T

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    pred = model.predict(X)

    predictions = deframe(pred, frame_length, frame_step)

    truth = deframe(label, frame_length, frame_step)
    
    return predictions, truth


