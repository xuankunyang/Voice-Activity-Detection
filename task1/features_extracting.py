import numpy as np
from pathlib import Path
import librosa
from evaluate import get_metrics
from vad_utils import prediction_to_vad_label
from vad_utils import read_label_from_file
from rich.progress import track
from tqdm import tqdm
from audio_file_process import process_audio_file_traning
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer



def features_and_labels(frame_size: float, frame_shift: float, 
                        sampling_rate: int = 16000, num_subbands: int = 6, window:str = 'hanning', 
                        smooth:bool = 1, smooth_type:str = 'medfilt', alpha: float = 0.97, overlap: float = 1.):
    """
    Obtain the features and labels
    """

    frame_length = int(frame_size * sampling_rate)
    frame_step = int(frame_shift * sampling_rate)

    audio_dir = Path('vad/wavs/dev')
    label_path = 'vad/data/dev_label.txt'

    labels_dict = read_label_from_file(label_path, frame_size, frame_shift)

    all_energies = []
    all_zcr = []
    all_spectral_means = []
    all_f0 = []
    all_labels = []
    all_subband_energies = np.empty((num_subbands, ), dtype=object)

    for i in range(num_subbands):
        all_subband_energies[i] = []

    audio_files = list(audio_dir.glob("*.wav"))

    for audio_file in tqdm(audio_files, desc="Processing audio files", unit="file"):
        features, labels = process_audio_file_traning(label_dict=labels_dict, audio_path=str(audio_file), 
                                                    frame_length=frame_length, frame_step=frame_step, 
                                                    num_subbands=num_subbands, alpha=alpha, 
                                                    smooth=smooth, sampling_rate=sampling_rate, 
                                                    smooth_type=smooth_type, window=window, overlap=overlap)

        if len(labels) > 0:
            all_energies.extend(features['energies'])
            all_spectral_means.extend(features['spectral_means'])
            all_zcr.extend(features['zcrs'])
            all_f0.extend(features['f0'])
            for i in range(num_subbands):
                all_subband_energies[i].extend(features['subband_energies'][i])
            all_labels.extend(labels)

    X = [all_energies, all_zcr, all_spectral_means, all_f0]
    for i in range(num_subbands):
        X.append(all_subband_energies[i])

    X = np.array(X).T
    y = np.array(all_labels)

    np.savez('task1/features_and_labels_new/length_{}_step_{}.npz'.format(frame_length, frame_step), 
             features = X, labels = y, frame_length=frame_length, frame_step=frame_step, num_subbands=num_subbands, 
             alpha=alpha, smooth=smooth, sampling_rate=sampling_rate, smooth_type=smooth_type, window=window, overlap = overlap)

    return X, y


if __name__ == '__main__':
    # frame_size_list = [float(0.032), float(0.064), float(0.128), float(0.256), float(0.032), float(0.064), float(0.128), float(0.256)]
    # frame_shift_list = [float(0.008), float(0.016), float(0.032), float(0.064), float(0.016), float(0.032), float(0.064), float(0.128)]
    # frame_size_list = [float(0.020), float(0.032), float(0.064), float(0.128), float(0.256), float(0.020), float(0.032), float(0.064), float(0.128), float(0.256)]
    # frame_shift_list = [float(0.005), float(0.008), float(0.016), float(0.032), float(0.064), float(0.010), float(0.016), float(0.032), float(0.064), float(0.128)]

    frame_size_list = [float(0.020), float(0.020)]
    frame_shift_list = [float(0.005), float(0.010)]

    for frame_size, frame_shift in zip(frame_size_list, frame_shift_list):
        features_and_labels(frame_size=frame_size, frame_shift=frame_shift)