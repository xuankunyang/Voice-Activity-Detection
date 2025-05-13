import numpy as np
from pathlib import Path
import librosa
from evaluate import get_metrics
from vad_utils import prediction_to_vad_label
from vad_utils import read_label_from_file
from rich.progress import track
from tqdm import tqdm
from audio_file_process import process_audio_file_traning, process_audio_file_test
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt


num_subbands = int(6)
smooth = bool(1)
smooth_type = str('medfilt')
pre_emphasis = bool(1)
alpha = float(0.97)
sampling_rate = int(16000)
num_filters = int(23)
num_mfcc = int(12)
overlap = float(1.0)
delta = int(1)
window = 'hanning'


def features_counting_on_frame_length_and_frame_step_dev(frame_size: float, frame_shift: float):
    """
    Extracting features on trainning data
    """

    frame_length = int(frame_size * sampling_rate)
    frame_step = int(frame_shift * sampling_rate)

    train_audio_dir = Path('vad/wavs/dev')
    train_label_path = 'vad/data/dev_label.txt'

    labels_dict = read_label_from_file(train_label_path, frame_size, frame_shift)

    all_energies = []
    all_zcr = []
    all_spectral_means = []
    all_f0 = []
    all_labels = []
    all_subband_energies = np.empty((num_subbands, ), dtype=object)
    all_fbanks = np.empty((num_filters, ), dtype=object)
    all_mfcc = np.empty((num_mfcc, ), dtype=object)
    all_delta_mfcc = np.empty((num_mfcc, ), dtype=object)
    all_delta_delta_mfcc = np.empty((num_mfcc, ), dtype=object)


    for i in range(num_subbands):
        all_subband_energies[i] = []

    for i in range(num_filters):
        all_fbanks[i] = []

    for i in range(num_mfcc):
        all_mfcc[i] = []

    for i in range(num_mfcc):
        all_delta_mfcc[i] = []

    for i in range(num_mfcc):
        all_delta_delta_mfcc[i] = []

    audio_files = list(train_audio_dir.glob("*.wav"))

    for audio_file in tqdm(audio_files, desc="Processing audio files", unit="file"):
        features, labels = process_audio_file_traning(label_dict=labels_dict, audio_path=str(audio_file), 
                                            frame_length=frame_length, frame_step=frame_step, 
                                            num_subbands=num_subbands, alpha=alpha, pre_emphasis=pre_emphasis, 
                                            smooth=smooth, sampling_rate=sampling_rate, 
                                            smooth_type=smooth_type, window=window, overlap=overlap,
                                            num_filters=num_filters, num_mfcc=num_mfcc, delta=delta)

        if len(labels) > 0:
            all_energies.extend(features['energies'])
            all_spectral_means.extend(features['spectral_means'])
            all_zcr.extend(features['zcrs'])
            for i in range(num_subbands):
                all_subband_energies[i].extend(features['subband_energies'][i])
            for i in range(num_filters):
                all_fbanks[i].extend(features['f_banks'][:, i])
            for i in range(num_mfcc):
                all_mfcc[i].extend(features['mfccs'][:, i])
            for i in range(num_mfcc):
                all_delta_mfcc[i].extend(features['delta_mfccs'][:, i])
            for i in range(num_mfcc):
                all_delta_delta_mfcc[i].extend(features['delta_delta_mfccs'][:, i])
            all_f0.extend(features['f0'])
            all_labels.extend(labels)
    
    X = [all_energies, all_zcr, all_spectral_means, all_f0]
    for i in range(num_subbands):
        X.append(all_subband_energies[i])

    for i in range(num_filters):
        X.append(all_fbanks[i])

    for i in range(num_mfcc):
        X.append(all_mfcc[i])

    for i in range(num_mfcc):
        X.append(all_delta_mfcc[i])

    for i in range(num_mfcc):
        X.append(all_delta_delta_mfcc[i])


    X = np.array(X).T
    y = np.array(all_labels)

    np.savez('task2/dev_features_and_labels/length_{}_step_{}.npz'.format(frame_length, frame_step), 
             features = X, labels = y, frame_length=frame_length, frame_step=frame_step, 
             num_subbands=num_subbands, alpha=alpha, pre_emphasis=pre_emphasis, 
             smooth=smooth, sampling_rate=sampling_rate, smooth_type=smooth_type, 
             window=window, overlap=overlap, num_filters=num_filters, num_mfcc=num_mfcc, delta=delta)

    return X, y

def features_counting_on_frame_length_and_frame_step_train(frame_size: float, frame_shift: float):
    """
    Extracting features on trainning data
    """

    frame_length = int(frame_size * sampling_rate)
    frame_step = int(frame_shift * sampling_rate)

    train_audio_dir = Path('vad/wavs/train')
    train_label_path = 'vad/data/train_label.txt'

    labels_dict = read_label_from_file(train_label_path, frame_size, frame_shift)

    all_energies = []
    all_zcr = []
    all_spectral_means = []
    all_f0 = []
    all_labels = []
    all_subband_energies = np.empty((num_subbands, ), dtype=object)
    all_fbanks = np.empty((num_filters, ), dtype=object)
    all_mfcc = np.empty((num_mfcc, ), dtype=object)
    all_delta_mfcc = np.empty((num_mfcc, ), dtype=object)
    all_delta_delta_mfcc = np.empty((num_mfcc, ), dtype=object)


    for i in range(num_subbands):
        all_subband_energies[i] = []

    for i in range(num_filters):
        all_fbanks[i] = []

    for i in range(num_mfcc):
        all_mfcc[i] = []

    for i in range(num_mfcc):
        all_delta_mfcc[i] = []

    for i in range(num_mfcc):
        all_delta_delta_mfcc[i] = []

    audio_files = list(train_audio_dir.glob("*.wav"))

    for audio_file in tqdm(audio_files, desc="Processing audio files", unit="file"):
        features, labels = process_audio_file_traning(label_dict=labels_dict, audio_path=str(audio_file), 
                                            frame_length=frame_length, frame_step=frame_step, 
                                            num_subbands=num_subbands, alpha=alpha, pre_emphasis=pre_emphasis, 
                                            smooth=smooth, sampling_rate=sampling_rate, 
                                            smooth_type=smooth_type, window=window, overlap=overlap,
                                            num_filters=num_filters, num_mfcc=num_mfcc, delta=delta)

        if len(labels) > 0:
            all_energies.extend(features['energies'])
            all_spectral_means.extend(features['spectral_means'])
            all_zcr.extend(features['zcrs'])
            for i in range(num_subbands):
                all_subband_energies[i].extend(features['subband_energies'][i])
            for i in range(num_filters):
                all_fbanks[i].extend(features['f_banks'][:, i])
            for i in range(num_mfcc):
                all_mfcc[i].extend(features['mfccs'][:, i])
            for i in range(num_mfcc):
                all_delta_mfcc[i].extend(features['delta_mfccs'][:, i])
            for i in range(num_mfcc):
                all_delta_delta_mfcc[i].extend(features['delta_delta_mfccs'][:, i])
            all_f0.extend(features['f0'])
            all_labels.extend(labels)
    
    X = [all_energies, all_zcr, all_spectral_means, all_f0]
    for i in range(num_subbands):
        X.append(all_subband_energies[i])

    for i in range(num_filters):
        X.append(all_fbanks[i])

    for i in range(num_mfcc):
        X.append(all_mfcc[i])

    for i in range(num_mfcc):
        X.append(all_delta_mfcc[i])

    for i in range(num_mfcc):
        X.append(all_delta_delta_mfcc[i])


    X = np.array(X).T
    y = np.array(all_labels)

    np.savez('task2/training_data_features_and_labels/length_{}_step_{}.npz'.format(frame_length, frame_step), 
             features = X, labels = y, frame_length=frame_length, frame_step=frame_step, 
             num_subbands=num_subbands, alpha=alpha, pre_emphasis=pre_emphasis, 
             smooth=smooth, sampling_rate=sampling_rate, smooth_type=smooth_type, 
             window=window, overlap=overlap, num_filters=num_filters, num_mfcc=num_mfcc, delta=delta)

    return X, y

def features_counting_on_frame_length_and_frame_step_test(frame_size: float, frame_shift: float):
    """
    Extracting features on test data
    """

    frame_length = int(frame_size * sampling_rate)
    frame_step = int(frame_shift * sampling_rate)

    audio_dir = Path('vad/wavs/test')


    all_energies = []
    all_zcr = []
    all_spectral_means = []
    all_f0 = []
    all_subband_energies = np.empty((num_subbands, ), dtype=object)
    all_fbanks = np.empty((num_filters, ), dtype=object)
    all_mfcc = np.empty((num_mfcc, ), dtype=object)
    all_delta_mfcc = np.empty((num_mfcc, ), dtype=object)
    all_delta_delta_mfcc = np.empty((num_mfcc, ), dtype=object)


    for i in range(num_subbands):
        all_subband_energies[i] = []

    for i in range(num_filters):
        all_fbanks[i] = []

    for i in range(num_mfcc):
        all_mfcc[i] = []

    for i in range(num_mfcc):
        all_delta_mfcc[i] = []

    for i in range(num_mfcc):
        all_delta_delta_mfcc[i] = []

    audio_files = list(audio_dir.glob("*.wav"))

    for audio_file in tqdm(audio_files, desc="Processing audio files", unit="file"):
        features = process_audio_file_test(audio_path=str(audio_file), 
                                            frame_length=frame_length, frame_step=frame_step, 
                                            num_subbands=num_subbands, alpha=alpha, pre_emphasis=pre_emphasis, 
                                            smooth=smooth, sampling_rate=sampling_rate, 
                                            smooth_type=smooth_type, window=window, overlap=overlap,
                                            num_filters=num_filters, num_mfcc=num_mfcc, delta=delta)

        all_energies.extend(features['energies'])
        all_spectral_means.extend(features['spectral_means'])
        all_zcr.extend(features['zcrs'])
        for i in range(num_subbands):
            all_subband_energies[i].extend(features['subband_energies'][i])
        for i in range(num_filters):
            all_fbanks[i].extend(features['f_banks'][:, i])
        for i in range(num_mfcc):
            all_mfcc[i].extend(features['mfccs'][:, i])
        for i in range(num_mfcc):
            all_delta_mfcc[i].extend(features['delta_mfccs'][:, i])
        for i in range(num_mfcc):
            all_delta_delta_mfcc[i].extend(features['delta_delta_mfccs'][:, i])
        all_f0.extend(features['f0'])
    
    X = [all_energies, all_zcr, all_spectral_means, all_f0]
    for i in range(num_subbands):
        X.append(all_subband_energies[i])

    for i in range(num_filters):
        X.append(all_fbanks[i])

    for i in range(num_mfcc):
        X.append(all_mfcc[i])

    for i in range(num_mfcc):
        X.append(all_delta_mfcc[i])

    for i in range(num_mfcc):
        X.append(all_delta_delta_mfcc[i])


    X = np.array(X).T

    np.savez('task2/test_data_features_and_labels/length_{}_step_{}.npz'.format(frame_length, frame_step), 
             features = X, frame_length=frame_length, frame_step=frame_step, 
             num_subbands=num_subbands, alpha=alpha, pre_emphasis=pre_emphasis, 
             smooth=smooth, sampling_rate=sampling_rate, smooth_type=smooth_type, 
             window=window, overlap=overlap, num_filters=num_filters, num_mfcc=num_mfcc, delta=delta)

    return X


if __name__ == '__main__':

    # frame_size_list = [float(0.032), float(0.064), float(0.128), float(0.256), float(0.032), float(0.064), float(0.128), float(0.256)]
    # frame_shift_list = [float(0.008), float(0.016), float(0.032), float(0.064), float(0.016), float(0.032), float(0.064), float(0.128)]
    frame_size_list = [float(0.020), float(0.032), float(0.064), float(0.128), float(0.256), float(0.020), float(0.032), float(0.064), float(0.128), float(0.256)]
    frame_shift_list = [float(0.005), float(0.008), float(0.016), float(0.032), float(0.064), float(0.010), float(0.016), float(0.032), float(0.064), float(0.128)]

    # frame_size_list = [float(0.032)]
    # frame_shift_list = [float(0.008)]
    
    # frame_size_list = [float(0.020), float(0.020)]
    # frame_shift_list = [float(0.005), float(0.010)]

    # frame_size_list = [float(0.032)]
    # frame_shift_list = [float(0.008)]

    # frame_size_list = [float(0.020), float(0.064), float(0.128), float(0.256), float(0.020), float(0.032), float(0.064), float(0.128)]
    # frame_shift_list = [float(0.005), float(0.016), float(0.032), float(0.064), float(0.010), float(0.016), float(0.032), float(0.064)]


    for frame_size, frame_shift in tqdm(zip(frame_size_list, frame_shift_list), desc="Processing different pairs", unit="pair"):
        features_counting_on_frame_length_and_frame_step_dev(frame_size=frame_size, frame_shift=frame_shift)
        features_counting_on_frame_length_and_frame_step_train(frame_size=frame_size, frame_shift=frame_shift)

        # features_counting_on_frame_length_and_frame_step_test(frame_size=frame_size, frame_shift=frame_shift)


