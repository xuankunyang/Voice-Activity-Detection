import numpy as np
from pathlib import Path
import librosa
from evaluate import get_metrics
from vad_utils import prediction_to_vad_label
from vad_utils import read_label_from_file
from rich.progress import track
from tqdm import tqdm
from audio_file_process_voiced_and_unvoiced import process_audio_file_traning
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer



def features_voiced_and_unvoiced(frame_size: float, frame_shift: float, 
                        dev_label_path: Path, dev_audio_dir: str, train_label_path: Path, train_audio_dir: str, 
                        sampling_rate: int = 16000, num_subbands: int = 6, window: str = 'hanning', 
                        smooth: bool = 1, smooth_type:str = 'medfilt', alpha: float = 0.97, overlap:float = 1.0):
    """
    Obtain the features for voiced and unvoiced **frames**
    """

    frame_length = int(frame_size * sampling_rate)
    frame_step = int(frame_shift * sampling_rate)

    dev_labels_dict = read_label_from_file(dev_label_path, frame_size, frame_shift)
    train_labels_dict = read_label_from_file(train_label_path, frame_size, frame_shift)
    labels_dict = dev_labels_dict | train_labels_dict

    dev_audio_files = list(dev_audio_dir.glob("*.wav"))
    train_audio_files = list(train_audio_dir.glob("*.wav"))
    audio_files = dev_audio_files + train_audio_files

    voiced_energies = []
    voiced_zcr = []
    voiced_freq_centers = []
    voiced_f0 = []
    voiced_subband_energies = np.empty((num_subbands, ), dtype=object)

    unvoiced_energies = []
    unvoiced_zcr = []
    unvoiced_freq_centers = []
    unvoiced_f0 = []
    unvoiced_subband_energies = np.empty((num_subbands, ), dtype=object)

    for i in range(num_subbands):
        voiced_subband_energies[i] = []
        unvoiced_subband_energies[i] = []


    for audio_file in tqdm(audio_files, desc="Processing audio files", unit="file"):
        voiced_features, unvoiced_features = process_audio_file_traning(label_dict=labels_dict, 
                                                                        audio_path=str(audio_file), 
                                                    frame_length=frame_length, frame_step=frame_step, 
                                                    num_subbands=num_subbands, alpha=alpha, overlap=overlap,
                                                    smooth=smooth, sampling_rate=sampling_rate, 
                                                    smooth_type=smooth_type, window=window)

        voiced_energies.extend(voiced_features['energies'])
        voiced_freq_centers.extend(voiced_features['freq_centers'])
        voiced_zcr.extend(voiced_features['zcrs'])
        voiced_f0.extend(voiced_features['f0'])
        for i in range(num_subbands):
            voiced_subband_energies[i].extend(voiced_features['subband_energies'][i])


        unvoiced_energies.extend(unvoiced_features['energies'])
        unvoiced_freq_centers.extend(unvoiced_features['freq_centers'])
        unvoiced_zcr.extend(unvoiced_features['zcrs'])
        unvoiced_f0.extend(unvoiced_features['f0'])
        for i in range(num_subbands):
            unvoiced_subband_energies[i].extend(unvoiced_features['subband_energies'][i])


    X_voiced = [voiced_energies, voiced_zcr, voiced_freq_centers, voiced_f0]
    for i in range(num_subbands):
        X_voiced.append(voiced_subband_energies[i])

    X_voiced = np.array(X_voiced).T


    X_unvoiced = [unvoiced_energies, unvoiced_zcr, unvoiced_freq_centers, unvoiced_f0]
    for i in range(num_subbands):
        X_unvoiced.append(unvoiced_subband_energies[i])

    X_unvoiced = np.array(X_unvoiced).T


    np.savez('task2/features_voiced_and_unvoiced/length_{}_step_{}.npz'.format(frame_length, frame_step), 
             features_voiced = X_voiced, features_unvoiced = X_unvoiced, 
             frame_length=frame_length, frame_step=frame_step, num_subbands=num_subbands, 
             alpha=alpha, smooth=smooth, sampling_rate=sampling_rate, smooth_type=smooth_type, window=window, overlap=overlap)

    return X_voiced, X_unvoiced
