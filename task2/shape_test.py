from audio_file_process import process_audio_file_traning
import numpy as np
from pathlib import Path
from vad_utils import read_label_from_file
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler




frame_size = float(0.128)
frame_shift = float(0.032)
num_subbands = int(6)
smooth = bool(1)
smooth_type = str('medfilt')
pre_emphasis = bool(1)
alpha = float(0.97)
sampling_rate = int(16000)
frame_length = int(sampling_rate * frame_size)
frame_step = int(sampling_rate * frame_shift)
num_filters = int(23)
num_mfcc = int(12)
overlap = float(1.0)
delta = int(1)

train_audio_dir = Path('vad/wavs/dev')
train_label_path = 'vad/data/dev_label.txt'

labels_dict = read_label_from_file(train_label_path, frame_size, frame_shift)


all_energies = []
all_zcr = []
all_freq_centers = []
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
                                        num_subbands=num_subbands, alpha=alpha, 
                                        smooth=smooth, sampling_rate=sampling_rate, 
                                        smooth_type=smooth_type, window='hanning', overlap=overlap,
                                        num_filters=num_filters, num_mfcc=num_mfcc, delta=delta)

    if len(labels) > 0:
        all_energies.extend(features['energies'])
        all_freq_centers.extend(features['freq_centers'])
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

print(len(all_energies))
print(len(all_subband_energies[0]))
print(len(all_zcr))
print(len(all_f0))
print(len(all_freq_centers))
print(len(all_labels))
print(len(all_fbanks[0]))
print(all_fbanks.shape)
print(len(all_mfcc[0]))
print(all_mfcc.shape)
print(len(all_delta_mfcc[0]))
print(all_delta_mfcc.shape)
print(len(all_delta_delta_mfcc[0]))
print(all_delta_delta_mfcc.shape)


X = [all_energies, all_zcr, all_freq_centers, all_f0]
for i in range(num_subbands):
    X.append(all_subband_energies[i])

for i in range(num_filters):
    X.append(all_fbanks[i])

for i in range(num_mfcc):
    X.append(all_mfcc[i])

X = np.array(X).T
y = np.array(all_labels)

print(X.shape)