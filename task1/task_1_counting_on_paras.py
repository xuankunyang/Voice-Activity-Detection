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
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt

num_subbands = int(6)
smooth = bool(1)
smooth_type = str('medfilt')
pre_emphasis = bool(1)
alpha = float(0.97)
sampling_rate = int(16000)
window = 'hanning'


def counting_on_frame_length_and_frame_step(frame_size: float, frame_shift: float):
    """
    Make comparision
    """

    frame_length = int(frame_size * sampling_rate)
    frame_step = int(frame_shift * sampling_rate)

    train_audio_dir = Path('vad/wavs/dev')
    train_label_path = 'vad/data/dev_label.txt'

    labels_dict = read_label_from_file(train_label_path, frame_size, frame_shift)

    all_energies = []
    all_zcr = []
    all_freq_centers = []
    all_f0 = []
    all_labels = []
    all_subband_energies = np.empty((num_subbands, ), dtype=object)

    for i in range(num_subbands):
        all_subband_energies[i] = []

    audio_files = list(train_audio_dir.glob("*.wav"))

    for audio_file in tqdm(audio_files, desc="Processing audio files", unit="file"):
        features, labels = process_audio_file_traning(label_dict=labels_dict, audio_path=str(audio_file), 
                                                    frame_length=frame_length, frame_step=frame_step, 
                                                    num_subbands=num_subbands, alpha=alpha, 
                                                    smooth=smooth, sampling_rate=sampling_rate, 
                                                    smooth_type=smooth_type, window=window)

        if len(labels) > 0:
            all_energies.extend(features['energies'])
            all_freq_centers.extend(features['freq_centers'])
            all_zcr.extend(features['zcrs'])
            for i in range(num_subbands):
                all_subband_energies[i].extend(features['subband_energies'][i])
            all_f0.extend(features['f0'])
            all_labels.extend(labels)

    X = [all_energies, all_zcr, all_freq_centers, all_f0]
    for i in range(num_subbands):
        X.append(all_subband_energies[i])

    X = np.array(X).T
    y = np.array(all_labels)

    # np.savez('task1/features_and_labels/length_{}_step_{}.npz'.format(frame_length, frame_step), 
    #          features = X, labels = y, frame_length=frame_length, frame_step=frame_step, num_subbands=num_subbands, 
    #          alpha=alpha, smooth=smooth, sampling_rate=sampling_rate, smooth_type=smooth_type, window=window)

    return X, y


if __name__ == '__main__':

    data = np.load('task1/features_and_labels/length_{}_step_{}.npz'.format(2048, 512))

    X = data['features']
    y = data['labels']

    imputer = SimpleImputer(strategy='mean')

    X = imputer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True)

    clf = LinearSVC(C=1, verbose=1, max_iter=1000)

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(acc)

    auc, eer = get_metrics(prediction=y_pred, label=y_test)
    print(auc, eer)

    # frame_size_list = [float(0.032), float(0.064), float(0.128), float(0.256), float(0.032), float(0.064), float(0.128), float(0.256)]
    # frame_shift_list = [float(0.008), float(0.016), float(0.032), float(0.064), float(0.016), float(0.032), float(0.064), float(0.128)]

    # all_auc = []
    # all_eer = []
    # all_acc = []

    # for frame_size, frame_shift in zip(frame_size_list, frame_shift_list):
    #     # X, y = counting_on_frame_length_and_frame_step(frame_size=frame_size, frame_shift=frame_shift)

    #     frame_length = int(frame_size * sampling_rate)
    #     frame_step = int(frame_shift * sampling_rate)

    #     data = np.load('task1/features_and_labels/length_{}_step_{}.npz'.format(frame_length, frame_step))

    #     X = data['features']
    #     y = data['labels']

    #     imputer = SimpleImputer(strategy='mean')

    #     X = imputer.fit_transform(X)

    #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True)

    #     clf = LinearSVC(C=1, verbose=1, max_iter=1000)

    #     clf.fit(X_train, y_train)

    #     y_pred = clf.predict(X_test)

    #     acc = accuracy_score(y_test, y_pred)
    #     print(acc)

    #     auc, eer = get_metrics(prediction=y_pred, label=y_test)
    #     print(auc, eer)

    #     all_auc.append(auc)
    #     all_acc.append(acc)
    #     all_eer.append(eer)


    # frame_sizes = np.array([1000 * x for x in frame_size_list])
    # frame_sizes = np.log2(frame_sizes)

    # plt.figure()
    # plt.plot(frame_sizes[:4], all_auc[:4], )
    # plt.plot(frame_sizes[:4], all_acc[:4])
    # plt.plot(frame_sizes[:4], all_eer[:4])
    # plt.xlabel('frame size(log2) / ms')
    # plt.legend(['auc', 'acc', 'eer'])
    # plt.title('frame size : frame shift = 4')
    # plt.savefig('task1/figs/ratio_4.png')
    # plt.close()

    # plt.figure()
    # plt.plot(frame_sizes[4:], all_auc[4:])
    # plt.plot(frame_sizes[4:], all_acc[4:])
    # plt.plot(frame_sizes[4:], all_eer[4:])
    # plt.xlabel('frame size(log2) / ms')
    # plt.legend(['auc', 'acc', 'eer'])
    # plt.title('frame size : frame shift = 2')
    # plt.savefig('task1/figs/ratio_2.png')
    



