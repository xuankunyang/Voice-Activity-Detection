import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,roc_curve
from evaluate import get_metrics
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime
from tqdm import tqdm
from sklearn.svm import LinearSVC
from visualize_results import visulize_results
import librosa
import joblib


class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()  # 将 ndarray 转换为列表
        return super().default(o)

def Comparision_on_frame_size_and_shift_SVM(frame_size: float, frame_shift: float, kernel: str = 'linear'):
    sampling_rate = 16000
    
    frame_length = int(frame_size * sampling_rate)
    frame_step = int(frame_shift * sampling_rate)

    C = 0.5
    max_iter = 10000
    penalty='l2'

    data = np.load('task1/features_and_labels_new/length_{}_step_{}.npz'.format(frame_length, frame_step))

    X = data['features']
    y = data['labels']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # model = SVC(probability=True, verbose=1, kernel=kernel,  C=C, gamma=gamma, max_iter=max_iter, shrinking=shrinking)
    model = LinearSVC(C=C, penalty=penalty, verbose=1, max_iter=max_iter)

    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    y_probs = model._predict_proba_lr(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    auc, eer = get_metrics(prediction=y_probs[:, 1], label=y_test)

    predictions, truth = visulize_results(model, frame_size=frame_size, frame_shift=frame_shift)

    experiment_data = {
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 
    'Feature_Args': "----------Feature Args----------", 
    'frame_length': frame_length, 
    'frame_step': frame_step, 
    'num_subbands': data['num_subbands'], 
    'alpha': data['alpha'], 
    'smooth': data['smooth'], 
    'sampling_rate': data['sampling_rate'], 
    'smooth_type': data['smooth_type'], 
    'window': data['window'], 
    'Performance': "----------Performance----------", 
    'accuracy': accuracy, 
    'auc': auc, 
    'eer': eer, 
    'precision': precision, 
    'recall': recall, 
    'f1_score': f1, 
    'confusion_matrix': cm, 
    'Args': "----------SVM Args----------", 
    'kernel': kernel, 
    'C': C, 
    'max_iter': max_iter, 
    }

    # 保存到 JSON 文件
    # log_dir = 'task1/logs'
    # log_file = os.path.join(log_dir, 'SVM_counting_on_frame_size_and_shift_experiment_log.json')
    # with open(log_file, 'a') as f:
    #     json.dump(experiment_data, f, indent=4, cls=NumpyEncoder)
    #     f.write('\n')  # 每次日志记录一行

    # 保存模型
    # file_path = f'task1/models/Linear_SVM/Linear_SVM_for_frame_length_{frame_length}_step_{frame_step}.pkl'
    # joblib.dump(model, file_path)
    # print(f"Model saved to {file_path}")

    return accuracy, auc, eer, predictions, truth, y_test, y_probs



if __name__ =='__main__':

    frame_size_list = [float(0.020), float(0.032), float(0.064), float(0.128), float(0.256), float(0.020), float(0.032), float(0.064), float(0.128), float(0.256)]
    frame_shift_list = [float(0.005), float(0.008), float(0.016), float(0.032), float(0.064), float(0.010), float(0.016), float(0.032), float(0.064), float(0.128)]


    all_auc = []
    all_eer = []
    all_acc = []
    all_predtions = []
    all_truth = []
    all_label = []
    all_prob = []

    kernel = 'linear'

    for frame_size, frame_shift in tqdm(zip(frame_size_list, frame_shift_list), desc="Processing different pairs", unit="pair"):
        accuracy, auc, eer, predictions, truth, y_test, y_prob = Comparision_on_frame_size_and_shift_SVM(frame_size=frame_size, frame_shift=frame_shift, kernel = kernel)
        all_auc.append(auc)
        all_acc.append(accuracy)
        all_eer.append(eer)
        all_predtions.append(predictions)
        all_truth.append(truth)
        all_label.append(y_test)
        all_prob.append(y_prob)


    # 指标曲线
    # frame_sizes = np.array([1000 * x for x in frame_size_list])
    # frame_sizes = np.log2(frame_sizes)

    # plt.figure()
    # plt.plot(frame_sizes[:5], all_auc[:5])
    # plt.plot(frame_sizes[:5], all_acc[:5])
    # plt.plot(frame_sizes[:5], all_eer[:5])
    # plt.xlabel('frame size(log2) / ms')
    # plt.legend(['auc', 'acc', 'eer'])
    # plt.title(f'frame size : frame shift = 4 (SVM kernel={kernel})')
    # plt.savefig('task1/figs/SVM_ratio_4.png')
    # plt.close()

    # plt.figure()
    # plt.plot(frame_sizes[5:], all_auc[5:])
    # plt.plot(frame_sizes[5:], all_acc[5:])
    # plt.plot(frame_sizes[5:], all_eer[5:])
    # plt.xlabel('frame size(log2) / ms')
    # plt.legend(['auc', 'acc', 'eer'])
    # plt.title(f'frame size : frame shift = 2 (SVM kernel={kernel})')
    # plt.savefig('task1/figs/SVM_ratio_2.png')

    # 时域划分
    # audio_path = 'vad/wavs/dev/54-121080-0009.wav'

    # audio, sampling_rate = librosa.load(audio_path, sr = None, )

    # fig, axes = plt.subplots(5, 2, figsize=(10, 24)) 
    # axes = axes.flatten() 

    # labels = ['Preds', 'Truth', 'Audio']

    # for i, preds in enumerate(all_predtions):
    #     preds = preds * 0.2
    #     truth = all_truth[i] * 0.3
    #     axes[i].plot(preds, linewidth=0.3, label='Predictions') 
    #     axes[i].plot(truth, linewidth=0.3, label='Truth')
    #     axes[i].plot(audio, linewidth=0.5, label='Audio')
    #     axes[i].set_xlabel('Samples')
    #     axes[i].set_ylabel('Amplitude')
    #     axes[i].set_title(f'Frame Length:{int(16000 * frame_size_list[i])} Frame Step:{int(16000 * frame_shift_list[i])}', fontsize=10)  # 给每个子图加标题

    # handle, label = axes[0].get_legend_handles_labels()
    # fig.legend(handle, label, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 0.07))

    # plt.tight_layout()
    # plt.subplots_adjust(hspace=0.6, wspace=0.2, top=0.95, bottom=0.1)
    # plt.show()

    fig, axes = plt.subplots(5, 2, figsize=(6, 25))  # 设置总图大小
    axes = axes.flatten()

    for i, prob in enumerate(all_prob):
        y_test = all_label[i]

        fpr, tpr, thresholds = roc_curve(y_true=y_test, y_score=prob[:, 1], pos_label=1)

        axes[i].set_aspect('equal')
        axes[i].set_xlim(0, 1)
        axes[i].set_ylim(0, 1)

        axes[i].plot(fpr, tpr, color='blue', lw=0.5, label='ROC curve')
        axes[i].plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random classifier', linewidth=0.5)

        axes[i].set_xlabel('False Positive Rate')
        axes[i].set_ylabel('True Positive Rate')
        axes[i].set_title(f'Frame Length:{int(16000 * frame_size_list[i])} Frame Step:{int(16000 * frame_shift_list[i])}', fontsize=10)

    handles, labels = axes[0].get_legend_handles_labels()

    plt.tight_layout(rect=[0, 0, 1, 0.95]) 
    plt.subplots_adjust(hspace=0.1, wspace=0.1) 

    plt.show()
