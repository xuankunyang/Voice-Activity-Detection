import numpy as np
from pathlib import Path
import librosa
from sklearn import metrics
from vad.evaluate import get_metrics
from vad.vad_utils import prediction_to_vad_label
from vad.vad_utils import read_label_from_file
from rich.progress import track
import tqdm

class VADClassifier_1:
    """
    基于线性分类器和语音短时信号特征
    """
    def __init__(self, energy_threshold: float = 0.1, zcr_threshold: float = 0.1, 
                 frame_size: float = 0.032, frame_shift: float = 0.008):
        self.energy_threshold = energy_threshold
        self.zcr_threshold = zcr_threshold
        self.frame_size = frame_size
        self.frame_shift = frame_shift

        
        
    def extract_short_time_features(self, audio: np.ndarray, sampling_rate: int):
        """
        短时信号特征提取
        """
        def short_time_energy(frame):
            return np.sum(frame ** 2) / len(frame)
        def short_time_zero_crossing_rate(frame):
            signs = np.sign(frame)
            signs_diff = np.diff(signs)
            signs_diff_abs = np.abs(signs_diff)
            return np.sum(signs_diff_abs) / (2 * len(frame))
        
        # 这里需要注意，librosa.util.frame分帧需要的参数单位为样本数，这里对应转化一下
        frame_length = int(self.frame_size * sampling_rate)
        frame_step = int(self.frame_shift * sampling_rate)

        frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=frame_step)

        energies = np.array([short_time_energy(frame=f) for f in frames])
        zcrs = np.array([short_time_zero_crossing_rate(frame=f) for f in frames])

        return energies, zcrs
    
    def process_audio_file(self, label_dict: dict, audio_path):
        """
        对单个语音文件进行特征提取以及获取真实标签
        """
        audio, sampling_rate = librosa.load(audio_path, sr = None) # 注意此处使用数据原始采样率
        energies, zcrs = self.extract_short_time_features(audio=audio, sampling_rate=sampling_rate)
        audio_key = Path(audio_path).stem
        labels = label_dict.get(audio_key, []) # 如果没找到，那就返回空list
        
        # 注意，如vad_utils中的note提到的，真实标签最后是可能需要补0的
        if len(labels) > 0:
            num_frame = len(energies)
            labels = np.pad(labels, (0, max(num_frame - len(labels), 0)))[:num_frame]

        return {'energies': energies, 'zcrs': zcrs, 'labels': labels}
    
    def predict(self, energies, zcrs):
        """
        简单的线性分类器
        """
        predictions = []
        for e, z in zip(energies, zcrs):
            if e > self.energy_threshold and z > self.zcr_threshold:
                predictions.append(1)
            else:
                predictions.append(0)
        return predictions
    
    def tune_thresholds(self, energies, zcrs, labels, lower_bound: float = 0.00001, upper_bound: float = 0.0005, steps: int = 10, criteira: str = 'ERR'):
        """
        使用线性搜索，调整阈值
        """
        best_eer = float('inf')
        best_auc = float(0)
        best_e_th = self.energy_threshold
        best_zcr_th = self.zcr_threshold

        for e_th in track(np.linspace(upper_bound, lower_bound, steps), description="进度..."):
            for zcr_th in np.linspace(upper_bound, lower_bound, steps):
                self.energy_threshold = e_th
                self.zcr_threshold = zcr_th
                predictions = self.predict(energies=energies, zcrs=zcrs)
                auc, eer = get_metrics(prediction=predictions, label=labels)

                if criteira == 'ERR' and eer < best_eer:
                    best_eer = eer
                    best_e_th = e_th
                    best_zcr_th = zcr_th
                if criteira == 'AUC' and auc > best_auc:
                    best_auc = auc
                    best_e_th = e_th
                    best_zcr_th = zcr_th

        self.energy_threshold = best_e_th
        self.zcr_threshold = best_zcr_th
        if criteira == 'AUC':
            print(f"Best thresholds - Energy: {best_e_th:.5f}, ZCR: {best_zcr_th:.5f}, AUC: {best_auc:.5f}")
        if criteira == 'AUC':
            print(f"Best thresholds - Energy: {best_e_th:.5f}, ZCR: {best_zcr_th:.5f}, ERR: {best_eer:.5f}")

def main():
    vad_classifier = VADClassifier_1(0.0005, 0.0005)

    print(f"Initial energy_threshold: {vad_classifier.energy_threshold}")
    print(f"Initial zcr_threshold: {vad_classifier.zcr_threshold}")

    train_audio_dir = Path("voice-activity-detection-sjtu-spring-2024\\vad\\wavs\\dev")
    train_label_path = "voice-activity-detection-sjtu-spring-2024\\vad\\data\\dev_label.txt"

    labels_dict = read_label_from_file(train_label_path, vad_classifier.frame_size, vad_classifier.frame_shift)

    all_energies = []
    all_zcr = []
    all_labels = []

    for audio_file in train_audio_dir.glob("*.wav"):
        result = vad_classifier.process_audio_file(label_dict=labels_dict, audio_path=str(audio_file))

        if len(result['labels'] > 0):
            all_energies.extend(result['energies'])
            all_labels.extend(result['labels'])
            all_zcr.extend(result['zcrs'])

    if all_zcr and all_energies and all_labels:
        vad_classifier.tune_thresholds(energies=all_energies,zcrs=all_zcr, labels=all_labels, steps=10)


    test_audio_dir = Path("voice-activity-detection-sjtu-spring-2024\\vad\\wavs\\dev")
    test_label_path = "voice-activity-detection-sjtu-spring-2024\\vad\\data\\dev_label.txt"
    
    test_predictions = []
    test_labels = []
    
    for audio_file in test_audio_dir.glob("*.wav"):
        result = vad_classifier.process_audio_file(label_dict=labels_dict, audio_path=str(audio_file))
        
        if len(result['labels']) > 0:
            predictions = vad_classifier.predict(result['energies'], result['zcrs'])
            test_predictions.extend(predictions)
            test_labels.extend(result['labels'])
    
    if test_predictions and test_labels:
        auc, eer = get_metrics(test_predictions, test_labels)
        print(f"Final Test Metrics - AUC: {auc:.5f}, EER: {eer:.5f}")
        
        vad_label = prediction_to_vad_label(test_predictions)
        # print(f"Predicted VAD labels: {vad_label}")

if __name__ == "__main__":
    main()
        


        

            

