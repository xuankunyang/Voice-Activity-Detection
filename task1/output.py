import numpy as np
from sklearn.svm import LinearSVC
import joblib
from pathlib import Path
from tqdm import tqdm
from audio_file_process import process_audio_file_test
from vad_utils import prediction_to_vad_label
from sklearn.preprocessing import StandardScaler


num_subbands = int(6)
smooth = bool(1)
smooth_type = str('medfilt')
pre_emphasis = bool(1)
alpha = float(0.97)
sampling_rate = int(16000)
overlap = float(1.0)
window = 'hanning'


def output(frame_size: float, frame_shift: float, model_type:str):

    frame_length = int(frame_size * 16000)
    frame_step = int(frame_shift * 16000)

    model_path = f'task1\models\{model_type}\{model_type}_for_frame_length_{frame_length}_step_{frame_step}.pkl'
    model = joblib.load(model_path)

    audio_dir = Path('vad/wavs/test')
    output_file = f'task1/outputs/{model_type}/vad_predictions_length_{frame_length}_step_{frame_step}.txt'

    with open(output_file, 'w') as f_out:
        audio_files = list(audio_dir.glob("*.wav"))

        for audio_file in tqdm(audio_files, desc="Processing audio files", unit="file"):
            features = process_audio_file_test(audio_path=str(audio_file), 
                                                frame_length=frame_length, frame_step=frame_step, 
                                                num_subbands=num_subbands, alpha=alpha, pre_emphasis=pre_emphasis, 
                                                smooth=smooth, sampling_rate=sampling_rate, 
                                                smooth_type=smooth_type, window=window, overlap=overlap,
                                                )

            all_energies = features['energies']
            all_spectral_means = features['spectral_means']
            all_zcr = features['zcrs']
            all_subband_energies = [features['subband_energies'][i] for i in range(num_subbands)]
            all_f0 = features['f0']

            X = [all_energies, all_zcr, all_spectral_means, all_f0] + all_subband_energies

            X = np.array(X).T

            scaler = StandardScaler()
            X = scaler.fit_transform(X)

            if model_type == 'Linear_SVM':
                outputs = model._predict_proba_lr(X)
                outputs = outputs[:, 1]
            elif model_type == 'Logistic_Regression':
                outputs = model.predict_proba(X)
                outputs = outputs[:, 1]

            vad_labels = prediction_to_vad_label(outputs, frame_size=frame_size, frame_shift=frame_shift)

            file_name = audio_file.stem  # 获取音频文件的名字，不带扩展名
            f_out.write(f"{file_name} {vad_labels}\n")
                
    print(f"Predictions saved to {output_file}")

            

if __name__ == '__main__':

    frame_size_list = [float(0.020), float(0.032), float(0.064), float(0.128), float(0.256), float(0.020), float(0.032), float(0.064), float(0.128), float(0.256)]
    frame_shift_list = [float(0.005), float(0.008), float(0.016), float(0.032), float(0.064), float(0.010), float(0.016), float(0.032), float(0.064), float(0.128)]
    
    # frame_size_list = [float(0.032)]
    # frame_shift_list = [float(0.008)]

    # frame_size_list = [float(0.020), float(0.064), float(0.128), float(0.256), float(0.020), float(0.032), float(0.064), float(0.128), float(0.256)]
    # frame_shift_list = [float(0.005), float(0.016), float(0.032), float(0.064), float(0.010), float(0.016), float(0.032), float(0.064), float(0.128)]

    all_preds = []
    all_truth = []

    for frame_size, frame_shift in tqdm(zip(frame_size_list, frame_shift_list), desc="Processing different pairs", unit="pair"):
        output(frame_size=frame_size, frame_shift=frame_shift, model_type='Linear_SVM')
        output(frame_size=frame_size, frame_shift=frame_shift, model_type='Logistic_Regression')
