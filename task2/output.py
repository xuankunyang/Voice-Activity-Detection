import torch
from vad_utils import prediction_to_vad_label
from DNN import DNN
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from tqdm import tqdm
from audio_file_process import process_audio_file_test
import librosa
import matplotlib.pyplot as plt
from visualize_results import visulize_results

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

def output(frame_size: float, frame_shift: float):
    input_dim = 69  # 特征维度

    frame_length = int(frame_size * 16000)
    frame_step = int(frame_shift * 16000)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on device: {device}")

    # 创建模型对象并加载训练好的权重
    model = DNN(input_dim).to(device)
    model_path = f'task2/models/DNN_for_frame_length_{frame_length}_step_{frame_step}.pth'
    model.load_state_dict(torch.load(model_path))
    model.eval()  # 设置为评估模式

    audio_dir = Path('vad/wavs/First_10_wavs')
    output_file = f'task2/outputs/first_10_vad_predictions_length_{frame_length}_step_{frame_step}.txt'

    with open(output_file, 'w') as f_out:
        audio_files = list(audio_dir.glob("*.wav"))

        for audio_file in tqdm(audio_files, desc="Processing audio files", unit="file"):
            # 提取音频特征
            features = process_audio_file_test(audio_path=str(audio_file), 
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

            # 前向传播
            with torch.no_grad():
                inputs = torch.tensor(X, dtype=torch.float32).to(device)  # 转换为 Tensor 并发送到设备
                outputs = model(inputs)  # 获取模型的输出

                # 转换模型输出为 VAD 标签
                vad_labels = prediction_to_vad_label(outputs.cpu().numpy(), frame_size=frame_size, frame_shift=frame_shift)

                # 写入文件，每行格式为：音频文件名 + VAD 标签
                file_name = audio_file.stem  # 获取音频文件的名字，不带扩展名
                f_out.write(f"{file_name} {vad_labels}\n")
                
    print(f"Predictions saved to {output_file}")

    predictions, truth = visulize_results(model, frame_size=frame_size, frame_shift=frame_shift)

    return predictions, truth

if __name__ == '__main__':

    # frame_size_list = [float(0.020), float(0.032), float(0.064), float(0.128), float(0.256), float(0.020), float(0.032), float(0.064), float(0.128), float(0.256)]
    # frame_shift_list = [float(0.005), float(0.008), float(0.016), float(0.032), float(0.064), float(0.010), float(0.016), float(0.032), float(0.064), float(0.128)]
    
    frame_size_list = [float(0.032)]
    frame_shift_list = [float(0.008)]

    # frame_size_list = [float(0.020), float(0.064), float(0.128), float(0.256), float(0.020), float(0.032), float(0.064), float(0.128), float(0.256)]
    # frame_shift_list = [float(0.005), float(0.016), float(0.032), float(0.064), float(0.010), float(0.016), float(0.032), float(0.064), float(0.128)]

    all_preds = []
    all_truth = []

    for frame_size, frame_shift in tqdm(zip(frame_size_list, frame_shift_list), desc="Processing different pairs", unit="pair"):
        predictions, truth = output(frame_size=frame_size, frame_shift=frame_shift)
        all_preds.append(predictions)
        all_truth.append(truth)
        

    audio_path = 'vad/wavs/dev/54-121080-0009.wav'

    audio, sampling_rate = librosa.load(audio_path, sr = None, )

    fig, axes = plt.subplots(5, 2, figsize=(10, 24)) 
    axes = axes.flatten() 

    labels = ['Preds', 'Truth', 'Audio']

    for i, preds in enumerate(all_preds):
        preds = preds * 0.2
        truth = all_truth[i] * 0.3
        axes[i].plot(preds, linewidth=0.3, label='Predictions') 
        axes[i].plot(truth, linewidth=0.3, label='Truth')
        axes[i].plot(audio, linewidth=0.5, label='Audio')
        axes[i].set_xlabel('Samples')
        axes[i].set_ylabel('Amplitude')
        axes[i].set_title(f'Frame Length:{int(16000 * frame_size_list[i])} Frame Step:{int(16000 * frame_shift_list[i])}', fontsize=10)  # 给每个子图加标题


    handle, label = axes[0].get_legend_handles_labels()
    fig.legend(handle, label, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 0.07))

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.6, wspace=0.2, top=0.95, bottom=0.1)
    # plt.show()