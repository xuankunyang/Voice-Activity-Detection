import matplotlib.pyplot as plt
import numpy as np
from extracting_short_time_features import fundamental_frequency, enframe
import librosa

def plot_f0_comparison(f0_raw, kernel_sizes, window_sizes, cutoffs, frames, frame_length, frame_step, sr):
    """
    绘制基频对比图，包括不同参数的子图
    """
    fig, axs = plt.subplots(3, 4, figsize=(20, 15))
    axs = axs.ravel()

    for i, kernel_size in enumerate(kernel_sizes):
        f0_medfilt = fundamental_frequency(frames, frame_length, frame_step, sr, smooth=True, smooth_type='medfilt', kernel_size=kernel_size)
        axs[i].plot(f0_raw, label='Raw F0', color='gray', alpha=0.5)
        axs[i].plot(f0_medfilt, label=f'MedFilt', color='b')
        axs[i].set_title(f'Median Filter (kernel_size={kernel_size})')
        axs[i].set_xlabel('Frame Index')
        axs[i].set_ylabel('F0 (Hz)')

    for i, window_size in enumerate(window_sizes):
        f0_convolve = fundamental_frequency(frames, frame_length, frame_step, sr, smooth=True, smooth_type='convolve', window_size=window_size)
        axs[i + 4].plot(f0_raw, label='Raw F0', color='gray', alpha=0.5)
        axs[i + 4].plot(f0_convolve, label=f'Convolve', color='g')
        axs[i + 4].set_title(f'Convolve Filter (window_size={window_size})')
        axs[i + 4].set_xlabel('Frame Index')
        axs[i + 4].set_ylabel('F0 (Hz)')

    for i, cutoff in enumerate(cutoffs):
        f0_fir = fundamental_frequency(frames, frame_length, frame_step, sr, smooth=True, smooth_type='FIR', cutoff=cutoff, numtaps=51)
        axs[i + 8].plot(f0_raw, label='Raw F0', color='gray', alpha=0.5)
        axs[i + 8].plot(f0_fir, label=f'FIR', color='r')
        axs[i + 8].set_title(f'FIR Filter (cut off={cutoff})')
        axs[i + 8].set_xlabel('Frame Index')
        axs[i + 8].set_ylabel('F0 (Hz)')

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4, wspace=0.2, top=0.95, bottom=0.11, left=0.05, right=0.95)

    handles, labels = [], []
    for i in range(0, 12, 4):
        handle, label = axs[i].get_legend_handles_labels()
        handles.extend(handle)
        labels.extend(label)

    handles = list(set(handles))
    labels = list(set(labels))

    plt.legend(handles, labels, loc='upper center', bbox_to_anchor=(-1.285, -0.2), title="Smooth Type", ncol=4)
    plt.show()

def main(audio_file):
    audio, sr = librosa.load(audio_file, sr=None)
    
    frame_length = 512
    frame_step = 128
    frames = enframe(audio, frame_length=frame_length, frame_step=frame_step)
    
    f0_raw = fundamental_frequency(frames, frame_length, frame_step, sr, smooth=False, t_max=0.01, t_min=0.003)
    
    kernel_sizes = [3, 5, 7, 9]
    
    window_sizes = [3, 5, 7, 9]
    
    cutoffs = [100, 500, 1000, 5000]
    
    plot_f0_comparison(f0_raw, kernel_sizes, window_sizes, cutoffs, frames, frame_length, frame_step, sr)

if __name__ == "__main__":
    audio_file = "vad/wavs/dev/54-121080-0009.wav"
    main(audio_file)
