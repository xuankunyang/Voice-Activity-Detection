import matplotlib.pyplot as plt
import numpy as np
from extracting_short_time_features import short_time_Fourier_transform, enframe
import librosa

def plot_subband_energy_multiple(stft, num_subbands_list, frame_length, sampling_rate, frame_idx):
    
    fig, axs = plt.subplots(2, 2, figsize=(18,12))
    
    for idx, num_subbands in enumerate(num_subbands_list):
        ax = axs[idx // 2, idx % 2]
        
        freq_bins = np.linspace(0, sampling_rate // 2, frame_length // 2 + 1)
        subband_limits = np.linspace(0, len(freq_bins) - 1, num_subbands + 1).astype(int)
        colors = plt.cm.viridis(np.linspace(0, 1, num_subbands))
        
        for i in range(num_subbands):
            subband_spectrum = stft[subband_limits[i] : subband_limits[i+1]]
            x = range(subband_limits[i], subband_limits[i+1])
            energy = np.sum(np.abs(subband_spectrum) ** 2)
            ax.plot(x, np.abs(subband_spectrum), color=colors[i], label=f"Subband #{i+1}, Energy: {energy:.4f}")
            
            if i < num_subbands - 1: 
                ax.axvline(x=subband_limits[i+1] - 0.5, color='r', linestyle='--')
        
        ax.set_title(f'Spectrum and Subbands for Frame {frame_idx} (Num Subbands= {num_subbands})',fontsize=16)
        ax.set_xlabel('Frequency Components', fontsize=12)
        ax.set_ylabel('Magnitude', fontsize=12)
        ax.legend()
    
    plt.tight_layout()
    plt.show()


def plot_subband_energy(stft, num_subbands, frame_length, sampling_rate, frame_idx):

    freq_bins = np.linspace(0, sampling_rate // 2, frame_length // 2 +1)
    subband_limits = np.linspace(0, len(freq_bins) - 1, num_subbands + 1).astype(int)
    colors = plt.cm.viridis(np.linspace(0, 1, num_subbands))
    fig = plt.figure(figsize=(10,6))
    for i in range(num_subbands):
        subband_spectrum = stft[subband_limits[i] : subband_limits[i+1]]
        x = range(subband_limits[i], subband_limits[i+1])
        energy = np.sum(np.abs(subband_spectrum) ** 2)
        plt.plot(x, np.abs(subband_spectrum), color=colors[i], label = f"Subband #{i+1}, Energy: {energy:.4f}")
        if i < num_subbands - 1: 
            plt.axvline(x=subband_limits[i+1] - 0.5, color='r', linestyle='--')
    plt.legend()
    plt.title(f'Spectrum and Subbands for Frame {frame_idx}',fontsize=30)
    plt.xlabel('Samples',fontsize=24)
    plt.ylabel('Magnitude',fontsize=24)
    plt.show()

audio, sr = librosa.load('vad/wavs/dev/54-121080-0009.wav', sr=None) 
    
frame_length = 512 
frame_step = 128 
frames = enframe(audio, frame_length=frame_length, frame_step=frame_step)

stft = short_time_Fourier_transform(frames)

num_subbands = 6
frame_idx = 100
sampling_rate = 16000

# plot_subband_energy(stft[frame_idx], num_subbands, frame_length, sampling_rate, frame_idx)

plot_subband_energy_multiple(stft[frame_idx], [4, 6, 8, 10], frame_length, sampling_rate, frame_idx)