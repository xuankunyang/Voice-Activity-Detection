import matplotlib.pyplot as plt
import numpy as np
from extracting_short_time_features import short_time_Fourier_transform, short_time_spectral_mean, enframe
import librosa

def plot_visualizations(frames, stft, audio):
    """
    Create visualizations for the speech signal waveform and short-time spectral mean.

    Args:
        frames: input speech frames (n, d)
        stft: result of Short-Time Fourier Transform (n, d)
    """
    stft_mean = short_time_spectral_mean(stft)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6))

    frame_index = 100 
    frame = frames[frame_index]

    frame_spectrum_mean = stft_mean[frame_index]

    axes[0].plot(np.arange(len(stft[frame_index])), np.abs(stft[frame_index]), color='b', label='Spectrum')

    axes[0].axhline(y=frame_spectrum_mean, color='r', linestyle='--', label=f'Spectral Mean: {frame_spectrum_mean:.2f}')

    axes[0].set_title(f'Spectrum and Spectral Mean for Frame {frame_index}')
    axes[0].set_xlabel('Frequency Components')
    axes[0].set_ylabel('Magnitude')
    axes[0].legend()


    # Subplot 1.2: Short-Time Spectral Mean per Frame
    axes[1].plot(np.arange(stft_mean.shape[0]), stft_mean, color='r')
    axes[1].set_title('Short-Time Spectral Mean per Frame')
    axes[1].set_xlabel('Frame Index')
    axes[1].set_ylabel('Spectral Mean')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    audio, sr = librosa.load('vad/wavs/dev/54-121080-0009.wav', sr=None) 
    
    frame_length = 512 
    frame_step = 128 
    frames = enframe(audio, frame_length=frame_length, frame_step=frame_step)

    stft = short_time_Fourier_transform(frames)

    plot_visualizations(frames, stft, audio)