import matplotlib.pyplot as plt
import numpy as np
from extracting_short_time_features import short_time_energy, enframe
import librosa

def plot_visualizations(frames, audio):
    """
    Create visualizations for the speech signal waveform and short-time spectral mean.

    Args:
        frames: input speech frames (n, d)
        stft: result of Short-Time Fourier Transform (n, d)
    """
    energies = short_time_energy(frames=frames)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6))

    frame_index = 100  
    frame = frames[frame_index]

    axes[0].plot(np.arange(len(frame)), frames[frame_index], color='b', label='Waveform')

    axes[0].set_title(f'Waveform and Energy for Frame {frame_index}')
    axes[0].set_xlabel('Samples')
    axes[0].set_ylabel('Amplitude')
    axes[0].text(0.98, 0.95, f'Energy for Frame {frame_index}: {energies[frame_index]}', 
             ha='right', va='top', transform=axes[0].transAxes,
             color='red', 
             bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.5', linewidth=1))
    
    axes[1].plot(np.arange(frames.shape[0]), energies, color='r')
    axes[1].set_title('Energy per Frame')
    axes[1].set_xlabel('Frame Index')
    axes[1].set_ylabel('Energy')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    audio, sr = librosa.load('vad/wavs/dev/54-121080-0009.wav', sr=None) 
    
    frame_length = 512 
    frame_step = 128 
    frames = enframe(audio, frame_length=frame_length, frame_step=frame_step)

    plot_visualizations(frames, audio)