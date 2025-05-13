import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import librosa

file_path = 'vad/wavs/dev/54-121080-0009.wav'
y, sr = librosa.load(file_path, sr=None)
frame_size = 512
hop_size = frame_size // 4


frame_start = 4096 
frame = y[frame_start:frame_start+frame_size]

window_types = ['rectangular', 'hamming', 'hann', 'bartlett', 'blackman']
windows = {
    'rectangular': np.ones(frame_size),
    'hamming': np.hamming(frame_size),
    'hann': np.hanning(frame_size),
    'bartlett': np.bartlett(frame_size),
    'blackman': np.blackman(frame_size)
}

window_colors = {
    'rectangular': 'b',  
    'hamming': 'g',      
    'hann': 'r',         
    'bartlett': 'c',     
    'blackman': 'm'      
}

fig, axs = plt.subplots(len(window_types), 6, figsize=(25, len(window_types)*6))

frame_sizes = [frame_size, int(frame_size * 1.2), int(frame_size * 1.5)]

for i, window_type in enumerate(window_types):
    
    for j, size in enumerate(frame_sizes):
        if i == 0:
            window = np.ones(size)
        elif i == 1:
            window = np.hamming(size)
        elif i == 2:
            window = np.hanning(size)
        elif i == 3:
            window = np.bartlett(size)
        else:
            window = np.blackman(size)


        shift = (size - frame_size) // 2
        current_window = window[shift :size - shift]
         
        # current_window_shifted = np.roll(current_window, shift)

        windowed_signal = frame * current_window
        
        spectrum = np.fft.fft(windowed_signal)
        freq = np.fft.fftfreq(len(windowed_signal), d=1/sr)
        spectrum_magnitude = np.abs(spectrum)[:frame_size//2]
        freq = freq[:frame_size//2]
        
        axs[i, j].plot(np.arange(frame_size), windowed_signal, color=window_colors[window_type], label=window_type)
        axs[i, j].set_title(f' Window Length={size}\n - Time Domain', fontsize=10)
        axs[i, j].set_xlabel('Samples', fontsize=8)
        axs[i, j].set_ylabel('Amplitude', fontsize=8)
        
        axs[i, j+3].plot(freq, spectrum_magnitude, color=window_colors[window_type], label=window_type)
        axs[i, j+3].set_title(f'Window Length={size}\n - Frequency Domain', fontsize=10)
        axs[i, j+3].set_xlabel('Frequency (Hz)', fontsize=8)
        axs[i, j+3].set_ylabel('Magnitude', fontsize=8)

handles, labels = [], []
for i in range(5):
    handle, label = axs[i, 0].get_legend_handles_labels()
    handles.extend(handle)
    labels.extend(label)
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.07), title="Window Type", ncol=5)

plt.subplots_adjust(hspace=0.95, wspace=0.5, top=0.95, bottom=0.11, left=0.05, right=0.95)

plt.show()
