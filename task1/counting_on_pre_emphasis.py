import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

def load_audio(filename):
    signal, sr = librosa.load(filename, sr=None)
    return signal, sr

def pre_emphasis(signal, alpha=0.95):
    return np.append(signal[0], signal[1:] - alpha * signal[:-1])

def plot_waveform(signal, ax, title="Signal"):
    x = range(128 * 11, 128 * 11 + 512)
    ax.plot(x, signal)
    ax.set_title(title)
    ax.set_xlabel("Index")
    ax.set_ylabel("Amplitude")
    ax.grid(True)

def plot_spectrum(signal, sr, ax, title="Spectrum"):
    N = len(signal)
    T = 1.0 / sr
    x = np.linspace(0.0, N * T, N, endpoint=False)
    yf = fft(signal)
    xf = fftfreq(N, T)[:N // 2]
    ax.plot(xf, 2.0 / N * np.abs(yf[:N // 2]))
    ax.set_title(title)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.grid(True)

def experiment(filename, alpha_values=[0.9, 0.97, 1.0]):
    signal, sr = load_audio(filename)
    signal = signal[128 * 11: 128 * 11 + 512]

    fig, axs = plt.subplots(2, 4, figsize=(20, 10))
    
    plot_waveform(signal, axs[0, 0], title="No Pre-emphasis Signal")
    plot_spectrum(signal, sr, axs[1, 0], title="No Pre-emphasis Spectrum")

    for i, alpha in enumerate(alpha_values):
        pre_emphasized_signal = pre_emphasis(signal, alpha)
        plot_waveform(pre_emphasized_signal, axs[0, i+1], title=f"Pre-emphasized Signal (alpha={alpha})")
        plot_spectrum(pre_emphasized_signal, sr, axs[1, i+1], title=f"Pre-emphasized Spectrum (alpha={alpha})")

    plt.tight_layout()
    plt.show()


experiment("vad/wavs/dev/54-121080-0009.wav", alpha_values=[0.5, 0.9, 1.0])
