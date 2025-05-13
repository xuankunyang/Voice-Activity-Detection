import matplotlib.pyplot as plt
import numpy as np
import librosa
from extracting_short_time_features import short_time_Fourier_transform, Filter_banks, Mfcc,weighted_difference, weighted_difference_of_delta
from audio_file_process import enframe

audio_file = "vad/wavs/dev/54-121080-0009.wav"

audio, sr = librosa.load(audio_file, sr=None) 
    

frame_length = 512 
frame_step = 128 
frames = enframe(audio, frame_length=frame_length, frame_step=frame_step)

stft = short_time_Fourier_transform(frames=frames, window_type='hanning')

fbanks = Filter_banks(stft, sr, frame_lenth=frame_length, num_filters=23)

mfcc = Mfcc(fbanks, num_mfcc=12)

librosa_mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, n_fft=512, hop_length=128, n_mels=23, window = 'hann')

plt.figure(figsize=(10, 6))
plt.imshow(librosa_mfcc, aspect='auto', origin='lower', cmap='jet')
plt.colorbar(format="%+2.0f")
plt.title("MFCC(librosa)")
plt.xlabel("Frame index")
plt.ylabel("MFCC Coefficient")
plt.show()


plt.figure(figsize=(10, 6))
plt.imshow(mfcc.T, aspect='auto', origin='lower', cmap='jet')
plt.colorbar(format="%+2.0f")
plt.title("MFCC")
plt.xlabel("Frame index")
plt.ylabel("MFCC Coefficient")
plt.show()

delta_mfcc = weighted_difference(mfcc, delta=1)

plt.figure(figsize=(10, 6))
plt.imshow(delta_mfcc.T, aspect='auto', origin='lower', cmap='jet')
plt.colorbar(format="%+2.0f")
plt.title("Delta MFCC")
plt.xlabel("Frame index")
plt.ylabel("Delta MFCC Coefficient")
plt.show()


delta_delta_mfcc = weighted_difference_of_delta(delta_mfcc, delta=1)

plt.figure(figsize=(10, 6))
plt.imshow(delta_delta_mfcc.T, aspect='auto', origin='lower', cmap='jet')
plt.colorbar(format="%+2.0f")
plt.title("Delta of Delta MFCC")
plt.xlabel("Frame index")
plt.ylabel("Delta of Delta MFCC Coefficient")
plt.show()
