import numpy as np
import librosa
from scipy import signal
import math
from scipy.fftpack import dct

def short_time_Fourier_transform(frames, window_type='hanning', overlap: float = 1.):
    """
    Implement STFT (Short Time Fourier Transform)

    Args:
        frames: input speech frames, dimension (n, d), where n is the number of frames and d is the length of each frame
        window_type: the type of window function used, default "hanning"
        overlap: degree of overlap

    Returns:
        X: result of STFT, dimension (n, d), each element is a complex spectrum
    """

    n_frames, frame_length = frames.shape
    win_length = int(frame_length / overlap)

    if window_type == 'rectangular':
        window = np.ones(win_length)
    elif window_type == 'hamming':
        window = np.hamming(win_length)
    elif window_type == 'hanning':
        window = np.hanning(win_length)
    elif window_type == 'blackman':
        window = np.blackman(win_length)
    else:
        raise ValueError("Unknown window type: {}".format(window_type))
    
    stft_result = []
    shift = int((win_length - frame_length) // 2)
    window = window[shift :win_length - shift]

    for i in range(n_frames):
        frame = frames[i] * window  
        spectrum = np.fft.fft(frame)  
        stft_result.append(spectrum[:len(spectrum) // 2 + 1])

    stft_result = np.array(stft_result)

    return stft_result


def short_time_energy(frames):
    """
    短时能量
    """
    return np.sum(frames ** 2, axis=1)


def short_time_zero_crossing_rate(frames):
    """
    短时过零率
    """
    signs = np.sign(frames)
    signs_diff = np.diff(signs,axis=1)
    signs_diff_abs = np.abs(signs_diff)
    return np.sum(signs_diff_abs, axis=1) / 2

def short_time_frequency_spectrum(audio: np.ndarray, frame_length: int, frame_step: int, window: str = 'hann'):
    """
    获取短时频谱
    """
    stft = librosa.stft(audio, n_fft=frame_length, hop_length=frame_step, 
                        frame_lengthgth=frame_length, window=window, center=False)
    return stft

def short_time_spectral_mean(stft: np.ndarray):
    """
    短时频谱均值
    """
    stft_abs = np.abs(stft)
    stft_mean = np.mean(stft_abs, axis=1)
    return stft_mean

def subband_energy(stft: np.ndarray, num_subbands: int, frame_length: int, sampling_rate: int):
    """
    获取短时频谱子带能量
    """
    freq_bins = np.linspace(0, sampling_rate // 2, frame_length // 2 +1)
    subband_limits = np.linspace(0, len(freq_bins) - 1, num_subbands + 1).astype(int)
    
    subband_energies = []
    for i in range(num_subbands):
        subband_spectrum = stft[:, subband_limits[i] : subband_limits[i+1]]
        energy = np.sum(np.abs(subband_spectrum) ** 2, axis=1)
        subband_energies.append(energy)
    return subband_energies

def pitch_detection(frame: object, t_min: float = 0.003, t_max: float = 0.01, sampling_rate: int = 16000):

    corr = np.correlate(frame, frame, mode='same')

    peak_index = np.argmax(corr[int(0.5 * t_min * sampling_rate) : int(1.5 * t_max * sampling_rate)]) + int(0.5 * t_min * sampling_rate)

    f0 = sampling_rate / peak_index
    return f0

def fundamental_frequency(frames: np.ndarray, frame_length: int, frame_step: int, 
                          sampling_rate: int, smooth: bool = 1, smooth_type: str = 'medfilt', 
                          t_min: float = 0.003, t_max:float = 0.01, **kwargs):
    """
    获取基频
    考虑采取不同的平滑方法
    """

    f0 = np.array([pitch_detection(frame, t_min=t_min, t_max=t_max, sampling_rate=sampling_rate) for frame in frames])
    
    if not smooth:
         return f0
    if smooth and smooth_type == 'medfilt':
        kernel_size = kwargs['kernel_size']
        return signal.medfilt(f0, kernel_size=kernel_size)
    if smooth and smooth_type == 'convolve':
        window_size = kwargs['winsow_size']
        return np.convolve(f0, np.ones(window_size)/window_size, mode='same')
    if smooth and smooth_type == 'FIR':
        cutoff = kwargs['cutoff'] # 截止频率 50
        numtaps = kwargs['numtaps'] # 滤波器阶数 51 一般取奇数
        fir_coeff = signal.firwin(numtaps, cutoff, fs=sampling_rate)
        return signal.lfilter(fir_coeff, 1.0, f0)
    return f0

def Filter_banks(stft: np.ndarray, sampling_rate: int = 16000, frame_lenth: int = 512, 
                 num_filters: int = 23):
    filter_banks = []
    mel_filters = librosa.filters.mel(sr=sampling_rate, n_fft=frame_lenth, n_mels=num_filters)

    num_frames = stft.shape[0]

    for i in range(num_frames):
        spectrum = np.abs(stft[i]) ** 2

        f_bank = np.dot(mel_filters, spectrum)

        filter_banks.append(f_bank)

    return np.array(filter_banks)

def Mfcc(f_banks: np.ndarray, num_mfcc: int =12):
    f_banks = np.where(f_banks == 0, np.finfo(float).eps, f_banks) # 数值稳定性

    log_mel = 10 * np.log10(f_banks)

    mfcc = dct(log_mel, type=2, axis=1, norm="ortho")

    mfcc = mfcc[:, :num_mfcc]

    return mfcc

def weighted_difference(mfcc, delta=1):
    """
    计算加权差分动态特征（Delta）
    参数：
        features: MFCC 特征，形状为 (num_frames, num_coeffs)
        delta: 差分的窗口大小，默认 1
    返回：
        delta_features: 加权差分后的 Delta 特征，形状为 (num_frames, num_coeffs)
    """
    num_frames, num_coeffs = mfcc.shape
    delta_features = np.zeros_like(mfcc)
    
    # 加权差分窗口的归一化因子
    weights = np.array([i for i in range(1, delta + 1)])  # 权重从 1 到 delta
    weight_sum = np.sum(weights**2)
    
    for t in range(delta, num_frames - delta):
        for f in range(num_coeffs):
            delta_features[t, f] = np.sum(weights * (mfcc[t + np.arange(1, delta + 1), f] - mfcc[t - np.arange(1, delta + 1), f])) / weight_sum
    
    return delta_features

def weighted_difference_of_delta(delta_mfcc, delta=1):
    """
    计算 Delta 特征的加权差分（即 Delta-Delta）
    参数：
        delta_features: Delta 特征，形状为 (num_frames, num_coeffs)
        delta: 差分的窗口大小，默认 1
    返回：
        delta_delta_features: 加权差分后的 Delta-Delta 特征，形状为 (num_frames, num_coeffs)
    """
    num_frames, num_coeffs = delta_mfcc.shape
    delta_delta_features = np.zeros_like(delta_mfcc)
    
    # 加权差分窗口的归一化因子
    weights = np.array([i for i in range(1, delta + 1)])  # 权重从 1 到 delta
    weight_sum = np.sum(weights**2)
    
    for t in range(delta, num_frames - delta):
        for f in range(num_coeffs):
            delta_delta_features[t, f] = np.sum(weights * (delta_mfcc[t + np.arange(1, delta + 1), f] - delta_mfcc[t - np.arange(1, delta + 1), f])) / weight_sum
    
    return delta_delta_features


def extract_short_time_features(frames: np.ndarray, sampling_rate: int = 16000, 
                                frame_step: int = 128, frame_length: int = 512, 
                                t_min: float = 0.003, t_max: float = 0.01, 
                                num_subbands: int = 6,num_filters:int=23, 
                                num_mfcc:int=23, delta:int=1, 
                                  **kwargs):
        """
        短时信号特征提取

        Args:
            audio(np.ndarray): The processed audio file
            sampling_rate(int): The sampling rate
            frame_size(float): The size of frame we use
            frame_shift(float): The shift of each frame
            num_subbands(int): The number of subbands in subbands analysis

            **kwargs: 
                'smooth'(bool): Take smooth process or not
                'smooth_type'(str): The smooth type
                'window'(str): The window type

        Returns:
            features(Dict): including the key-values as follows:
                'energies': Short time energies
                'zrcs': Short time zero-crossing-rate
                'freq_centers': The center of the short time frequency spectrum
                'subband_energies': Subbands energies
                'f0': Fundamental frequency considering smoothing or not
        """

        energies = np.array(short_time_energy(frames))
        zcrs = np.array(short_time_zero_crossing_rate(frames))
        stft = short_time_Fourier_transform(frames=frames, window_type='hanning', overlap=1.0)
        spectral_mean = short_time_spectral_mean(stft)
        subband_energies = subband_energy(stft, num_subbands, frame_length, sampling_rate)
        f0 = fundamental_frequency(frames=frames, frame_length=frame_length, frame_step=frame_step, 
                                   sampling_rate=sampling_rate, t_min=t_min, t_max=t_max, 
                                   smooth=kwargs['smooth'], smooth_type=kwargs['smooth_type'], 
                                   kernel_size = 5)
        f_banks = Filter_banks(stft, sampling_rate=sampling_rate, frame_lenth=frame_length, num_filters=num_filters)
        mfcc = Mfcc(f_banks=f_banks, num_mfcc=num_mfcc)
        delta_mfcc = weighted_difference(mfcc=mfcc, delta=delta)
        delta_delta_mfcc = weighted_difference_of_delta(delta_mfcc=delta_mfcc, delta=delta)        

        return {
             'energies': energies,
             'zcrs': zcrs,
             'spectral_means': spectral_mean,
             'subband_energies': subband_energies,
             'f0': f0, 
             'f_banks': f_banks, 
             'mfccs': mfcc, 
             'delta_mfccs': delta_mfcc, 
             'delta_delta_mfccs': delta_delta_mfcc
        }