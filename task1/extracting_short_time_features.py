import numpy as np
import librosa
from scipy import signal
import math

def enframe(x, frame_length, frame_step, ceil_or_floor: str = 'ceil'):
    """
        Function:
            Receives a 1D numpy array and divides it into frames.
            Outputs a numpy matrix with the frames on the rows.

        Args:
            x(np.ndarray): The audio data
            frame_lenth(int): The frame_length
            frame_step(int): The frame step
            ceil_or_floor(str): Using ceil or floor to count the number of frames

        Returns:
            a numpy matrix with the frames on the rows.

        上取整，下取整要注意分别
    """
    x = np.squeeze(x)
    if x.ndim != 1:
        raise TypeError("enframe input must be a 1-dimensional array.")
    
    if ceil_or_floor == 'ceil':
        n_frames = 1 + int(math.ceil((len(x) - frame_length) / float(frame_step)))
    elif ceil_or_floor == 'floor':
        n_frames = 1 + int(math.floor((len(x) - frame_length) / float(frame_step)))
    else: 
         raise TypeError("Please choose one in 'ceil' or 'floor'.")


    x_framed = np.zeros((n_frames, frame_length))

    padlen = int((n_frames - 1) * frame_step + frame_length)

    zeros = np.zeros((max(0 ,padlen - len(x)),))
    padsignal = np.concatenate((x, zeros))
    for i in range(n_frames):
        x_framed[i] = padsignal[i * frame_step : i * frame_step + frame_length]
    return x_framed

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
        window_size = kwargs['window_size']
        return np.convolve(f0, np.ones(window_size)/window_size, mode='same')
    if smooth and smooth_type == 'FIR':
        cutoff = kwargs['cutoff'] # 截止频率 50
        numtaps = kwargs['numtaps'] # 滤波器阶数 51 一般取奇数
        fir_coeff = signal.firwin(numtaps, cutoff, fs=sampling_rate)
        return signal.lfilter(fir_coeff, 1.0, f0)
    return f0



def fundamental_frequency_1(audio: np.ndarray, frame_length: int, frame_step: int, 
                            sampling_rate: int, smooth: bool = 1, smooth_type: str = 'medfilt', **kwargs):
    """
    获取基频
    考虑采取不同的平滑方法
    """
    f0, voiced_flag, voiced_probs = librosa.pyin(audio, 
                                                 fmin=librosa.note_to_hz('C1'), fmax=librosa.note_to_hz('C7'), 
                                                 sr=sampling_rate, center=False,
                                                 frame_length=frame_length, hop_length=frame_step)
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


def extract_short_time_features(audio: np.ndarray, sampling_rate: int = 16000, 
                                frame_step: int = 128, frame_length: int = 512, 
                                t_min: float = 0.003, t_max: float = 0.01, 
                                num_subbands: int = 6, **kwargs):
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

        frames = enframe(audio, frame_length=frame_length, frame_step=frame_step)

        energies = np.array(short_time_energy(frames))
        zcrs = np.array(short_time_zero_crossing_rate(frames))
        stft = short_time_Fourier_transform(frames=frames, window_type='hanning', overlap=1.0)
        spectral_mean = short_time_spectral_mean(stft)
        subband_energies = subband_energy(stft, num_subbands, frame_length, sampling_rate)
        f0 = fundamental_frequency(frames=frames, frame_length=frame_length, frame_step=frame_step, 
                                   sampling_rate=sampling_rate, t_min=t_min, t_max=t_max, 
                                   smooth=kwargs['smooth'], smooth_type=kwargs['smooth_type'], 
                                   kernel_size = 5)        

        return {
             'energies': energies,
             'zcrs': zcrs,
             'spectral_means': spectral_mean,
             'subband_energies': subband_energies,
             'f0': f0
        }