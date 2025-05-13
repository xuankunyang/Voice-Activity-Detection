import numpy as np
from pathlib import Path
import librosa
from extracting_short_time_features import extract_short_time_features

def pre_emphasize(x, alpha=0.97):
    """
    一阶差分滤波器
    """
    return np.append(x[0], x[1:] - alpha * x[:-1])

def process_audio_file_traning(label_dict: dict, audio_path: str, pre_emphasis: bool = 1, 
                               frame_length: int = 512, frame_step: int = 128, 
                               t_min: float = 0.003, t_max: float = 0.01, 
                               sampling_rate:int = 16000, num_subbands: int = 6, **kwargs): 
    """
    对单个训练文件进行特征提取以及获取真实标签

    Args:
        label_dict(dict): The true label of all training data
        audio_path(str): The path of one exact audio file
        pre_emphasis(bool): Make pre_emphasis or not
        frame_length(int): The frame length
        frame_step(int): The frame step
        t_min(float): The lower bound of the fundamental period
        t_max(float): The upper bound of the fundamental period
        sampling_rate(int): The sampling rate
        num_subbands(int): The number of subbands in spectrum analysis

        **kwargs: 
            'alpha'(float): alpha in pre_emphasize
            'smooth'(bool): Take smooth process or not
            'smooth_type'(str): The smooth type
            'window'(str): The window type

    Returns:
        Tuple(features(dict), label(list)): The features of the exact audio and the true labels
    """
    audio, _ = librosa.load(audio_path, sr = None, ) # 注意此处使用数据原始采样率，且已知采样率均为16000Hz

    if pre_emphasis:
        audio = pre_emphasize(audio, kwargs['alpha'])
        
    features = extract_short_time_features(audio=audio, sampling_rate=sampling_rate, 
                                           frame_length=frame_length, frame_step=frame_step, 
                                           t_min=t_min, t_max=t_max, 
                                           num_subbands=num_subbands, window=kwargs['window'], 
                                           smooth=kwargs['smooth'], smooth_type=kwargs['smooth_type'])
    audio_key = Path(audio_path).stem
    labels = label_dict.get(audio_key, []) # 如果没找到，那就返回空list
        
    # 注意，如vad_utils中的note提到的，真实标签最后是可能需要补0的
    if len(labels) > 0:
        num_frame = len(features['energies'])
        labels = np.pad(labels, (0, max(num_frame - len(labels), 0)))[:num_frame]

    return (features, labels)

def process_audio_file_test(audio_path: str, pre_emphasis: bool = 1, 
                               frame_length: int = 512, frame_step: int = 128, 
                               t_min: float = 0.003, t_max: float = 0.01, 
                               sampling_rate:int = 16000, num_subbands: int = 6, **kwargs): 
    """
    对单个训练文件进行特征提取以及获取真实标签

    Args:
        audio_path(str): The path of one exact audio file
        pre_emphasis(bool): Make pre_emphasis or not
        frame_length(int): The frame length
        frame_step(int): The frame step
        t_min(float): The lower bound of the fundamental period
        t_max(float): The upper bound of the fundamental period
        sampling_rate(int): The sampling rate
        num_subbands(int): The number of subbands in spectrum analysis

        **kwargs: 
            'alpha'(float): alpha in pre_emphasize
            'smooth'(bool): Take smooth process or not
            'smooth_type'(str): The smooth type
            'window'(str): The window type


    Returns:
        features(dict), label(list: The features of the exact audio
    """
    audio, _ = librosa.load(audio_path, sr = None, ) # 注意此处使用数据原始采样率

    if pre_emphasis:
        audio = pre_emphasize(audio, kwargs['alpha'])
        
    features = extract_short_time_features(audio=audio, sampling_rate=sampling_rate, 
                                           frame_length=frame_length, frame_step=frame_step, 
                                           t_min=t_min, t_max=t_max, 
                                           num_subbands=num_subbands, window=kwargs['window'], 
                                           smooth=kwargs['smooth'], smooth_type=kwargs['smooth_type'])

    return features
        
