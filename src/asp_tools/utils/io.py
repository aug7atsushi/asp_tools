import numpy as np
from scipy.io import wavfile


def read_wav(wav_path: str) -> tuple[np.ndarray, int]:
    """_summary_

    Args:
        wav_path (str): _description_
        int (_type_): _description_

    Returns:
        _type_: _description_
    """
    fs, signal = wavfile.read(wav_path)
    signal = signal / 32768

    return signal, fs


def write_wav(wav_path: str, fs: int, signal: np.ndarray) -> None:
    """_summary_

    Args:
        wav_path (str): _description_
        fs (int): _description_
        signal (np.ndarray): _description_
    """
    signal = signal * 32768
    signal = np.clip(signal, np.iinfo(np.int16).min, np.iinfo(np.int16).max).astype(
        np.int16
    )
    wavfile.write(wav_path, fs, signal)
