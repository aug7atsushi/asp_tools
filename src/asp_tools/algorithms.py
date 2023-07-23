import librosa
import numpy as np


def gla(
    amp_spec: np.ndarray,
    n_fft: int,
    win_length: int,
    hop_length: int,
    max_iteration=200,
) -> np.ndarray:
    for i in range(max_iteration):
        if i == 0:
            est_phase_spec = np.random.rand(*amp_spec.shape)
        else:
            est_spec = amp_spec * np.exp(1j * est_phase_spec)

            est_signal = librosa.core.istft(
                est_spec, hop_length=hop_length, win_length=win_length
            )

            complex_spec = librosa.core.stft(
                est_signal,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
            )

            est_phase_spec = np.angle(complex_spec)
    return est_phase_spec


def main():
    from asp_tools.utils.io import read_wav

    signal, fs = read_wav("../../../data/OSR_us_000_0010_8k.wav")

    iteration = 80
    hop_length = 256
    win_length = 512
    n_fft = win_length
    T = signal.size

    amp_spec = np.abs(
        librosa.core.stft(
            signal, n_fft=n_fft, hop_length=hop_length, win_length=win_length
        )
    )

    est_phase_spec = gla(
        amp_spec, n_fft, win_length, hop_length, max_iteration=iteration
    )
    est_spec = amp_spec * np.exp(1j * est_phase_spec)

    est_signal = librosa.core.istft(
        est_spec,
        hop_length=hop_length,
        win_length=win_length,
        length=T,
    )
    est_signal
