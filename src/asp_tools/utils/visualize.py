import matplotlib.pyplot as plt


def show_spectrogram_from_waveform(
    signal, fs, frame_length=1024, hop_length=80, figsize=(10, 4)
):
    noverlap = frame_length - hop_length
    nfft = frame_length

    # プロット枠を確保
    fig, ax = plt.subplots(figsize=figsize)

    # スペクトログラムのプロット
    _, _, _, imgs = ax.specgram(
        signal,
        NFFT=nfft,
        noverlap=noverlap,
        Fs=fs,
        cmap="viridis",
    )

    fig.colorbar(imgs, ax=ax)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Frequency [Hz]")
    ax.set_title("Spectrogram")

    return fig, ax
