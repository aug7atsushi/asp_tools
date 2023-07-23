import numpy as np
import torch
from scipy.io import wavfile
from torchaudio.compliance import kaldi
from xvector_jtubespeech import XVector


def extract_xvector(
    model: XVector, wav: np.ndarray, num_ceps: int = 24, num_mel_bins: int = 24
) -> np.ndarray:
    wav = torch.from_numpy(wav.astype(np.float32)).unsqueeze(0)
    mfcc = kaldi.mfcc(
        wav, num_ceps=num_ceps, num_mel_bins=num_mel_bins
    )  # (1, T, num_ceps)
    mfcc = mfcc.unsqueeze(0)

    # extract xvector
    xvector = model.vectorize(mfcc)  # (1, 512)
    xvector = xvector.to("cpu").detach().numpy().copy()[0]

    return xvector


def main():
    # Input16kHz mono
    _, wav = wavfile.read(
        "/Users/atsushi/Documents/Workspace/Github/asp_tools/data/xvector_data/sample.wav"
    )
    model = XVector(
        "/Users/atsushi/Documents/Workspace/Github/asp_tools/data/xvector_data/xvector.pth",
        map_location=torch.device("cpu"),
    )

    xvector = extract_xvector(model, wav)  # (512, )
    print(xvector.shape)


if __name__ == "__main__":
    main()
