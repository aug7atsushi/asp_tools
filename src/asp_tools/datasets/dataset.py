import json
import os

import torch
import torchaudio

from asp_tools.utils.logging import get_module_logger

logger = get_module_logger(__name__)


class LibriSpeechDataset(torch.utils.data.Dataset):
    def __init__(self, wav_root_path: str, json_path: str):
        super().__init__()

        self.wav_root_path = os.path.abspath(wav_root_path)
        json_path = os.path.abspath(json_path)

        with open(json_path) as f:
            self.json_data = json.load(f)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, list]:
        """
        Returns:
            mixture (1, T) <torch.Tensor>
            sources (2, T) <torch.Tensor>
            segment_IDs (2,) <list<str>>
        """

        source0_info = self.json_data[idx]["sources"]["source-0"]
        source1_info = self.json_data[idx]["sources"]["source-1"]
        source0_length = source0_info["end"] - source0_info["start"]
        source1_length = source1_info["end"] - source1_info["start"]
        segment_ids = [
            "{}_{}-{}".format(
                source0_info["utterance-ID"], source0_info["start"], source0_info["end"]
            ),
            "{}_{}-{}".format(
                source1_info["utterance-ID"], source1_info["start"], source1_info["end"]
            ),
        ]

        if source0_length != source1_length:
            raise ValueError(f"音源の切り出し区間の長さがファイル間で異なります。settingsファイルを確認してください。")

        source0, fs1 = torchaudio.load(
            filepath=os.path.join(self.wav_root_path, source0_info["path"]),
            frame_offset=source0_info["start"],
            num_frames=source0_length,
        )

        source1, fs2 = torchaudio.load(
            filepath=os.path.join(self.wav_root_path, source1_info["path"]),
            frame_offset=source1_info["start"],
            num_frames=source1_length,
        )

        if fs1 != fs2:
            raise ValueError(f"サンプリングレートが異なります。")

        mixture = source0 + source1
        sources = torch.concat([source0, source1], dim=0)

        return mixture, sources, segment_ids

    def __len__(self):
        return len(self.json_data)


class WaveTrainDataset(LibriSpeechDataset):
    def __init__(self, wav_root_path, json_path):
        super().__init__(wav_root_path, json_path)

    def __getitem__(self, idx):
        mixture, sources, _ = super().__getitem__(idx)
        return mixture, sources


class WaveValidDataset(LibriSpeechDataset):
    def __init__(self, wav_root_path, json_path):
        super().__init__(wav_root_path, json_path)

    def __getitem__(self, idx):
        mixture, sources, _ = super().__getitem__(idx)
        return mixture, sources


class WaveTestDataset(LibriSpeechDataset):
    def __init__(self, wav_root_path, json_path):
        super().__init__(wav_root_path, json_path)

    def __getitem__(self, idx):
        mixture, sources, segment_IDs = super().__getitem__(idx)
        return mixture, sources, segment_IDs
