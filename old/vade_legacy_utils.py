"""Legacy helpers extracted from train/VADE.py during refactor.

These utilities are not used by current training/inference paths.
"""

import numpy as np
import torch


class AVtrans:
    @classmethod
    def deal_data(cls, audio, visual):
        audio = audio.astype(float) / 32768.0
        audio_tensor = cls.mel_audio(audio)
        visual = visual[:, :, :-1]
        image_tensor = torch.tensor(np.array(visual), dtype=torch.float32)
        return audio_tensor, image_tensor

    @classmethod
    def mel_audio(cls, audio):
        import librosa

        left_audio = audio[0]
        right_audio = audio[1]

        sr = 48000
        mel_spec_left = librosa.feature.melspectrogram(
            y=left_audio, sr=sr, n_fft=1024, hop_length=512, n_mels=128
        )
        mel_spec_left = librosa.power_to_db(mel_spec_left, ref=np.max)

        mel_spec_right = librosa.feature.melspectrogram(
            y=right_audio, sr=sr, n_fft=1024, hop_length=512, n_mels=128
        )
        mel_spec_right = librosa.power_to_db(mel_spec_right, ref=np.max)

        mel_spec = np.stack([mel_spec_left, mel_spec_right], axis=0)
        audio_tensor = torch.tensor(mel_spec, dtype=torch.float32)
        return audio_tensor
