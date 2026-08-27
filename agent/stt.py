"""Speech-to-text via faster-whisper, running fully on-device."""

import numpy as np
from faster_whisper import WhisperModel

from .config import Config


class SpeechToText:
    def __init__(self, config: Config):
        self.model = WhisperModel(
            config.whisper_model,
            device="cpu",
            compute_type=config.whisper_compute_type,
        )

    def transcribe(self, audio: np.ndarray) -> str:
        segments, _info = self.model.transcribe(
            audio,
            language="en",
            beam_size=1,
            vad_filter=True,
        )
        return " ".join(seg.text.strip() for seg in segments).strip()
