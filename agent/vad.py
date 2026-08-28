"""Voice activity detection.

Two engines, both fully local:

- "silero": Silero VAD v5 (~2 MB ONNX model, runs on CPU via onnxruntime).
  Robust to fans, keyboards, and room noise; the default.
- "energy": the original RMS-threshold detector, kept as a zero-download
  fallback. Requires ambient-noise calibration at startup.

Both expose is_speech(frame) for 512-sample float32 frames at 16 kHz.
"""

import os
import urllib.request

import numpy as np

from .config import Config

SILERO_URL = (
    "https://raw.githubusercontent.com/snakers4/silero-vad/v5.1.2/"
    "src/silero_vad/data/silero_vad.onnx"
)
CACHE_DIR = os.path.expanduser("~/.cache/local-voice-agent")

FRAME_SAMPLES = 512  # Silero v5 requires exactly 512 samples per chunk at 16 kHz
_CONTEXT = 64        # samples of left context the v5 model expects prepended


class SileroVAD:
    def __init__(self, config: Config):
        import onnxruntime as ort

        model_path = config.vad_model_path or os.path.join(
            CACHE_DIR, "silero_vad.onnx"
        )
        if not os.path.exists(model_path):
            print(f"Downloading Silero VAD model (~2 MB) to {model_path}...")
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            urllib.request.urlretrieve(SILERO_URL, model_path)

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self.session = ort.InferenceSession(
            model_path, opts, providers=["CPUExecutionProvider"]
        )
        self.threshold = config.vad_threshold
        self._sr = np.array(config.sample_rate, dtype=np.int64)
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, _CONTEXT), dtype=np.float32)

    def prob(self, frame: np.ndarray) -> float:
        x = np.concatenate(
            [self._context, frame.reshape(1, -1)], axis=1
        ).astype(np.float32)
        out, self._state = self.session.run(
            ["output", "stateN"],
            {"input": x, "state": self._state, "sr": self._sr},
        )
        self._context = x[:, -_CONTEXT:]
        return float(out[0, 0])

    def is_speech(self, frame: np.ndarray, threshold: float | None = None) -> bool:
        return self.prob(frame) >= (self.threshold if threshold is None else threshold)


class EnergyVAD:
    def __init__(self, config: Config):
        self.cfg = config
        self.threshold = config.vad_min_threshold

    def calibrate(self, ambient_rms: float) -> None:
        self.threshold = max(
            self.cfg.vad_min_threshold, ambient_rms * self.cfg.vad_energy_multiplier
        )

    def reset(self) -> None:
        pass

    def is_speech(self, frame: np.ndarray, threshold: float | None = None) -> bool:
        return float(np.sqrt(np.mean(frame**2))) >= self.threshold


def make_vad(config: Config):
    if config.vad_engine == "silero":
        return SileroVAD(config)
    if config.vad_engine == "energy":
        return EnergyVAD(config)
    raise ValueError(f"Unknown VAD engine: {config.vad_engine}")
