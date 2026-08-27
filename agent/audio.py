"""Microphone capture with a simple energy-based voice activity detector.

Records from the default input device and returns one utterance at a time:
capture starts when the caller begins speaking and stops after a stretch of
silence. The ambient noise floor is measured at startup so the threshold
adapts to the room.
"""

import queue
import sys

import numpy as np
import sounddevice as sd

from .config import Config


class MicListener:
    def __init__(self, config: Config):
        self.cfg = config
        self.frame_samples = int(config.sample_rate * config.frame_ms / 1000)
        self.threshold = config.vad_min_threshold

    def calibrate(self) -> None:
        """Sample ambient noise to set the speech energy threshold."""
        frames = int(self.cfg.calibration_sec * self.cfg.sample_rate)
        recording = sd.rec(
            frames, samplerate=self.cfg.sample_rate, channels=1, dtype="float32"
        )
        sd.wait()
        ambient = float(np.sqrt(np.mean(recording**2)))
        self.threshold = max(
            self.cfg.vad_min_threshold, ambient * self.cfg.vad_energy_multiplier
        )

    def listen(self) -> np.ndarray:
        """Block until the caller speaks, then return the utterance as
        float32 mono audio at the configured sample rate."""
        frames: "queue.Queue[np.ndarray]" = queue.Queue()

        def callback(indata, _frames, _time, status):
            if status:
                print(f"[audio] {status}", file=sys.stderr)
            frames.put(indata[:, 0].copy())

        silence_frames_needed = int(
            self.cfg.silence_end_sec * 1000 / self.cfg.frame_ms
        )
        max_frames = int(self.cfg.max_utterance_sec * 1000 / self.cfg.frame_ms)

        captured: list[np.ndarray] = []
        speaking = False
        silent_run = 0

        with sd.InputStream(
            samplerate=self.cfg.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.frame_samples,
            callback=callback,
        ):
            while True:
                frame = frames.get()
                energy = float(np.sqrt(np.mean(frame**2)))
                if not speaking:
                    if energy >= self.threshold:
                        speaking = True
                        captured.append(frame)
                        silent_run = 0
                else:
                    captured.append(frame)
                    if energy < self.threshold:
                        silent_run += 1
                        if silent_run >= silence_frames_needed:
                            break
                    else:
                        silent_run = 0
                    if len(captured) >= max_frames:
                        break

        return np.concatenate(captured)
