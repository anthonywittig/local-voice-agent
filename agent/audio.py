"""Microphone capture and turn-taking.

Two capture modes, both driven by the configured VAD:

- listen(): block until the caller speaks, return the utterance once they
  stop (the normal turn).
- monitor_playback(): watch the mic *while the agent is talking*. If the
  caller barges in with sustained speech, stop playback and capture their
  utterance; if playback finishes uninterrupted, return None.
"""

import collections
import queue
import sys
from typing import Optional

import numpy as np
import sounddevice as sd

from .config import Config
from .vad import FRAME_SAMPLES, EnergyVAD


class MicListener:
    def __init__(self, config: Config, vad):
        self.cfg = config
        self.vad = vad
        self.frame_sec = FRAME_SAMPLES / config.sample_rate

    def calibrate(self) -> None:
        """Sample ambient noise to set the energy-VAD threshold (no-op for Silero)."""
        if not isinstance(self.vad, EnergyVAD):
            return
        frames = int(self.cfg.calibration_sec * self.cfg.sample_rate)
        recording = sd.rec(
            frames, samplerate=self.cfg.sample_rate, channels=1, dtype="float32"
        )
        sd.wait()
        self.vad.calibrate(float(np.sqrt(np.mean(recording**2))))

    def _stream(self, frames: "queue.Queue[np.ndarray]") -> sd.InputStream:
        def callback(indata, _frames, _time, status):
            if status:
                print(f"[audio] {status}", file=sys.stderr)
            frames.put(indata[:, 0].copy())

        return sd.InputStream(
            samplerate=self.cfg.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=FRAME_SAMPLES,
            callback=callback,
        )

    def listen(self) -> np.ndarray:
        """Block until the caller speaks, then return the utterance as
        float32 mono audio at the configured sample rate."""
        frames: "queue.Queue[np.ndarray]" = queue.Queue()
        silence_frames_needed = int(self.cfg.silence_end_sec / self.frame_sec)
        max_frames = int(self.cfg.max_utterance_sec / self.frame_sec)

        captured: list[np.ndarray] = []
        speaking = False
        silent_run = 0

        self.vad.reset()
        with self._stream(frames):
            while True:
                frame = frames.get()
                voiced = self.vad.is_speech(frame)
                if not speaking:
                    if voiced:
                        speaking = True
                        captured.append(frame)
                        silent_run = 0
                else:
                    captured.append(frame)
                    if voiced:
                        silent_run = 0
                    else:
                        silent_run += 1
                        if silent_run >= silence_frames_needed:
                            break
                    if len(captured) >= max_frames:
                        break

        return np.concatenate(captured)

    def monitor_playback(self, playback, is_real_interrupt) -> Optional[np.ndarray]:
        """Listen while the agent speaks. On sustained caller speech, ask
        `is_real_interrupt(audio)` whether it is the caller (and not the
        agent's own voice echoing from the speakers). Only then stop
        `playback`, capture the rest of the utterance, and return it.
        Return None if playback finished without a real interruption.

        The check runs while playback continues, so a false trigger never
        cuts the agent off; frames arriving during the check queue up and
        are still processed.
        """
        frames: "queue.Queue[np.ndarray]" = queue.Queue()
        trigger_frames = max(
            1, int(self.cfg.barge_in_min_speech_sec / self.frame_sec)
        )
        # Rolling window handed to the echo check: the sustained-speech run
        # plus some lead-in so the start of the interjection isn't lost.
        preroll = collections.deque(
            maxlen=trigger_frames
            + int(self.cfg.barge_in_preroll_sec / self.frame_sec)
        )
        silence_frames_needed = int(self.cfg.silence_end_sec / self.frame_sec)
        max_frames = int(self.cfg.max_utterance_sec / self.frame_sec)

        speech_run = 0
        interrupted = False
        self.vad.reset()
        with self._stream(frames):
            # Phase 1: agent talking; look for a sustained, confident interruption.
            while playback.is_playing():
                try:
                    frame = frames.get(timeout=0.5)
                except queue.Empty:
                    continue
                preroll.append(frame)
                if self.vad.is_speech(frame, threshold=self.cfg.barge_in_threshold):
                    speech_run += 1
                    if speech_run >= trigger_frames:
                        if is_real_interrupt(np.concatenate(preroll)):
                            playback.stop()
                            interrupted = True
                            break
                        # Echo (or unintelligible): keep talking, look for a
                        # fresh window of sustained speech.
                        speech_run = 0
                else:
                    speech_run = 0
            if not interrupted:
                return None  # playback finished on its own

            # Phase 2: caller has the floor; capture until they go quiet.
            captured = list(preroll)
            silent_run = 0
            while True:
                frame = frames.get()
                captured.append(frame)
                if self.vad.is_speech(frame):
                    silent_run = 0
                else:
                    silent_run += 1
                    if silent_run >= silence_frames_needed:
                        break
                if len(captured) >= max_frames:
                    break

        return np.concatenate(captured)
