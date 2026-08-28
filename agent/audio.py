"""Microphone capture with a simple energy-based voice activity detector.

Records from the default input device and returns one utterance at a time:
capture starts when the caller begins speaking and stops after a stretch of
silence. The ambient noise floor is measured at startup so the threshold
adapts to the room.

During TTS, listen_during_playback() watches for barge-in so the caller can
talk over the agent. It compares mic energy to a delayed copy of the TTS
waveform so speaker echo is not mistaken for the caller.
"""

from collections.abc import Callable
from typing import Protocol
import queue
import sys

import numpy as np
import sounddevice as sd

from .config import Config

# Mic lags the speaker by this much at most (afplay buffer + acoustic path).
_MAX_ECHO_DELAY_FRAMES = 20


class _Playback(Protocol):
    reference: np.ndarray | None

    def alive(self) -> bool: ...
    def stop(self) -> None: ...
    def wait(self) -> None: ...


def frame_rms(audio: np.ndarray, frame_samples: int) -> np.ndarray:
    """RMS energy of consecutive frames; drops a trailing partial frame."""
    n = audio.size // frame_samples
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    trimmed = audio[: n * frame_samples].reshape(n, frame_samples)
    return np.sqrt(np.mean(trimmed.astype(np.float64) ** 2, axis=1))


def fit_echo(
    mic_energy: list[float], ref_energy: np.ndarray, max_delay_frames: int
) -> tuple[int, float, float]:
    """Estimate how mic energy lines up with TTS: (delay_frames, gain, corr).

    delay is how many frames the mic lags the reference (afplay + acoustic
    path). We correlate mic[delay:] with ref[0:] so leading silence is skipped.
    """
    mic = np.asarray(mic_energy, dtype=np.float64)
    n = mic.size
    if n < 8 or ref_energy.size < 4:
        return 0, 1.0, 0.0

    best_d = 0
    best_score = -1.0
    max_d = min(max_delay_frames, n - 8)
    for delay in range(0, max(0, max_d) + 1):
        m = mic[delay:]
        r = ref_energy[: m.size]
        length = min(m.size, r.size)
        m = m[:length]
        r = r[:length]
        if m.size < 8 or float(np.std(r)) < 1e-8 or float(np.std(m)) < 1e-8:
            continue
        score = float(np.corrcoef(m, r)[0, 1])
        if np.isnan(score):
            continue
        if score > best_score:
            best_score = score
            best_d = delay

    if best_score < 0:
        return 0, 1.0, 0.0

    m = mic[best_d:]
    r = ref_energy[: m.size]
    length = min(m.size, r.size)
    m = m[:length]
    r = r[:length]
    loud = r > max(float(np.percentile(r, 25)), 1e-5)
    if np.any(loud):
        gain = float(np.median(m[loud] / np.maximum(r[loud], 1e-8)))
    else:
        gain = float(np.median(m / np.maximum(r, 1e-8)))
    return best_d, float(np.clip(gain, 0.05, 8.0)), float(best_score)


# Correlation below this means we cannot trust echo prediction; skip barge-in.
# Real speaker-echo envelopes usually score ~0.6–0.95 once delay is right.
_MIN_ECHO_CORR = 0.5


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

    def listen_during_playback(
        self,
        start_playback: Callable[[], _Playback],
        barge_threshold: float,
        min_speech_frames: int,
        grace_frames: int,
        echo_spike: float,
    ) -> np.ndarray | None:
        """Open the mic, then start TTS. If the caller talks over it, stop
        playback and return the utterance. If playback finishes first, return None.

        Mic energy is compared to a delayed, gained copy of the TTS waveform so
        a loud syllable from the speakers does not count as barge-in.
        """
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
        speech_run = 0
        seen = 0
        grace_energy: list[float] = []
        delay = 0
        gain = 1.0
        echo_corr = 0.0
        fitted = False
        allow_barge_in = True
        ref_energy = np.zeros(0, dtype=np.float64)
        playback: _Playback | None = None

        try:
            with sd.InputStream(
                samplerate=self.cfg.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=self.frame_samples,
                callback=callback,
            ):
                playback = start_playback()
                if playback.reference is not None and playback.reference.size:
                    ref_energy = frame_rms(playback.reference, self.frame_samples)

                while True:
                    try:
                        frame = frames.get(timeout=0.05)
                    except queue.Empty:
                        if not speaking and (playback is None or not playback.alive()):
                            break
                        continue

                    seen += 1
                    energy = float(np.sqrt(np.mean(frame**2)))

                    if not speaking:
                        if playback is None or not playback.alive():
                            captured.clear()
                            break
                        if seen <= grace_frames:
                            grace_energy.append(energy)
                            continue
                        if not fitted:
                            if ref_energy.size:
                                delay, gain, echo_corr = fit_echo(
                                    grace_energy, ref_energy, _MAX_ECHO_DELAY_FRAMES
                                )
                                allow_barge_in = echo_corr >= _MIN_ECHO_CORR
                            else:
                                allow_barge_in = False
                            fitted = True
                            if allow_barge_in:
                                print("(barge-in ready)")
                            else:
                                print(
                                    f"(barge-in off, echo corr={echo_corr:.2f})",
                                    file=sys.stderr,
                                )

                        if not allow_barge_in:
                            continue

                        ref_i = seen - 1 - delay
                        if ref_i < 0 or ref_i >= ref_energy.size:
                            continue
                        expected = gain * float(ref_energy[ref_i])
                        floor = max(barge_threshold, expected * echo_spike)

                        if energy >= floor:
                            speech_run += 1
                            captured.append(frame)
                            if speech_run >= min_speech_frames:
                                speaking = True
                                silent_run = 0
                                playback.stop()
                        else:
                            speech_run = 0
                            captured.clear()
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
        except BaseException:
            if playback is not None:
                playback.stop()
            raise

        if not speaking:
            if playback is not None:
                playback.wait()
            return None
        if not captured:
            return None
        return np.concatenate(captured)
