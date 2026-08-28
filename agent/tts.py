"""Text-to-speech engines. All options are fully local.

- "say":   macOS built-in synthesizer (default; zero install, offline)
- "piper": Piper neural TTS via its CLI, for non-Mac machines or nicer voices
"""

import os
import shutil
import subprocess
import sys
import tempfile
import wave

import numpy as np

from .config import Config


def _warn_if_output_muted() -> None:
    """say succeeds even when the Mac is muted, so surface that footgun."""
    try:
        result = subprocess.run(
            ["osascript", "-e", "output muted of (get volume settings)"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return
    if result.returncode == 0 and result.stdout.strip().lower() == "true":
        print(
            "Warning: system audio is muted, so you will not hear the agent. "
            "Unmute in Control Center, or run: osascript -e 'set volume output muted false'",
            file=sys.stderr,
        )


def read_wav_mono(path: str, sample_rate: int) -> np.ndarray:
    """Load a wav as float32 mono at ``sample_rate`` (linear resample if needed)."""
    with wave.open(path, "rb") as wf:
        channels = wf.getnchannels()
        width = wf.getsampwidth()
        rate = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    if width == 2:
        audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 4:
        audio = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    elif width == 1:
        audio = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise RuntimeError(f"Unsupported wav sample width: {width}")
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    if rate != sample_rate and audio.size > 1:
        old_t = np.linspace(0.0, 1.0, audio.size, endpoint=False)
        new_n = max(1, int(round(audio.size * sample_rate / rate)))
        new_t = np.linspace(0.0, 1.0, new_n, endpoint=False)
        audio = np.interp(new_t, old_t, audio).astype(np.float32)
    return audio


class Playback:
    """A running TTS process that can finish on its own or be killed for barge-in."""

    def __init__(
        self,
        proc: subprocess.Popen,
        tmp_path: str | None = None,
        reference: np.ndarray | None = None,
    ):
        self.proc = proc
        self.reference = reference
        self._tmp_path = tmp_path

    def alive(self) -> bool:
        return self.proc.poll() is None

    def stop(self) -> None:
        if self.alive():
            self.proc.kill()
            try:
                self.proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass
        self._cleanup()

    def wait(self) -> None:
        rc = self.proc.wait()
        self._cleanup()
        if rc != 0:
            raise subprocess.CalledProcessError(rc, self.proc.args)

    def _cleanup(self) -> None:
        if self._tmp_path:
            try:
                os.unlink(self._tmp_path)
            except OSError:
                pass
            self._tmp_path = None


class SpeechClip:
    """Rendered TTS audio, not yet playing. Used so the mic can open first."""

    def __init__(self, path: str, reference: np.ndarray, player: str):
        self.path = path
        self.reference = reference
        self._player = player
        self._consumed = False

    def play(self) -> Playback:
        self._consumed = True
        return Playback(
            subprocess.Popen([self._player, self.path]),
            tmp_path=self.path,
            reference=self.reference,
        )

    def discard(self) -> None:
        if self._consumed:
            return
        self._consumed = True
        try:
            os.unlink(self.path)
        except OSError:
            pass


class TextToSpeech:
    def __init__(self, config: Config):
        self.cfg = config
        if config.tts_engine == "say" and shutil.which("say") is None:
            raise RuntimeError(
                "The 'say' command was not found (it is macOS-only). "
                "Set PIZZA_TTS_ENGINE=piper and install piper-tts, or run on macOS."
            )
        if config.tts_engine == "piper" and shutil.which("piper") is None:
            raise RuntimeError("piper not found on PATH. Install with: pip install piper-tts")
        if config.tts_engine == "say":
            _warn_if_output_muted()

    def speak(self, text: str) -> None:
        """Blocking playback with no barge-in (live ``say``, or Piper)."""
        if self.cfg.tts_engine == "say":
            subprocess.run(
                ["say", "-v", self.cfg.tts_voice, "-r", str(self.cfg.tts_rate_wpm), text],
                check=True,
            )
            return
        clip = self.render(text)
        try:
            clip.play().wait()
        except BaseException:
            clip.discard()
            raise

    def render(self, text: str) -> SpeechClip:
        """Synthesize to a wav so barge-in can compare the mic against playback."""
        player = shutil.which("afplay") or shutil.which("aplay")
        if player is None:
            raise RuntimeError("No audio player found (afplay/aplay).")
        if self.cfg.tts_engine == "say":
            return self._render_say(text, player)
        if self.cfg.tts_engine == "piper":
            return self._render_piper(text, player)
        raise ValueError(f"Unknown TTS engine: {self.cfg.tts_engine}")

    def _render_say(self, text: str, player: str) -> SpeechClip:
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            subprocess.run(
                [
                    "say",
                    "-v",
                    self.cfg.tts_voice,
                    "-r",
                    str(self.cfg.tts_rate_wpm),
                    "-o",
                    path,
                    f"--data-format=LEI16@{self.cfg.sample_rate}",
                    text,
                ],
                check=True,
            )
            reference = read_wav_mono(path, self.cfg.sample_rate)
        except Exception:
            try:
                os.unlink(path)
            except OSError:
                pass
            raise
        return SpeechClip(path, reference, player)

    def _render_piper(self, text: str, player: str) -> SpeechClip:
        voice = self.cfg.extra.get("piper_voice", "en_US-lessac-medium")
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            subprocess.run(
                ["piper", "--model", voice, "--output_file", path],
                input=text.encode(),
                check=True,
            )
            reference = read_wav_mono(path, self.cfg.sample_rate)
        except Exception:
            try:
                os.unlink(path)
            except OSError:
                pass
            raise
        return SpeechClip(path, reference, player)
