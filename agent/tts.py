"""Text-to-speech engines. All options are fully local.

- "say":   macOS built-in synthesizer (default; zero install, offline)
- "piper": Piper neural TTS via its CLI, for non-Mac machines or nicer voices

speak_async() returns a Playback handle so the caller can watch the mic while
audio plays and cut it off when the caller barges in.
"""

import os
import shutil
import subprocess
import sys
import tempfile
from typing import Optional

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


class Playback:
    """A running TTS utterance that can be polled, awaited, or cut off."""

    def __init__(self, proc: subprocess.Popen, cleanup_path: Optional[str] = None):
        self._proc = proc
        self._cleanup_path = cleanup_path

    def is_playing(self) -> bool:
        if self._proc.poll() is None:
            return True
        self._cleanup()
        return False

    def stop(self) -> None:
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()
        self._cleanup()

    def wait(self) -> None:
        self._proc.wait()
        self._cleanup()

    def _cleanup(self) -> None:
        if self._cleanup_path:
            try:
                os.unlink(self._cleanup_path)
            except OSError:
                pass
            self._cleanup_path = None


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

    def speak_async(self, text: str) -> Playback:
        if self.cfg.tts_engine == "say":
            proc = subprocess.Popen(
                ["say", "-v", self.cfg.tts_voice, "-r", str(self.cfg.tts_rate_wpm), text]
            )
            return Playback(proc)
        elif self.cfg.tts_engine == "piper":
            voice = self.cfg.extra.get("piper_voice", "en_US-lessac-medium")
            fd, wav_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            subprocess.run(
                ["piper", "--model", voice, "--output_file", wav_path],
                input=text.encode(),
                check=True,
            )
            player = shutil.which("afplay") or shutil.which("aplay")
            if player is None:
                os.unlink(wav_path)
                raise RuntimeError("No audio player found (afplay/aplay).")
            proc = subprocess.Popen([player, wav_path])
            return Playback(proc, cleanup_path=wav_path)
        else:
            raise ValueError(f"Unknown TTS engine: {self.cfg.tts_engine}")

    def speak(self, text: str) -> None:
        self.speak_async(text).wait()
