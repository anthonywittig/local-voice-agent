"""Text-to-speech engines. All options are fully local.

- "say":   macOS built-in synthesizer (default; zero install, offline)
- "piper": Piper neural TTS via its CLI, for non-Mac machines or nicer voices
"""

import shutil
import subprocess
import sys
import tempfile

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
        if self.cfg.tts_engine == "say":
            subprocess.run(
                ["say", "-v", self.cfg.tts_voice, "-r", str(self.cfg.tts_rate_wpm), text],
                check=True,
            )
        elif self.cfg.tts_engine == "piper":
            voice = self.cfg.extra.get("piper_voice", "en_US-lessac-medium")
            with tempfile.NamedTemporaryFile(suffix=".wav") as wav:
                subprocess.run(
                    ["piper", "--model", voice, "--output_file", wav.name],
                    input=text.encode(),
                    check=True,
                )
                player = shutil.which("afplay") or shutil.which("aplay")
                if player is None:
                    raise RuntimeError("No audio player found (afplay/aplay).")
                subprocess.run([player, wav.name], check=True)
        else:
            raise ValueError(f"Unknown TTS engine: {self.cfg.tts_engine}")
