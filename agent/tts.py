"""Text-to-speech engines. All options are fully local.

- "say":   macOS built-in synthesizer (default; zero install, offline)
- "piper": Piper neural TTS via its CLI, for non-Mac machines or nicer voices
"""

import shutil
import subprocess
import tempfile

from .config import Config


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
