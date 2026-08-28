"""Pizza order taker: a local speech-to-speech agent.

Cascade pipeline, everything on-device:
    microphone -> faster-whisper (STT) -> Ollama LLM -> TTS -> speakers

While the agent speaks, the mic stays open: sustained caller speech cuts the
TTS off (barge-in). A transcript comparison discards triggers caused by the
agent's own voice echoing from open speakers.
"""

import re
import sys
import time
from typing import Optional

from .audio import MicListener
from .config import GREETING, Config
from .llm import OrderTaker
from .stt import SpeechToText
from .tts import TextToSpeech
from .vad import make_vad


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def looks_like_echo(heard: str, agent_said: str, overlap_ratio: float) -> bool:
    """True if what the mic heard is mostly words the agent was just saying —
    i.e. the agent's own voice leaking from the speakers into the mic."""
    heard_words = _words(heard)
    if not heard_words:
        return True
    said = set(_words(agent_said))
    overlap = sum(w in said for w in heard_words) / len(heard_words)
    return overlap >= overlap_ratio


def main() -> None:
    cfg = Config()

    print("Loading models (first run downloads the Whisper weights)...")
    stt = SpeechToText(cfg)
    tts = TextToSpeech(cfg)
    vad = make_vad(cfg)
    brain = OrderTaker(cfg)
    try:
        brain.check_server()
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    mic = MicListener(cfg, vad)
    if cfg.vad_engine == "energy":
        print("Calibrating microphone, please stay quiet for a second...")
        mic.calibrate()

    def speak(text: str, interruptible: bool = True) -> Optional[str]:
        """Speak `text`; if the caller barges in, return their transcribed
        words (None otherwise, or when the trigger was our own echo)."""
        playback = tts.speak_async(text)
        if not (cfg.barge_in and interruptible):
            playback.wait()
            return None
        audio = mic.monitor_playback(playback)
        if audio is None:
            return None
        heard = stt.transcribe(audio)
        if looks_like_echo(heard, text, cfg.echo_overlap_ratio):
            print(
                "(barge-in ignored as self-echo — for reliable barge-in use "
                "headphones or enable macOS Voice Isolation on the mic)"
            )
            return None
        return heard

    print("\n--- Call connected. Speak after the greeting; Ctrl-C to hang up. ---")
    print("--- You can interrupt the agent mid-sentence. ---\n")
    print(f"Agent: {GREETING}")
    interrupt_text = speak(GREETING)
    brain.seed_greeting(GREETING)

    try:
        while True:
            if interrupt_text is not None:
                user_text = interrupt_text
                print(f"Caller (interrupting): {user_text}")
                t_stt = 0.0
            else:
                print("(listening...)")
                audio = mic.listen()
                t0 = time.perf_counter()
                user_text = stt.transcribe(audio)
                t_stt = time.perf_counter() - t0
                if not user_text:
                    continue
                print(f"Caller: {user_text}")

            t0 = time.perf_counter()
            reply = brain.reply(user_text)
            t_llm = time.perf_counter() - t0

            done = cfg.goodbye_marker in reply
            spoken = reply.replace(cfg.goodbye_marker, "").strip()
            print(f"Agent: {spoken}   [stt {t_stt:.1f}s, llm {t_llm:.1f}s]")
            interrupt_text = None
            if spoken:
                # The goodbye is not interruptible; the order is already confirmed.
                interrupt_text = speak(spoken, interruptible=not done)

            if done:
                print("\n--- Order complete. Call ended. ---")
                break
    except KeyboardInterrupt:
        print("\n--- Call ended. ---")


if __name__ == "__main__":
    main()
