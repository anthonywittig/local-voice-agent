"""Pizza order taker: a local speech-to-speech agent.

Cascade pipeline, everything on-device:
    microphone -> faster-whisper (STT) -> Ollama LLM -> TTS -> speakers
"""

import sys
import time

from .audio import MicListener
from .config import GREETING, Config
from .llm import OrderTaker
from .stt import SpeechToText
from .tts import TextToSpeech


def main() -> None:
    cfg = Config()

    print("Loading models (first run downloads the Whisper weights)...")
    stt = SpeechToText(cfg)
    tts = TextToSpeech(cfg)
    brain = OrderTaker(cfg)
    try:
        brain.check_server()
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    mic = MicListener(cfg)
    print("Calibrating microphone, please stay quiet for a second...")
    mic.calibrate()

    print("\n--- Call connected. Speak after the greeting; Ctrl-C to hang up. ---\n")
    print(f"Agent: {GREETING}")
    tts.speak(GREETING)
    brain.seed_greeting(GREETING)

    try:
        while True:
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
            if spoken:
                tts.speak(spoken)

            if done:
                print("\n--- Order complete. Call ended. ---")
                break
    except KeyboardInterrupt:
        print("\n--- Call ended. ---")


if __name__ == "__main__":
    main()
