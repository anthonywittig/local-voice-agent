"""Pizza order taker: a local speech-to-speech agent.

Cascade pipeline, everything on-device:
    microphone -> faster-whisper (STT) -> Ollama LLM -> TTS -> speakers
"""

import re
import sys
import time

import numpy as np

from .audio import MicListener
from .config import GREETING, Config
from .llm import OrderTaker
from .stt import SpeechToText
from .tts import TextToSpeech


def speak_maybe_interrupted(
    tts: TextToSpeech, mic: MicListener, cfg: Config, text: str
) -> np.ndarray | None:
    """Speak text. If barge-in is on and the caller talks over it, stop TTS
    and return their utterance; otherwise return None after playback finishes.
    """
    if not cfg.barge_in:
        tts.speak(text)
        return None

    clip = tts.render(text)
    try:
        min_speech_frames = max(1, int(cfg.barge_in_min_ms / cfg.frame_ms))
        grace_frames = max(0, int(cfg.barge_in_grace_ms / cfg.frame_ms))
        return mic.listen_during_playback(
            start_playback=clip.play,
            barge_threshold=mic.threshold * cfg.barge_in_multiplier,
            min_speech_frames=min_speech_frames,
            grace_frames=grace_frames,
            echo_spike=cfg.barge_in_echo_spike,
        )
    except BaseException:
        clip.discard()
        raise


def _normalize_words(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9\s]", "", text.lower()).split())


def is_likely_echo(transcript: str, agent_text: str) -> bool:
    """True when a barge-in transcript is just the agent's own speech leaking
    into the mic (prefix or a contiguous phrase from the line being spoken).
    """
    t = _normalize_words(transcript)
    s = _normalize_words(agent_text)
    if not t:
        return True
    words = t.split()
    spoken = s.split()
    if s.startswith(t):
        return True
    if len(words) < 2:
        return False
    if t in s:
        return True
    n = len(words)
    return any(spoken[i : i + n] == words for i in range(len(spoken) - n + 1))


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

    print("\n--- Call connected. Speak after the greeting (you can interrupt); Ctrl-C to hang up. ---\n")
    print(f"Agent: {GREETING}")
    last_spoken = GREETING
    pending_audio = speak_maybe_interrupted(tts, mic, cfg, GREETING)
    brain.seed_greeting(GREETING)

    try:
        while True:
            from_barge_in = pending_audio is not None
            if pending_audio is None:
                print("(listening...)")
                audio = mic.listen()
            else:
                print("(interrupted)")
                audio = pending_audio
                pending_audio = None

            t0 = time.perf_counter()
            user_text = stt.transcribe(audio)
            t_stt = time.perf_counter() - t0
            if not user_text:
                continue
            if from_barge_in and is_likely_echo(user_text, last_spoken):
                print(f"(echo ignored: {user_text})")
                continue
            if from_barge_in:
                brain.note_interrupted()
            print(f"Caller: {user_text}")

            t0 = time.perf_counter()
            reply = brain.reply(user_text)
            t_llm = time.perf_counter() - t0

            done = cfg.goodbye_marker in reply
            spoken = reply.replace(cfg.goodbye_marker, "").strip()
            print(f"Agent: {spoken}   [stt {t_stt:.1f}s, llm {t_llm:.1f}s]")
            if spoken:
                last_spoken = spoken
                pending_audio = speak_maybe_interrupted(tts, mic, cfg, spoken)
                if pending_audio is not None:
                    done = False

            if done:
                print("\n--- Order complete. Call ended. ---")
                break
    except KeyboardInterrupt:
        print("\n--- Call ended. ---")


if __name__ == "__main__":
    main()
