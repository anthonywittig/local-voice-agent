# local-voice-agent

A speech-to-speech pizza order taker that runs **entirely locally** on a MacBook Pro —
no cloud APIs, no network calls (after the one-time model downloads).

It uses the classic cascade architecture:

```
microphone ──► STT ──────────► LLM ─────────► TTS ──► speakers
              faster-whisper   Ollama         macOS `say`
              (base.en)        (llama3.2:3b)  (or Piper)
```

You call Mario's Pizzeria, and "Sam" takes your order: size, toppings, sides,
pickup or delivery, reads the order back with the total, and hangs up when you
confirm.

## Can this really run locally on a MacBook Pro?

Yes — comfortably, on any Apple Silicon MacBook Pro (and it's fine on 8 GB of RAM
with the default models):

| Stage | Model | Footprint | Typical latency |
|---|---|---|---|
| STT | Whisper `base.en` via faster-whisper (int8, CPU) | ~150 MB | well under 1 s per utterance |
| LLM | `llama3.2:3b` via Ollama (Metal-accelerated) | ~2 GB | 1–3 s per reply |
| TTS | macOS built-in `say` | 0 (ships with macOS) | near-instant |

If you have 16 GB+ of RAM, try `small.en` for STT and `llama3.1:8b` or
`qwen2.5:7b` for noticeably better conversation quality — still very usable
latency.

## Setup

1. **Install Ollama** and pull the LLM:

   ```sh
   brew install ollama
   ollama serve &          # or run the Ollama app
   ollama pull llama3.2:3b
   ```

2. **Install this package** (Python 3.10+):

   ```sh
   python3 -m venv .venv && source .venv/bin/activate
   pip install -e .
   ```

   You may also need PortAudio for microphone capture: `brew install portaudio`.

3. **Run it:**

   ```sh
   pizza-agent
   ```

   Stay quiet for the one-second mic calibration, listen to the greeting, then
   just talk. The agent detects when you stop speaking, transcribes, thinks,
   and talks back. Ctrl-C hangs up.

The first run downloads the Whisper weights (~150 MB) to the Hugging Face
cache; after that, everything works offline.

## Configuration

Everything is tunable via environment variables (see `agent/config.py`):

| Variable | Default | Purpose |
|---|---|---|
| `PIZZA_WHISPER_MODEL` | `base.en` | Whisper size: `tiny.en` / `base.en` / `small.en` |
| `PIZZA_LLM_MODEL` | `llama3.2:3b` | Any Ollama chat model |
| `PIZZA_OLLAMA_URL` | `http://localhost:11434` | Ollama endpoint |
| `PIZZA_TTS_ENGINE` | `say` | `say` (macOS) or `piper` |
| `PIZZA_TTS_VOICE` | `Samantha` | macOS voice (`say -v '?'` lists them) |
| `PIZZA_SILENCE_END` | `0.9` | Seconds of silence that end your turn |
| `PIZZA_VAD_MULT` | `4.0` | Speech threshold as a multiple of ambient noise |

### Non-Mac / nicer voices

Set `PIZZA_TTS_ENGINE=piper` and `pip install piper-tts` to use
[Piper](https://github.com/rhasspy/piper), a fast local neural TTS that works on
Linux too.

## How it works

- **`agent/audio.py`** — captures mic audio with a simple energy-based voice
  activity detector. It calibrates against ambient noise at startup, starts
  recording when you speak, and ends your turn after ~0.9 s of silence.
- **`agent/stt.py`** — transcribes the utterance with faster-whisper
  (CTranslate2, int8), which is fast on Apple Silicon CPUs.
- **`agent/llm.py`** — sends the conversation to Ollama's chat API. The system
  prompt makes the model a pizza order taker with a fixed menu and rules to
  keep replies short and voice-friendly. When the order is confirmed, the model
  appends an `[END_CALL]` token, which cleanly ends the session.
- **`agent/tts.py`** — speaks the reply with macOS `say` (or Piper).
- **`agent/main.py`** — the loop tying it together, printing the transcript and
  per-stage latencies as you go.

## Ideas for later

- Stream the LLM output sentence-by-sentence into TTS to cut response latency.
- Swap the energy VAD for [Silero VAD](https://github.com/snakers4/silero-vad)
  for robustness in noisy rooms.
- Barge-in support (stop TTS when the caller starts talking).
- Structured order extraction (JSON) alongside the conversation for a real POS.
