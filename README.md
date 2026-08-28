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
confirm. You can **interrupt the agent mid-sentence** (barge-in) — the mic
stays open while it talks.

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

```sh
make setup   # venv, Python deps, PortAudio, Ollama, LLM weights
make run     # start taking orders
```

Listen to the greeting, then just talk. The agent detects when you stop
speaking, transcribes, thinks, and talks back — and you can talk over it to
interrupt. Ctrl-C hangs up.

`make help` lists the other targets (`install`, `ollama`, `clean`). To use a
different LLM: `make setup LLM_MODEL=qwen2.5:7b` then `make run`.

### Manual setup

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
| `PIZZA_VAD_ENGINE` | `silero` | `silero` (neural, robust) or `energy` (no download) |
| `PIZZA_VAD_THRESHOLD` | `0.5` | Silero speech probability that counts as speech |
| `PIZZA_VAD_MULT` | `4.0` | Energy engine: threshold as a multiple of ambient noise |
| `PIZZA_BARGE_IN` | `1` | Set `0` to disable interrupting the agent |
| `PIZZA_BARGE_THRESHOLD` | `0.85` | Speech probability required to cut the agent off |
| `PIZZA_BARGE_MIN_SPEECH` | `0.35` | Seconds of sustained speech before TTS stops |

### Non-Mac / nicer voices

Set `PIZZA_TTS_ENGINE=piper` and `pip install piper-tts` to use
[Piper](https://github.com/rhasspy/piper), a fast local neural TTS that works on
Linux too.

## Barge-in (interrupting the agent)

While the agent speaks, the mic stays open. If you talk over it with
sustained, confident speech (~0.35 s above a 0.85 Silero speech probability),
the TTS process is killed, your words — including a half-second of pre-roll
from before the trigger — are captured and transcribed, and the conversation
continues from your interruption.

The hard part of barge-in is the agent hearing **itself** through open
speakers. Two defenses are in place:

1. The barge-in trigger uses a stricter threshold and requires sustained
   speech, so quieter echo usually doesn't trip it.
2. If it does trip, the transcript is compared against what the agent was
   saying; when most of the heard words match, it's discarded as self-echo
   instead of being sent to the LLM.

For reliable barge-in, use **headphones**, or enable macOS **Voice Isolation**
on the mic (click the orange mic icon in the menu bar while the agent is
running → Mic Mode → Voice Isolation) — Apple's own echo/noise suppression
then strips the agent's voice from the input. Set `PIZZA_BARGE_IN=0` to turn
barge-in off entirely.

## How it works

- **`agent/vad.py`** — voice activity detection. Default is [Silero VAD](https://github.com/snakers4/silero-vad)
  v5 (a ~2 MB ONNX model, one-time download, CPU inference via onnxruntime),
  which is robust to fans and room noise. `PIZZA_VAD_ENGINE=energy` falls back
  to the original RMS-threshold detector with ambient calibration.
- **`agent/audio.py`** — turn-taking. `listen()` waits for you to speak and
  ends your turn after ~0.9 s of silence; `monitor_playback()` watches the mic
  while the agent talks and implements the barge-in trigger.
- **`agent/stt.py`** — transcribes the utterance with faster-whisper
  (CTranslate2, int8), which is fast on Apple Silicon CPUs.
- **`agent/llm.py`** — sends the conversation to Ollama's chat API. The system
  prompt makes the model a pizza order taker with a fixed menu and rules to
  keep replies short and voice-friendly. When the order is confirmed, the model
  appends an `[END_CALL]` token, which cleanly ends the session.
- **`agent/tts.py`** — speaks the reply with macOS `say` (or Piper), returning
  a `Playback` handle that can be polled and stopped for barge-in.
- **`agent/main.py`** — the loop tying it together, printing the transcript and
  per-stage latencies as you go, plus the self-echo transcript check.

## Ideas for later

- Stream the LLM output sentence-by-sentence into TTS to cut response latency.
- Full acoustic echo cancellation (speexdsp or macOS voice-processing I/O) so
  open-speaker barge-in works without Voice Isolation.
- Structured order extraction (JSON) alongside the conversation for a real POS.
