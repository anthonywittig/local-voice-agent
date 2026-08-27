"""Configuration for the pizza voice agent.

Everything runs locally; these settings pick the models and tune the
voice-activity detection. Override via environment variables.
"""

import os
from dataclasses import dataclass, field


@dataclass
class Config:
    # --- Speech to text (faster-whisper) ---
    # "base.en" is a good speed/accuracy tradeoff on a MacBook Pro.
    # Try "small.en" for better accuracy or "tiny.en" for lower latency.
    whisper_model: str = os.environ.get("PIZZA_WHISPER_MODEL", "base.en")
    whisper_compute_type: str = os.environ.get("PIZZA_WHISPER_COMPUTE", "int8")

    # --- LLM (Ollama) ---
    ollama_url: str = os.environ.get("PIZZA_OLLAMA_URL", "http://localhost:11434")
    llm_model: str = os.environ.get("PIZZA_LLM_MODEL", "llama3.2:3b")
    llm_temperature: float = float(os.environ.get("PIZZA_LLM_TEMPERATURE", "0.6"))

    # --- Text to speech ---
    # "say" uses the built-in macOS synthesizer (fully offline, no install).
    tts_engine: str = os.environ.get("PIZZA_TTS_ENGINE", "say")
    tts_voice: str = os.environ.get("PIZZA_TTS_VOICE", "Samantha")
    tts_rate_wpm: int = int(os.environ.get("PIZZA_TTS_RATE", "185"))

    # --- Audio capture / VAD ---
    sample_rate: int = 16000
    frame_ms: int = 30                      # analysis frame size
    silence_end_sec: float = float(os.environ.get("PIZZA_SILENCE_END", "0.9"))
    max_utterance_sec: float = 30.0
    calibration_sec: float = 1.0            # ambient-noise sampling at startup
    vad_energy_multiplier: float = float(os.environ.get("PIZZA_VAD_MULT", "4.0"))
    vad_min_threshold: float = 0.004        # floor so a silent room doesn't trigger

    # Words/phrases that end the call once the order is confirmed
    goodbye_marker: str = "[END_CALL]"

    extra: dict = field(default_factory=dict)


SYSTEM_PROMPT = """\
You are Sam, a friendly and efficient order taker at Mario's Pizzeria, speaking with a
customer on the phone. Your job is to take their pizza order.

MENU
- Pizzas (small $9, medium $12, large $15): Margherita, Pepperoni, Hawaiian, Veggie, Meat Lovers, BBQ Chicken
- Extra toppings ($1.50 each): mushrooms, olives, onions, peppers, extra cheese, bacon, pineapple, jalapenos
- Sides: garlic bread $5, wings (8pc) $8, caesar salad $7
- Drinks ($2.50): cola, diet cola, lemonade, sparkling water

RULES
- You are on a VOICE call. Keep every reply short and natural: one or two spoken
  sentences. Never use lists, markdown, emoji, or stage directions.
- Greet the caller once at the start, then take their order.
- Ask one question at a time (size, toppings, sides, drinks, pickup or delivery,
  name; address only if delivery).
- If they ask for something not on the menu, apologize briefly and offer the
  closest item.
- When the order seems complete, read the full order back with the total price
  and ask them to confirm.
- After they confirm, thank them, tell them pickup takes about 20 minutes (or
  delivery about 40), say goodbye, and end your reply with the exact token
  [END_CALL] on the same line.
- Speech recognition may garble words; if something is unclear, ask them to repeat it.
"""

GREETING = "Thanks for calling Mario's Pizzeria, this is Sam. What can I get for you today?"
