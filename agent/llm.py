"""Conversation brain: a local LLM served by Ollama's chat API."""

import requests

from .config import SYSTEM_PROMPT, Config


class OrderTaker:
    def __init__(self, config: Config):
        self.cfg = config
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    def check_server(self) -> None:
        """Raise a helpful error if Ollama isn't running or lacks the model."""
        try:
            resp = requests.get(f"{self.cfg.ollama_url}/api/tags", timeout=3)
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Cannot reach Ollama at {self.cfg.ollama_url}. "
                "Start it with: ollama serve"
            ) from exc
        names = [m["name"] for m in resp.json().get("models", [])]
        wanted = self.cfg.llm_model
        if not any(n == wanted or n.split(":")[0] == wanted for n in names):
            raise RuntimeError(
                f"Model '{wanted}' not found in Ollama. "
                f"Pull it with: ollama pull {wanted}"
            )

    def seed_greeting(self, greeting: str) -> None:
        """Record the agent's spoken greeting so the LLM knows it happened."""
        self.messages.append({"role": "assistant", "content": greeting})

    def reply(self, user_text: str) -> str:
        self.messages.append({"role": "user", "content": user_text})
        resp = requests.post(
            f"{self.cfg.ollama_url}/api/chat",
            json={
                "model": self.cfg.llm_model,
                "messages": self.messages,
                "stream": False,
                "options": {"temperature": self.cfg.llm_temperature},
            },
            timeout=120,
        )
        resp.raise_for_status()
        text = resp.json()["message"]["content"].strip()
        self.messages.append({"role": "assistant", "content": text})
        return text
