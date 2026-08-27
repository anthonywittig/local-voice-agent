PYTHON ?= python3
VENV := .venv
BIN := $(VENV)/bin
LLM_MODEL ?= llama3.2:3b
OLLAMA_URL ?= http://localhost:11434

.DEFAULT_GOAL := help

.PHONY: help setup install brew-deps ollama ollama-serve ollama-pull run clean

help:
	@echo "Targets:"
	@echo "  make setup   First-time setup: venv, Python deps, PortAudio, Ollama, LLM"
	@echo "  make run     Start the pizza order agent"
	@echo "  make install Python venv + package only"
	@echo "  make ollama  Ensure Ollama is running and $(LLM_MODEL) is pulled"
	@echo "  make clean   Remove the virtualenv and build artifacts"

setup: brew-deps install ollama
	@echo ""
	@echo "Setup complete. Run the agent with: make run"

install: $(BIN)/pizza-agent

$(VENV)/bin/python:
	$(PYTHON) -m venv $(VENV)

$(BIN)/pizza-agent: $(VENV)/bin/python pyproject.toml
	$(BIN)/pip install -e .

ifeq ($(shell uname -s),Darwin)
brew-deps:
	@command -v brew >/dev/null 2>&1 || { \
		echo "Homebrew is required on macOS. Install it from https://brew.sh"; \
		exit 1; \
	}
	@if brew list portaudio >/dev/null 2>&1; then \
		echo "PortAudio already installed."; \
	else \
		echo "Installing PortAudio..."; \
		brew install portaudio; \
	fi
	@if command -v ollama >/dev/null 2>&1; then \
		echo "Ollama already installed."; \
	else \
		echo "Installing Ollama..."; \
		brew install ollama; \
	fi
else
brew-deps:
	@echo "Non-macOS: install PortAudio and Ollama for your platform, then rerun 'make install ollama'."
	@command -v ollama >/dev/null 2>&1 || echo "  ollama not found on PATH"
endif

ollama: ollama-serve ollama-pull

ollama-serve:
	@command -v ollama >/dev/null 2>&1 || { \
		echo "ollama not found. Run 'make setup' first."; \
		exit 1; \
	}
	@if curl -sf "$(OLLAMA_URL)/api/tags" >/dev/null; then \
		echo "Ollama is already running at $(OLLAMA_URL)."; \
	else \
		echo "Starting Ollama server..."; \
		nohup ollama serve >/tmp/ollama-serve.log 2>&1 & \
		i=0; \
		while [ $$i -lt 40 ]; do \
			curl -sf "$(OLLAMA_URL)/api/tags" >/dev/null && break; \
			sleep 0.25; \
			i=$$((i + 1)); \
		done; \
		if curl -sf "$(OLLAMA_URL)/api/tags" >/dev/null; then \
			echo "Ollama is up."; \
		else \
			echo "Ollama failed to start. See /tmp/ollama-serve.log"; \
			exit 1; \
		fi; \
	fi

ollama-pull: ollama-serve
	@echo "Pulling $(LLM_MODEL) (no-op if already present)..."
	ollama pull $(LLM_MODEL)

run: install
	$(BIN)/pizza-agent

clean:
	rm -rf $(VENV) *.egg-info build dist
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
