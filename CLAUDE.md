# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

There is no build system, linter, or test suite in this repo — it's a small standalone script
project. Run everything from `src/` (module imports are unqualified, e.g. `from pokemon_agent
import ...`, so `src/` must be the working directory or on `PYTHONPATH`).

```bash
# setup
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
ollama pull llama3.2            # chat model
ollama pull qwen3-embedding:4b  # embedding model

# ingest Pokémon lore into the local ChromaDB store (required before --ask can use lore)
python src/main.py --ingest

# ask a question
python src/main.py --ask "What type is Charizard?"
```

Ollama must be running locally for both ingestion (embeddings) and `--ask` (chat + embeddings).

## Architecture

This is a tool-calling LangChain agent, not a RAG-only or fine-tuned system. The agent
(`pokemon_agent.py`, model `llama3.2` via `ChatOllama`) is explicitly instructed to answer
**only** from tool output, never from its own pretrained Pokémon knowledge — this constraint
drives the tool design and system prompt wording, so preserve it when editing the prompt or
adding tools.

Two data paths feed the agent, and they don't overlap:

- **Structured facts (live PokéAPI calls, no caching):** `pokemon_tools.py` exposes
  `get_pokemon_by_type`, `get_pokemon_by_name`, `get_pokemon_by_id` as LangChain `@tool`s that hit
  `pokeapi.co` directly and parse responses into `pokemon_models.py` Pydantic models
  (`Pokemon` / `PokemonNotFound`). Use these for types, stats, and abilities.
- **Lore (local vector search, requires `--ingest` first):** `get_pokemon_lore` embeds the query
  via `pokemon_embedder.py` (`OllamaEmbeddings`, instruction-prefixed query template) and searches
  a persistent ChromaDB collection (`pokemon_chromadb.py`, stored at `src/.pokemon_chroma/`).
  `DISTANCE_THRESHOLD = 0.8` filters out weak matches unless a specific Pokémon `name` is given, in
  which case the filter is skipped and results are scoped with a `where={"name": ...}` clause
  instead. The ingestion side (`pokemon_data.py`) does a full species-list crawl of PokéAPI,
  dedupes flavor text across game versions, and writes one combined document per species.

The system prompt in `pokemon_agent.py` explicitly disambiguates elemental types (use
`get_pokemon_by_type`) from Pokédex genus/category phrasing like "seed" or "mouse" Pokémon (use
`get_pokemon_lore` instead) — this distinction is easy to regress if the prompt or tool
docstrings are reworded.

`main.py` is a thin Click CLI with two mutually exclusive modes: `--ingest` (populate ChromaDB,
exits) and `--ask <question>` (run the agent once, print the final message). Both load `.env` via
`python-dotenv` and configure logging through `pokemon_logging.py` (`LOG_LEVEL` env var, default
`INFO`).

All PokéAPI calls use a shared 10s timeout (`REQUEST_TIMEOUT_SECONDS`) and catch
`requests.exceptions.RequestException` / `KeyError` / `pydantic.ValidationError` to return a typed
"not found" model or empty list rather than raising — callers (the agent) always get a tool result,
never an exception, so the agent can try a different tool instead of failing outright.
