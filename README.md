# Pokémon Agent

A command-line Pokémon assistant built with [LangChain](https://www.langchain.com/) and a local
[Ollama](https://ollama.com/) model. It answers questions about Pokémon by calling tools backed by
[PokéAPI](https://pokeapi.co/) for structured data (types, stats, abilities) and a local
[ChromaDB](https://www.trychroma.com/) vector store for semantic lore search. The agent never
relies on its own pretrained knowledge — every answer must come from a tool call.

## How it works

- **Structured lookups** (`get_pokemon_by_type`, `get_pokemon_by_name`, `get_pokemon_by_id`) query
  PokéAPI directly for types, stats, and abilities.
- **Lore search** (`get_pokemon_lore`) embeds the user's query with an Ollama embedding model and
  runs a similarity search against Pokédex flavor text stored in a local ChromaDB collection.
- The agent (`llama3.2` via Ollama) decides which tool to call based on the question, guided by a
  system prompt that restricts it to tool output only.

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com/) running locally, with these models pulled:
  ```bash
  ollama pull llama3.2
  ollama pull qwen3-embedding:4b
  ```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root if you need to override defaults (see
[Configuration](#configuration)).

## Usage

Run commands from the `src/` directory (or adjust `PYTHONPATH` accordingly).

**Ingest Pokémon lore into the local vector store** (required once before asking lore questions):

```bash
python src/main.py --ingest
```

This fetches every Pokémon species from PokéAPI and embeds their English flavor text into
`src/.pokemon_chroma/`.

**Ask a question:**

```bash
python src/main.py --ask "What type is Charizard?"
python src/main.py --ask "What Pokémon is described as a mouse?"
```

## Configuration

Environment variables (loaded from `.env` via `python-dotenv`):

| Variable    | Default | Description                                      |
|-------------|---------|---------------------------------------------------|
| `LOG_LEVEL` | `INFO`  | Logging verbosity (`DEBUG`, `INFO`, `WARNING`, …) |

> **Note:** Never commit real API keys or secrets in `.env` — it's excluded via `.gitignore`.

## Project structure

```
src/
├── main.py              # CLI entry point (--ask / --ingest)
├── pokemon_agent.py     # Agent construction and system prompt
├── pokemon_tools.py     # LangChain tools exposed to the agent
├── pokemon_data.py      # Bulk PokéAPI fetching for ingestion
├── pokemon_embedder.py  # Ollama embedding helpers
├── pokemon_chromadb.py  # ChromaDB ingestion and similarity search
├── pokemon_models.py    # Pydantic models
└── pokemon_logging.py   # Logging configuration
```
