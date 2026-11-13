import logging

from langchain_ollama import OllamaEmbeddings

from pokemon_models import PokemonEmbeddingRecord

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "qwen3-embedding:4b"
QUERY_INSTRUCTION = "Instruct: Given a question about a Pokémon, retrieve the Pokémon description that best answers it\nQuery:{query}"


def embed_pokemons_text(pokemon_data: list[PokemonEmbeddingRecord]) -> list[list[float]]:
    logger.info("Embedding %d Pokémon documents with '%s'", len(pokemon_data), EMBEDDING_MODEL)
    try:
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        return embeddings.embed_documents([pokemon.embedding_text for pokemon in pokemon_data])
    except Exception:
        logger.exception("Failed to embed Pokémon documents with '%s'. Is Ollama running?", EMBEDDING_MODEL)
        raise


def embed_query_text(question: str) -> list:
    logger.debug("Embedding query text with '%s'", EMBEDDING_MODEL)
    try:
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        return embeddings.embed_query(QUERY_INSTRUCTION.format(query=question))
    except Exception:
        logger.exception("Failed to embed query with '%s'. Is Ollama running?", EMBEDDING_MODEL)
        raise