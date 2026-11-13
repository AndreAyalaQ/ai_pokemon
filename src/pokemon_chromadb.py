import logging

import chromadb
from pokemon_data import get_pokemon_data_for_embeding
from pokemon_embedder import embed_pokemons_text, embed_query_text
from pokemon_models import ChromaHit, ChromaHitMetadata

logger = logging.getLogger(__name__)

DISTANCE_THRESHOLD = 0.8
CHROMA_DIR = ".pokemon_chroma"

def register_pokemon_chromadb() -> None:
    logger.info("Starting Pokémon ChromaDB ingestion")

    try:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        collection = client.get_or_create_collection("pokemon_collection")

        pokemon_data = get_pokemon_data_for_embeding()
        logger.info("Fetched %d Pokémon records to embed", len(pokemon_data))

        text_original = [pokemon.embedding_text for pokemon in pokemon_data]
        pokemon_vectors = embed_pokemons_text(pokemon_data)
        ids = [str(pokemon.id) for pokemon in pokemon_data]

        metadatas = [
            {
                "id": pokemon.id,
                "name": pokemon.name,
                "color": pokemon.color
            }
            for pokemon in pokemon_data
        ]

        collection.upsert(ids=ids, embeddings=pokemon_vectors, metadatas=metadatas, documents=text_original)
    except Exception:
        logger.exception("Failed to register Pokémon data in ChromaDB")
        raise

    logger.info("Pokémon data registered successfully in ChromaDB (%d records)", len(pokemon_data))


def get_vector_from_chromadb(query: str, name: str = None) -> list[ChromaHit]:
    logger.debug("Searching Pokémon lore for query=%r name=%r", query, name)

    try:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        collection = client.get_collection("pokemon_collection")
        vector = embed_query_text(query)

        results = collection.query(
            query_embeddings=[vector],
            where={"name": name} if name else None
        )
    except Exception:
        logger.exception(
            "Failed to query Pokémon lore for '%s'. Has the DB been ingested with --ingest?", query
        )
        return []

    hits = []
    for id, doc, metadata, distance in zip(results['ids'][0], results['documents'][0], results['metadatas'][0], results['distances'][0]):
        if not name and distance > DISTANCE_THRESHOLD:
            continue
        hits.append(ChromaHit(id=id, text=doc, metadata=ChromaHitMetadata(**metadata), distance=distance))

    logger.debug("Found %d lore hits within distance threshold", len(hits))
    return hits
