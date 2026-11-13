import logging

import requests
from pydantic import ValidationError

from pokemon_models import PokemonEmbeddingRecord

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT_SECONDS = 10
SPECIES_LIST_URL = "https://pokeapi.co/api/v2/pokemon-species"

def list_pokemon_species_urls() -> list[str]:
    response = requests.get(SPECIES_LIST_URL, params={"limit": 1}, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    count = response.json()["count"]

    response = requests.get(SPECIES_LIST_URL, params={"limit": count}, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    return [result["url"] for result in response.json()["results"]]

def get_pokemon_data_for_embeding() -> list[PokemonEmbeddingRecord]:
    species_urls = list_pokemon_species_urls()
    logger.info("Discovered %d Pokémon species to fetch", len(species_urls))

    pokemon_data = []
    for url in species_urls:
        try:
            response = requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
            response.raise_for_status()
            pokemon_data.append(process_pokemon_species(response.json()))
        except (requests.exceptions.RequestException, KeyError, ValidationError):
            logger.warning("Failed to fetch Pokémon species '%s', skipping", url, exc_info=True)

    logger.info("Fetched %d/%d Pokémon species records", len(pokemon_data), len(species_urls))
    return pokemon_data

def clean_flavor_text(text):
    return text.replace("\n", " ").replace("\f", " ").strip()

def process_pokemon_species(raw_json) -> PokemonEmbeddingRecord:
    seen = set()
    unique_texts = []

    for entry in raw_json["flavor_text_entries"]:
        if entry["language"]["name"] != "en":
            continue
        cleaned = clean_flavor_text(entry["flavor_text"])
        if cleaned not in seen:
            seen.add(cleaned)
            unique_texts.append(cleaned)

    combined_flavor_text = " ".join(unique_texts)

    return PokemonEmbeddingRecord(
        id=raw_json["id"],
        name=raw_json["name"],
        color=raw_json["color"]["name"],
        embedding_text=f"{raw_json['name'].title()}. {combined_flavor_text}"
    )