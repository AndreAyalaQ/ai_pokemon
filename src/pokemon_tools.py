import logging

from langchain.tools import tool
from pydantic import ValidationError

from pokemon_chromadb import get_vector_from_chromadb
from pokemon_models import Pokemon, PokemonNotFound, PokemonSpecies, PokemonSpeciesNotFound, FlavorTextEntry
import requests

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT_SECONDS = 10


@tool
def get_pokemon_by_type(pokemon_type: str) -> list:
    """
    Fetches a list of Pokémon names based on the specified type.

    Args:
        pokemon_type (str): The type of Pokémon to search for (e.g., "fire", "water").

    Returns:
        list: A list of Pokémon names that match the specified type.
    """

    logger.info("Fetching Pokémon by type: %s", pokemon_type)
    url = f"https://pokeapi.co/api/v2/type/{pokemon_type.lower()}"

    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        data = response.json()
        return [pokemon["pokemon"]["name"] for pokemon in data["pokemon"]]
    except (requests.exceptions.RequestException, KeyError) as exc:
        logger.error("Failed to fetch Pokémon type '%s': %s", pokemon_type, exc)
        return []

@tool
def get_pokemon_by_name(pokemon_name: str) -> Pokemon | PokemonNotFound:
    """
    Fetches detailed information about a Pokémon based on its name.

    Args:
        pokemon_name (str): The name of the Pokémon to search for (e.g., "pikachu").

    Returns:
        Pokemon | PokemonNotFound: The Pokémon's details, or an error if it wasn't found.
    """

    logger.info("Fetching Pokémon by name: %s", pokemon_name)
    url = f"https://pokeapi.co/api/v2/pokemon/{pokemon_name.lower()}"

    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        data = response.json()
        return Pokemon(
            name=data["name"],
            id=data["id"],
            height=data["height"],
            weight=data["weight"],
            types=[t["type"]["name"] for t in data["types"]],
            abilities=[a["ability"]["name"] for a in data["abilities"]],
            stats={s["stat"]["name"]: s["base_stat"] for s in data["stats"]},
        )
    except (requests.exceptions.RequestException, KeyError, ValidationError) as exc:
        logger.error("Failed to fetch Pokémon '%s': %s", pokemon_name, exc)
        return PokemonNotFound(error=f"Pokémon '{pokemon_name}' not found.")

@tool
def get_pokemon_by_id(pokemon_id: int) -> Pokemon | PokemonNotFound:
    """
    Fetches detailed information about a Pokémon based on its ID.

    Args:
        pokemon_id (int): The ID of the Pokémon to search for (e.g., 25 for Pikachu).

    Returns:
        Pokemon | PokemonNotFound: The Pokémon's details, or an error if it wasn't found.
    """

    logger.info("Fetching Pokémon by id: %s", pokemon_id)
    url = f"https://pokeapi.co/api/v2/pokemon/{pokemon_id}"

    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        data = response.json()
        return Pokemon(
            name=data["name"],
            id=data["id"],
            height=data["height"],
            weight=data["weight"],
            types=[t["type"]["name"] for t in data["types"]],
            abilities=[a["ability"]["name"] for a in data["abilities"]],
            stats={s["stat"]["name"]: s["base_stat"] for s in data["stats"]},
        )
    except (requests.exceptions.RequestException, KeyError, ValidationError) as exc:
        logger.error("Failed to fetch Pokémon with ID '%s': %s", pokemon_id, exc)
        return PokemonNotFound(error=f"Pokémon with ID '{pokemon_id}' not found.")

@tool
def get_all_pokemon_names() -> list:
    """
    Fetches a list of all Pokémon names.

    Returns:
        list: A list of all Pokémon names.
    """

    logger.info("Fetching all Pokémon names")
    url = "https://pokeapi.co/api/v2/pokemon?limit=10000"

    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        data = response.json()
        return [pokemon["name"] for pokemon in data["results"]]
    except (requests.exceptions.RequestException, KeyError) as exc:
        logger.error("Failed to fetch all Pokémon names: %s", exc)
        return []


def get_pokemon_species_info(pokemon_name: str) -> PokemonSpecies | PokemonSpeciesNotFound:
    """
    Fetches detailed species information about a Pokémon based on its name.

    Args:
        pokemon_name (str): The name of the Pokémon to search for (e.g., "pikachu").

    Returns:
        PokemonSpecies | PokemonSpeciesNotFound: The species details, or an error if not found.
    """

    logger.info("Fetching Pokémon species info: %s", pokemon_name)
    url = f"https://pokeapi.co/api/v2/pokemon-species/{pokemon_name.lower()}"

    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        data = response.json()
        return PokemonSpecies(
            name=data["name"],
            id=data["id"],
            color=data["color"]["name"],
            habitat=data["habitat"]["name"] if data["habitat"] else None,
            flavor_text_entries=[
                FlavorTextEntry(
                    flavor_text=entry["flavor_text"],
                    language=entry["language"]["name"],
                    version=entry["version"]["name"],
                )
                for entry in data["flavor_text_entries"]
                if entry["language"]["name"] == "en"
            ],
        )
    except (requests.exceptions.RequestException, KeyError, ValidationError) as exc:
        logger.error("Failed to fetch Pokémon species '%s': %s", pokemon_name, exc)
        return PokemonSpeciesNotFound(error=f"Pokémon species '{pokemon_name}' not found.")

@tool
def get_pokemon_lore(query: str, name: str | None = None) -> str:
    """
    Fetches lore information about a Pokémon based on a query, using semantic
    similarity search over Pokémon flavor text.

    Args:
        query (str): Pass the user's full question or a descriptive phrase
            (e.g. "a small yellow electric mouse pokemon"), not a single
            keyword. The query is embedded and matched by semantic
            similarity, so more context produces better matches.
        name (str | None): If you already know which Pokémon the user is
            asking about (e.g. "pikachu"), pass its name to restrict the
            search to that Pokémon's lore instead of searching across all
            Pokémon.

    Returns:
        str: A string containing the lore information about the specified Pokémon.
    """

    logger.info("Fetching Pokémon lore for query=%r name=%r", query, name)
    hits = get_vector_from_chromadb(query, name=name.lower() if name else None)

    if not hits:
        return f"No lore found for the query '{query}'."

    return hits[0].text
    