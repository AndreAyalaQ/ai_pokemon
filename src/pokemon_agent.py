import logging

from langchain.agents import create_agent
from langchain_ollama import ChatOllama

from pokemon_tools import get_pokemon_by_type, get_pokemon_by_name, get_pokemon_by_id, get_pokemon_lore

logger = logging.getLogger(__name__)

CHAT_MODEL = "llama3.2"
AGENT_TEMPERATURE = 0.5
AGENT_MAX_ITERATIONS = 10 * 2 + 1
SYSTEM_PROMPT = """
    You are a Pokémon assistant. Answer ONLY using data returned by your tools.
    Never use your own pre-existing knowledge about Pokémon, even if you are confident it's correct.

    Tool selection guidance:
    - get_pokemon_by_type: elemental types only (e.g. "fire", "water", "electric"). Do NOT use it
      for Pokédex categories/genus like "seed", "mouse", or "flame" Pokémon — those are lore,
      use get_pokemon_lore instead.
    - get_pokemon_by_name / get_pokemon_by_id: a specific Pokémon's stats, types, and abilities.
    - get_pokemon_lore: descriptive, category, or "what Pokémon is/does X" style questions.

    If a tool call fails or returns no useful data, try a different tool before giving up.
    Only after exhausting reasonable tool options should you tell the user you don't have that
    information. Never fill the gap with your own memorized knowledge.
"""


def build_pokemon_agent():
    logger.debug("Building Pokémon agent with model '%s'", CHAT_MODEL)
    tools = [get_pokemon_by_type, get_pokemon_by_name, get_pokemon_by_id, get_pokemon_lore]

    return create_agent(
        model=ChatOllama(model=CHAT_MODEL, temperature=AGENT_TEMPERATURE),
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
    )


def run_pokemon_agent(query: str) -> str:
    logger.info("Running Pokémon agent for query: %s", query)

    try:
        agent = build_pokemon_agent()
        result = agent.invoke(
            {"messages": [{"role": "user", "content": query}]},
            config={"recursion_limit": AGENT_MAX_ITERATIONS},
        )
    except Exception:
        logger.exception("Pokémon agent failed for query: %s", query)
        raise

    logger.info("Pokémon agent finished for query: %s", query)
    return result["messages"][-1].content
