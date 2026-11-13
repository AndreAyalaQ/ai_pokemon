import logging

import click
from dotenv import load_dotenv

from pokemon_agent import run_pokemon_agent
from pokemon_chromadb import register_pokemon_chromadb
from pokemon_logging import configure_logging

logger = logging.getLogger(__name__)


@click.command()
@click.option("--ask", "question", required=False, default=None, type=str, help="Ask a question about Pokemons.")
@click.option("--ingest", "ingest", is_flag=True, default=False, help="Ingest pokemon lore to db.")
def main(question: str | None, ingest: bool) -> None:
    load_dotenv()
    configure_logging()

    if ingest:
        try:
            register_pokemon_chromadb()
        except Exception as exc:
            logger.exception("Ingestion failed")
            raise click.ClickException(f"Ingestion failed: {exc}") from exc
        return

    if not question:
        raise click.UsageError("Provide --ask <question> or use --ingest.")

    try:
        response = run_pokemon_agent(question)
    except Exception as exc:
        logger.exception("Failed to answer question: %s", question)
        raise click.ClickException(f"Failed to answer question: {exc}") from exc

    click.echo(response)


if __name__ == "__main__":
    main()
