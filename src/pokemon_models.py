from pydantic import BaseModel


class Pokemon(BaseModel):
    name: str
    id: int
    height: int
    weight: int
    types: list[str]
    abilities: list[str]
    stats: dict[str, int]


class PokemonNotFound(BaseModel):
    error: str


class FlavorTextEntry(BaseModel):
    flavor_text: str
    language: str
    version: str


class PokemonSpecies(BaseModel):
    name: str
    id: int
    color: str
    habitat: str | None
    flavor_text_entries: list[FlavorTextEntry]


class PokemonSpeciesNotFound(BaseModel):
    error: str


class PokemonEmbeddingRecord(BaseModel):
    id: int
    name: str
    color: str
    embedding_text: str


class ChromaHitMetadata(BaseModel):
    id: int
    name: str
    color: str


class ChromaHit(BaseModel):
    id: str
    text: str
    metadata: ChromaHitMetadata
    distance: float
