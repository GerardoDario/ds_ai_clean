from pydantic import BaseModel

class EmbeddingExplanation(BaseModel):
    definicion: str
    ejemplo: str
    casos_uso: list[str]
