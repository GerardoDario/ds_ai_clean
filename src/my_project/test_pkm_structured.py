from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic import BaseModel

load_dotenv()

# --- MODELO ESTRUCTURADO ---
class PokemonInfo(BaseModel):
    name: str
    type: str
    strong_against: list[str]
    weak_against: list[str]

# --- AGENTE ---
agent = Agent(
    model="groq:llama-3.1-8b-instant",
    instructions=(
        "Responde SIEMPRE usando el modelo PokémonInfo. "
        "No inventes campos adicionales. "
        "Si no puedes responder, di 'no sé' según la estructura."
    ),
    output_type=PokemonInfo,
)

def main() -> None:
    resp = agent.run_sync("Dame la información del Pokémon Charizard.")
    data: PokemonInfo = resp.output
    print(data)

if __name__ == "__main__":
    main()
