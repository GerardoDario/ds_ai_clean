from dotenv import load_dotenv
from pydantic_ai import Agent, Tool
from pydantic import BaseModel
from my_project.tools import sumar

load_dotenv()

class EmbeddingInfo(BaseModel):
    definition: str
    intuition: str
    simple_example: str

agent = Agent(
    model="groq:llama-3.1-8b-instant",
    tools=[Tool(sumar)],
    instructions=(
        "Eres un asistente técnico, conciso. "
        #"Eres un asistente bromista. "
        "Cuando no sepas, di 'no sé' y sugiere cómo verificar."
    ),
    output_type=EmbeddingInfo,
)

def main() -> None:
    resp = agent.run_sync(
        "Cuántos huesos huesos tiene el cuerpo humano? y el pez nemo?"
    )
    data: EmbeddingInfo = resp.output
    print("Definición:", data.definition)
    print("Intuición:", data.intuition)
    print("Ejemplo:", data.simple_example)

if __name__ == "__main__":
    main()
