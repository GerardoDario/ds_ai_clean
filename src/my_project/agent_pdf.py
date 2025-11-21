from dotenv import load_dotenv
from pydantic_ai import Agent, Tool
from pydantic import BaseModel

from my_project.tools import leer_pdf

load_dotenv()

class Summary(BaseModel):
    resumen_detallado: str

agent = Agent(
    model="groq:llama-3.1-8b-instant",
    instructions=(
        "Eres un analista experto en textos. "
        "Genera resúmenes profundos, precisos y detallados. "
        "Incluye personajes, conflicto, estructura, moraleja y hechos clave. "
        "No inventes nada que no esté en el texto."
    ),
    output_type=Summary,
)

def main() -> None:
    #pdf_path = "data/external/Resume (ESP) - Gerardo Sepúlveda.pdf"
    pdf_path = "data/external/cuentos-cortos-los-tres-cerditos.pdf"

    resp = agent.run_sync(
        f"Lee el PDF ubicado en '{pdf_path}' y dame un resumen breve."
    )

    print("\n=== RESUMEN ===")
    print(resp.output.resumen_detallado)

if __name__ == "__main__":
    main()
