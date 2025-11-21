from dotenv import load_dotenv
from pydantic_ai import Agent, Tool
from pydantic import BaseModel

from my_project.tools import leer_pdf

load_dotenv()

class PDFSummary(BaseModel):
    resumen: str

agent = Agent(
    model="groq:llama-3.1-8b-instant",
    tools=[Tool(leer_pdf)],
    instructions=(
        "Eres un asistente experto en documentos. "
        "Puedes usar la herramienta leer_pdf cuando te lo pidan. "
        "Siempre resume con claridad."
    ),
    output_type=PDFSummary,
)

def main() -> None:
    pdf_path = "data/external/Resume (ESP) - Gerardo Sepúlveda.pdf"

    resp = agent.run_sync(
        f"Lee el PDF ubicado en '{pdf_path}' y dame un resumen breve."
    )

    print("\n=== RESUMEN ===")
    print(resp.output.resumen)

if __name__ == "__main__":
    main()
