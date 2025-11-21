import PyPDF2
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic import BaseModel
import math

load_dotenv()

PDF_PATH = "data/external/cuentos-cortos-los-tres-cerditos.pdf"

# --------------------------------------------------------
# 1. Modelo estructurado para la salida
# --------------------------------------------------------

class DetailedSummary(BaseModel):
    resumen_general: str
    personajes: list[str]
    trama_detallada: str
    moraleja: str


# --------------------------------------------------------
# 2. Función para extraer texto del PDF
# --------------------------------------------------------

def extract_pdf_text(path: str) -> str:
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text


# --------------------------------------------------------
# 3. Chunking del texto (muy importante)
# --------------------------------------------------------

def chunk_text(text: str, max_chars=1500):
    chunks = []
    current = 0
    total = len(text)
    while current < total:
        chunks.append(text[current: current + max_chars])
        current += max_chars
    return chunks


# --------------------------------------------------------
# 4. Crear agente especializado en resúmenes
# --------------------------------------------------------

agent = Agent(
    model="groq:llama-3.1-8b-instant",
    instructions=(
        "Eres un experto en comprensión lectora. "
        "Debes generar un resumen extremadamente detallado, ordenado y fiel al texto. "
        "Es obligatorio que el resumen incluya: "
        "- Un resumen general completo\n"
        "- Lista de personajes relevantes\n"
        "- Trama explicada paso a paso\n"
        "- La moraleja o enseñanza del texto\n"
        "Si el texto es infantil, mantén un tono pedagógico."
    ),
    output_type=DetailedSummary,
)


# --------------------------------------------------------
# 5. Función principal de resumen
# --------------------------------------------------------

def summarize_pdf(path=PDF_PATH) -> DetailedSummary:
    text = extract_pdf_text(path)
    chunks = chunk_text(text)

    partial_summaries = []

    # Procesar chunk por chunk
    for i, chunk in enumerate(chunks):
        print(f"🔹 Procesando chunk {i+1}/{len(chunks)}...")
        resp = agent.run_sync(
            f"Resume detalladamente el siguiente fragmento:\n\n{chunk}"
        )
        partial_summaries.append(resp.output)

    # Combinar todos los parciales en uno solo
    combined_text = "\n\n".join([p.trama_detallada for p in partial_summaries])

    # Crear un resumen final global
    final = agent.run_sync(
        f"Combina todos los resúmenes parciales y produce un resumen global EXTREMADAMENTE detallado.\n\n"
        f"Resúmenes previos:\n{combined_text}"
    )

    return final.output


# --------------------------------------------------------
# 6. Ejecutar desde terminal
# --------------------------------------------------------

if __name__ == "__main__":
    final = summarize_pdf()
    print("\n=== RESUMEN DETALLADO ===\n")
    print("Resumen general:", final.resumen_general)
    print("\nPersonajes:", final.personajes)
    print("\nTrama detallada:", final.trama_detallada)
    print("\nMoraleja:", final.moraleja)
