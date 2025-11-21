from dotenv import load_dotenv
from pydantic_ai import Agent, Tool
from pydantic import BaseModel
from my_project.tools import sumar, contar_palabras, es_palindromo

load_dotenv()

# --------------------------------------------------------------------
# 1) Modelo de salida: estructura clara para respuestas del agente
# --------------------------------------------------------------------
class EmbeddingInfo(BaseModel):
    definition: str
    intuition: str
    simple_example: str

# --------------------------------------------------------------------
# 2) Crear el agente
# --------------------------------------------------------------------
agent = Agent(
    model="groq:llama-3.1-8b-instant",

    # Herramientas disponibles
    tools=[Tool(sumar), Tool(contar_palabras), Tool(es_palindromo)],

    # Instrucciones que guían el comportamiento
    instructions=(
        "Eres un asistente técnico, claro y conciso.\n"
        "Si no sabes algo, responde 'no sé' y explica cómo verificar.\n"
        "Puedes usar herramientas cuando sea útil.\n"
        "Responde con la estructura definida en el output_type.\n"
    ),

    # Formato estructurado de salida
    output_type=EmbeddingInfo,
)

# --------------------------------------------------------------------
# 3) Función de ejecución
# --------------------------------------------------------------------
#"Cuántos huesos huesos tiene el cuerpo humano? y el pez nemo?"
def main() -> None:
    query = "Devuélveme: definición, intuición y un ejemplo simple de embedding."

    # Ejecuta de forma síncrona (ideal para scripts)
    resp = agent.run_sync(query)

    # Respuesta estructurada como objeto Python
    data: EmbeddingInfo = resp.output

    print("\n📌 Respuesta estructurada del agente:\n")
    print("Definición:", data.definition)
    print("Intuición:", data.intuition)
    print("Ejemplo:", data.simple_example)


# --------------------------------------------------------------------
# 4) Solo corre main() si llamas el script directamente
# --------------------------------------------------------------------
if __name__ == "__main__":
    main()