from dotenv import load_dotenv
from pydantic_ai import Agent, Tool
from my_project.tools import sumar

load_dotenv()

agent = Agent(
    model="groq:llama-3.1-8b-instant",
    tools=[Tool(sumar)],
    instructions="Eres un asistente técnico. Usa la herramienta cuando sea necesario.",
)

def main():
    resp = agent.run_sync("¿Puedes usar la herramienta para sumar 10 y 35?")
    print("Resultado:", resp.output)

if __name__ == "__main__":
    main()
