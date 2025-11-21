import asyncio
from dotenv import load_dotenv
from pydantic_ai import Agent, Tool
from my_project.tools import sumar

load_dotenv()

agent = Agent(
    model="groq:llama-3.1-8b-instant",
    tools=[Tool(sumar)],
    instructions="Eres un asistente técnico. Usa la herramienta cuando sea útil.",
)

async def main():
    print("=== TRACE (async iter) ===")

    async with agent.iter("¿Puedes usar la herramienta para sumar 10 y 35?") as stream:
        async for step in stream:
            print("---- STEP ----")
            print(step)

            # si el step tiene output final
            if hasattr(step, "output") and step.output is not None:
                print("\n=== OUTPUT FINAL ===")
                print(step.output)

if __name__ == "__main__":
    asyncio.run(main())
