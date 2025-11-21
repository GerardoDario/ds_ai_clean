from pydantic_ai import Agent, Tool
from dotenv import load_dotenv
from my_project.tools import sumar

load_dotenv()

agent = Agent(
    model="groq:llama-3.1-8b-instant",
    tools=[Tool(sumar)]
)

resp = agent.run_sync("¿Puedes usar la herramienta para sumar 10 y 35?")
print(resp.output)
