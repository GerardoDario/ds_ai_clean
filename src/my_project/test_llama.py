from dotenv import load_dotenv
load_dotenv()

from pydantic_ai import Agent

agent = Agent("groq:llama-3.1-8b-instant")
result = agent.run_sync("hola")     # en script: versión síncrona
print(result.output)