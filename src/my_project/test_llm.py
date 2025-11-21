# src/my_project/test_llm.py
from dotenv import load_dotenv
load_dotenv()  # lee .env

from pydantic_ai import Agent

# OPENAI (si tuvieras cuota):
# agent = Agent(model="openai:gpt-4.1")

# GROQ (free tier; requiere GROQ_API_KEY en .env):
agent = Agent(model="groq:llama-3.1-8b-instant")

# En scripts: usa run_sync
resp = agent.run_sync("Dime un número entre 1 y 10")
print(resp.data)
