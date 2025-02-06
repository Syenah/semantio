import os
os.environ["GROQ_API_KEY"] = "YOUR-API-KEY"  # Set the API key here

# Initialize the Agent
from semantio.agent import Agent

agent = Agent(
    name="Agent",
    description="general purpose agent",
    instructions=[
        "Always give a detailed description of the problem",
    ],
    llm="Groq",
    user_name="Semantio",
    markdown=True,
)
agent.print_response("What is the capital of France?")