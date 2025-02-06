from semantio.agent import Agent
import os
os.environ["GROQ_API_KEY"] = "YOUR-API-KEY"  # Set the API key here

# Create an agent which can give crypto prices
agent = Agent(
    name="Agent",
    description="You are a agent that can give the user the perfect crypto price",
    instructions=[
        "Give the user the perfect crypto price",
    ],
    llm="Groq",
    show_tool_calls=True,
)

response = agent.print_response(
    "what is the current price of bitcoin?",
)
