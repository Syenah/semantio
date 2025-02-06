from semantio.agent import Agent
import os
os.environ["GROQ_API_KEY"] = "YOUR-API-KEY"  # Set the API key here

# Create an agent which can do multiple tasks
agent = Agent(
    name="Agent",
    description="You are a agent that can give the user perfect answers by using the tools",
    instructions=[
        "Give the user the perfect answer",
    ],
    llm="Groq",
    show_tool_calls=True,
)

response = agent.print_response(
    "what is the current price of bitcoin and give me the top 10 news articles about bitcoin?",
)
