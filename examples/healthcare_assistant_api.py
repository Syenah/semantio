from semantio.agent import Agent
import os

# Set your Groq API key
os.environ["GROQ_API_KEY"] = "YOUR-API-KEY"

# Create an agent with API enabled
agent = Agent(
    name="Healthcare Agent",
    description="Extract and structure medical information from the provided text into a JSON format used in healthcare",
    instructions=[
        "Always use medical terminology while creating json",
        "Extract and structure medical information from the provided text into a JSON format used in healthcare",
        "Don't stringify the JSON",
    ],
    model="Groq",
    show_tool_calls=True,
    user_name="Researcher",
    markdown=True,
    json_output=True,
    api=True,  # Enable API generation
    api_config={
        "host": "127.0.0.1",  # Custom API host
        "port": 8000,  # Custom API port
        "cors": {  # CORS configuration
            "allow_origins": ["http://localhost:3000", "https://example.com"],  # Allowed origins
            "allow_methods": ["GET", "POST"],  # Allowed HTTP methods
            "allow_headers": ["Authorization", "Content-Type"],  # Allowed headers
            "allow_credentials": True,  # Allow credentials (e.g., cookies)
        }
    }
)

# Run the API server
agent.run_api()