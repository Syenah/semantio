from .fastapi_app import create_fastapi_app  # Import the factory function

class APIGenerator:
    def __init__(self, assistant):
        """
        Initialize the APIGenerator with the given assistant.

        Args:
            assistant: The assistant instance for which the API is being created.
        """
        self.assistant = assistant
        self.app = create_fastapi_app(assistant, assistant.api_config)  # Pass api_config to create_fastapi_app

    def run(self, host: str = "0.0.0.0", port: int = 8000):
        """
        Run the FastAPI app.

        Args:
            host (str): The host address to run the API server on. Default is "0.0.0.0".
            port (int): The port to run the API server on. Default is 8000.
        """
        import uvicorn
        uvicorn.run(self.app, host=host, port=port)