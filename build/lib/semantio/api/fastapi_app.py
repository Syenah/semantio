from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Dict, List
from ..agent import Agent

def create_fastapi_app(agent: Agent, api_config: Optional[Dict] = None) -> FastAPI:
    """
    Create a FastAPI app for the given agent.

    Args:
        agent (Agent): The agent instance for which the API is being created.
        api_config (Optional[Dict]): Configuration for the API, including CORS settings.

    Returns:
        FastAPI: A FastAPI app with endpoints for interacting with the agent.
    """
    app = FastAPI()

    # Default CORS settings (allow all)
    cors_config = {
        "allow_origins": ["*"],
        "allow_credentials": True,
        "allow_methods": ["*"],
        "allow_headers": ["*"],
    }

    # Override default CORS settings with user-provided settings
    if api_config and "cors" in api_config:
        cors_config.update(api_config["cors"])

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_config["allow_origins"],
        allow_credentials=cors_config["allow_credentials"],
        allow_methods=cors_config["allow_methods"],
        allow_headers=cors_config["allow_headers"],
    )

    @app.post("/chat")
    async def chat(message: str):
        """
        Endpoint to interact with the agent.
        """
        response = agent.print_response(message=message)

        if agent.json_output:
            return response
        else:
            return {"response": response}

    @app.get("/tools")
    async def get_tools():
        """
        Endpoint to get the list of tools available to the agent.
        """
        return {"tools": agent.tools}

    @app.post("/load_image")
    async def load_image(image_url: str):
        """
        Endpoint to load an image from a URL.
        """
        try:
            image = agent.load_image_from_url(image_url)
            return {"status": "success", "image": "Image loaded successfully"}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    return app