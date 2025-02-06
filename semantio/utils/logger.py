import logging
import sys
from pathlib import Path

# Create a logs directory if it doesn't exist
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)

# Logger configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(logs_dir / "agent.log"),  # Log to a file
        logging.StreamHandler(sys.stdout),  # Log to the console
    ],
)

# Create a logger instance
logger = logging.getLogger("agent")