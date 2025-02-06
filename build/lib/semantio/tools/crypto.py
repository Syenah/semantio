from .base_tool import BaseTool
import yfinance as yf
from typing import Dict, Any, List, Optional
from pydantic import Field
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CryptoPriceChecker(BaseTool):
    default_symbol: str = Field(default="BTC-USD", description="Default cryptocurrency symbol (e.g., 'BTC-USD').")
    enable_historical_data: bool = Field(default=True, description="Enable fetching historical data.")
    enable_multiple_symbols: bool = Field(default=True, description="Enable fetching data for multiple cryptocurrencies.")
    enable_metrics: bool = Field(default=True, description="Enable additional metrics like volume and market cap.")
    llm: Optional[Any] = Field(None, description="The LLM instance to use for symbol extraction.")

    def __init__(self, **kwargs):
        super().__init__(
            name="CryptoPriceChecker",
            description="Fetch real-time and historical cryptocurrency prices and metrics.",
            **kwargs
        )

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the cryptocurrency price check based on the provided input.

        Args:
            input_data (Dict[str, Any]): Input data containing the query and optional parameters.

        Returns:
            Dict[str, Any]: Cryptocurrency price and additional metrics.
        """
        try:
            query = input_data.get("query", "")
            symbols = self._extract_symbols(query) if query else [self.default_symbol]
            time_range = input_data.get("time_range", "1d")  # Default to 1 day
            metrics = input_data.get("metrics", ["price"])  # Default to price only

            results = {}

            for symbol in symbols:
                # Validate the symbol
                if not self._is_valid_symbol(symbol):
                    logger.warning(f"Invalid cryptocurrency symbol: {symbol}")
                    results[symbol] = {"error": "Invalid symbol"}
                    continue

                try:
                    crypto = yf.Ticker(symbol)
                    data = {}

                    # Fetch real-time price
                    if "price" in metrics:
                        history = crypto.history(period=time_range)
                        if not history.empty:
                            data["price"] = history["Close"].iloc[-1]
                        else:
                            data["price"] = "No data available"

                    # Fetch historical data
                    if self.enable_historical_data and "history" in metrics:
                        history = crypto.history(period=time_range)
                        if not history.empty:
                            data["history"] = history.to_dict()
                        else:
                            data["history"] = "No historical data available"

                    # Fetch additional metrics
                    if self.enable_metrics:
                        info = crypto.info
                        if "volume" in metrics:
                            data["volume"] = info.get("volume24Hr", "No volume data available")
                        if "market_cap" in metrics:
                            data["market_cap"] = info.get("marketCap", "No market cap data available")

                    results[symbol] = data
                except Exception as e:
                    results[symbol] = {"error": str(e)}

            return results
        except Exception as e:
            return {"error": str(e)}

    def _extract_symbols(self, query: str) -> List[str]:
        """
        Use the LLM to extract cryptocurrency symbols from the user's query.
        """
        if not self.llm:
            logger.error("LLM instance not available for symbol extraction.")
            return []

        # Create a prompt for the LLM
        prompt = f"""
        Extract the cryptocurrency symbols from the following user query. 
        Return the symbols as a comma-separated list in the format CRYPTO-FIAT (e.g., "BTC-USD,ETH-USD"). 
        If no symbols are found, return "None".

        User Query: "{query}"
        """

        try:
            # Call the LLM to generate the response
            response = self.llm.generate(prompt=prompt)
            logger.info(f"LLM response: {response}")
            symbols = response.strip().replace('"', '').replace("'", "")

            # Fallback: Extract symbols from verbose responses
            if ":" in symbols or "\n" in symbols:
                # Look for patterns like "BTC-USD,ETH-USD" in the response
                symbol_pattern = re.compile(r"\b[A-Z]{2,5}-[A-Z]{2,5}\b")
                symbols = ",".join(symbol_pattern.findall(symbols))
                logger.info(f"Extracted symbols: {symbols}")

            # Parse the response into a list of symbols
            if symbols.lower() == "none":
                return []
            return [s.strip() for s in symbols.split(",")]
        except Exception as e:
            logger.error(f"Failed to extract symbols: {e}")
            return []

    def _is_valid_symbol(self, symbol: str) -> bool:
        """
        Validate if the symbol is a valid cryptocurrency symbol.
        """
        # Check if the symbol follows the CRYPTO-FIAT format (e.g., BTC-USD, ETH-USD)
        if "-" not in symbol:
            return False
        crypto, fiat = symbol.split("-")
        return bool(crypto.strip()) and bool(fiat.strip())