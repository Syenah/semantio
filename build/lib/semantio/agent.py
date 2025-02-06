from typing import Optional, List, Dict, Union, Iterator, Any
from pydantic import BaseModel, Field, ConfigDict
from PIL.Image import Image
import requests
import logging
import re
import io
import json
from .rag import RAG
from .llm.base_llm import BaseLLM
from .knowledge_base.retriever import Retriever
from .knowledge_base.vector_store import VectorStore
from sentence_transformers import SentenceTransformer, util
from fuzzywuzzy import fuzz
from .tools.base_tool import BaseTool
from pathlib import Path
import importlib
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Agent(BaseModel):
    # -*- Agent settings
    name: Optional[str] = Field(None, description="Name of the agent.")
    description: Optional[str] = Field(None, description="Description of the agent's role.")
    instructions: Optional[List[str]] = Field(None, description="List of instructions for the agent.")
    model: Optional[str] = Field(None, description="This one is not in the use.")
    show_tool_calls: bool = Field(False, description="Whether to show tool calls in the response.")
    markdown: bool = Field(False, description="Whether to format the response in markdown.")
    tools: Optional[List[BaseTool]] = Field(None, description="List of tools available to the agent.")
    user_name: Optional[str] = Field("User", description="Name of the user interacting with the agent.")
    emoji: Optional[str] = Field(":robot:", description="Emoji to represent the agent in the CLI.")
    rag: Optional[RAG] = Field(None, description="RAG instance for context retrieval.")
    knowledge_base: Optional[Any] = Field(None, description="Knowledge base for domain-specific information.")
    llm: Optional[str] = Field(None, description="The LLM provider to use (e.g., 'groq', 'openai', 'anthropic').")
    llm_model: Optional[str] = Field(None, description="The specific model to use for the LLM provider.")
    llm_instance: Optional[BaseLLM] = Field(None, description="The LLM instance to use.")
    json_output: bool = Field(False, description="Whether to format the response as JSON.")
    api: bool = Field(False, description="Whether to generate an API for the agent.")
    api_config: Optional[Dict] = Field(
        None,
        description="Configuration for the API (e.g., host, port, authentication).",
    )
    api_generator: Optional[Any] = Field(None, description="The API generator instance.")
    expected_output: Optional[Union[str, Dict]] = Field(None, description="The expected format or structure of the output.")
    semantic_model: Optional[Any] = Field(None, description="SentenceTransformer model for semantic matching.")
    team: Optional[List['Agent']] = Field(None, description="List of assistants in the team.")
    auto_tool: bool = Field(False, description="Whether to automatically detect and call tools.")
    
    # Allow arbitrary types
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize the model and tools here if needed
        self._initialize_model()
        # Initialize tools as an empty list if not provided
        if self.tools is None:
            self.tools = []
        # Automatically discover and register tools if auto tool is enabled
        if self.auto_tool and not self.tools:
            self.tools = self._discover_tools()
        # Pass the LLM instance to each tool
        for tool in self.tools:
            tool.llm = self.llm_instance
        # Initialize the SentenceTransformer model for semantic matching
        self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        # Initialize RAG if not provided
        if self.rag is None:
            self.rag = self._initialize_default_rag()
        # Automatically generate API if api=True
        if self.api:
            self._generate_api()


    def _generate_response_from_image(self,message: str, image: Union[str, Image], markdown: bool = False, **kwargs) -> str:
        """
        Send the image to the LLM for analysis if the LLM supports vision.
        Supports both local images (PIL.Image) and image URLs.
        """
        try:
            # Check if the LLM supports vision
            if not self.llm_instance or not self.llm_instance.supports_vision:
                raise ValueError("Vision is not supported for the current model.")
            prompt = self._build_prompt(message, context=None)
            # Handle image URL
            if isinstance(image, str) and image.startswith("http"):
                # Directly pass the URL to the LLM
                return self.llm_instance.generate_from_image_url(prompt,image, **kwargs)

            # Handle local image (PIL.Image)
            elif isinstance(image, Image):
                # Convert the image to bytes
                if image.mode == "RGBA":
                    image = image.convert("RGB")  # Convert RGBA to RGB
                image_bytes = io.BytesIO()
                image.save(image_bytes, format="JPEG")  # Save as PNG (or any supported format)
                image_bytes = image_bytes.getvalue()

                # Generate response using base64-encoded image bytes
                return self.llm_instance.generate_from_image(prompt,image_bytes, **kwargs)

            else:
                raise ValueError("Unsupported image type. Provide either a URL or a PIL.Image.")

        except Exception as e:
            logger.error(f"Failed to generate response from image: {e}")
            return f"An error occurred while processing the image: {e}"
        
    def _discover_tools(self) -> List[BaseTool]:
        """
        Automatically discover and register tools from the 'tools' directory.
        """
        tools = []
        tools_dir = Path(__file__).parent / "tools"

        if not tools_dir.exists():
            logger.warning(f"Tools directory not found: {tools_dir}")
            return tools

        # Iterate over all Python files in the 'tools' directory
        for file in tools_dir.glob("*.py"):
            if file.name == "base_tool.py":
                continue  # Skip the base tool file

            try:
                # Import the module
                module_name = file.stem
                module = importlib.import_module(f"semantio.tools.{module_name}")

                # Find all classes that inherit from BaseTool
                for name, obj in module.__dict__.items():
                    if isinstance(obj, type) and issubclass(obj, BaseTool) and obj != BaseTool:
                        # Instantiate the tool and add it to the list
                        tools.append(obj())
                        logger.info(f"Registered tool: {obj.__name__}")
            except Exception as e:
                logger.error(f"Failed to load tool from {file}: {e}")

        return tools
    
    def _get_tool_descriptions(self) -> str:
        """Generate a description of all available tools for the LLM prompt."""
        return "\n".join(
            f"{tool.name}: {tool.description}" for tool in self.tools
        )
    
    def _initialize_model(self):
        """Initialize the model based on the provided configuration."""
        if self.llm_instance is not None:
            return  # LLM is already initialized, do nothing
        if self.llm is None:
            raise ValueError("llm must be specified.")
    
        # Get the API key from the environment or the provided configuration
        api_key = getattr(self, 'api_key', None) or os.getenv(f"{self.llm.upper()}_API_KEY")
    
        # Map LLM providers to their respective classes and default models
        llm_providers = {
            "groq": {
                "class": "GroqLlm",
                "default_model": "mixtral-8x7b-32768",
            },
            "openai": {
                "class": "OpenAILlm",
                "default_model": "gpt-4o",
            },
            "anthropic": {
                "class": "AnthropicLlm",
                "default_model": "claude-2.1",
            },
            "deepseek": {
                "class": "DeepSeekLLM",
                "default_model": "deepseek-chat",
            },
            "gemini": {
                "class": "GeminiLLM",
                "default_model": "gemini-1.5-flash",
            },
            "mistral": {
                "class": "MistralLLM",
                "default_model": "mistral-large-latest",
            },
        }

        # Normalize the LLM provider name (case-insensitive)
        llm_provider = self.llm.lower()

        if llm_provider not in llm_providers:
            raise ValueError(f"Unsupported LLM provider: {self.llm}. Supported providers are: {list(llm_providers.keys())}")

        # Get the LLM class and default model
        llm_config = llm_providers[llm_provider]
        llm_class_name = llm_config["class"]
        default_model = llm_config["default_model"]

        # Use the user-provided model or fallback to the default model
        model_to_use = self.llm_model or default_model

        # Dynamically import and initialize the LLM class
        module_name = f"semantio.llm.{llm_provider}"
        llm_module = importlib.import_module(module_name)
        llm_class = getattr(llm_module, llm_class_name)
        self.llm_instance = llm_class(model=model_to_use, api_key=api_key)

    def _initialize_default_rag(self) -> RAG:
        """Initialize a default RAG instance with a dummy vector store."""
        vector_store = VectorStore()
        retriever = Retriever(vector_store)
        return RAG(retriever)

    def print_response(
        self,
        message: Optional[Union[str, Image, List, Dict]] = None,
        stream: bool = False,
        markdown: bool = False,
        team: Optional[List['Agent']] = None,
        **kwargs,
    ) -> Union[str, Dict]:  # Add return type hint
        """Print the agent's response to the console and return it."""

        if stream:
            # Handle streaming response
            response = ""
            for chunk in self._stream_response(message, markdown=markdown, **kwargs):
                print(chunk)
                response += chunk
            return response
        else:
            # Generate and return the response
            response = self._generate_response(message, markdown=markdown, team=team, **kwargs)
            print(response)  # Print the response to the console
            return response


    def _stream_response(self, message: str, markdown: bool = False, **kwargs) -> Iterator[str]:
        """Stream the agent's response."""
        # Simulate streaming by yielding chunks of the response
        response = self._generate_response(message, markdown=markdown, **kwargs)
        for chunk in response.split():
            yield chunk + " "

    def register_tool(self, tool: BaseTool):
        """Register a tool for the agent."""
        if self.tools is None:
            self.tools = []
        self.tools.append(tool)
    
    def _analyze_query_and_select_tools(self, query: str) -> List[Dict[str, Any]]:
        """
        Use the LLM to analyze the query and dynamically select tools.
        Returns a list of tool calls, each with the tool name and input.
        """
        # Create a prompt for the LLM to analyze the query and select tools
        prompt = f"""
        You are an AI agent that helps analyze user queries and select the most appropriate tools.
        Below is a list of available tools and their functionalities:

        {self._get_tool_descriptions()}

        For the following query, analyze the intent and select the most appropriate tools.
        Respond with a JSON array of tool names and their inputs.
        If no tool is suitable, respond with an empty array.

        Query: "{query}"

        Respond in the following JSON format:
        [
            {{
                "tool": "tool_name",
                "input": {{
                    "query": "user_query",
                    "context": "optional_context"
                }}
            }}
        ]
        """

        try:
            # Call the LLM to generate the response
            response = self.llm_instance.generate(prompt=prompt)
            # Parse the response as JSON
            tool_calls = json.loads(response)
            return tool_calls
        except Exception as e:
            logger.error(f"Failed to analyze query and select tools: {e}")
            return []
    

    def _generate_response(self, message: str, markdown: bool = False, team: Optional[List['Agent']] = None, **kwargs) -> str:
        """Generate the agent's response, including tool execution and context retrieval."""
        # Use the specified team if provided
        if team is not None:
            return self._generate_team_response(message, team, markdown=markdown, **kwargs)

        # Initialize tool_outputs as an empty dictionary
        tool_outputs = {}
        responses = []
        tool_calls = []

        # Use the LLM to analyze the query and dynamically select tools when auto_tool is enabled
        if self.auto_tool:
            tool_calls = self._analyze_query_and_select_tools(message)
        else:
            # Check if tools are provided
            if self.tools:
                tool_calls = [
                    {
                        "tool": tool.name,
                        "input": {
                            "query": message,  # Use the message as the query
                            "context": None,  # No context provided by default
                        }
                    }
                    for tool in self.tools
                ]

        # Execute tools if any are detected
        if tool_calls:
            for tool_call in tool_calls:
                tool_name = tool_call["tool"]
                tool_input = tool_call["input"]

                # Find the tool
                tool = next((t for t in self.tools if t.name.lower() == tool_name.lower()), None)
                if tool:
                    try:
                        # Execute the tool
                        tool_output = tool.execute(tool_input)
                        response = f"Tool '{tool_name}' executed. Output: {tool_output}"
                        if self.show_tool_calls:
                            response = f"**Tool Called:** {tool_name}\n\n{response}"
                        responses.append(response)

                        # Store the tool output for collaboration
                        tool_outputs[tool_name] = tool_output
                    except Exception as e:
                        logger.error(f"Tool called:** {tool_name}\n\n{response}")
                        responses.append(f"An error occurred while executing the tool '{tool_name}': {e}")
                else:
                    responses.append(f"Tool '{tool_name}' not found.")

        # If multiple tools were executed, combine their outputs for analysis
        if tool_outputs:
            try:
                # Prepare the context for the LLM
                context = {
                    "tool_outputs": tool_outputs,
                    "rag_context": self.rag.retrieve(message) if self.rag else None,
                    "knowledge_base_context": self._find_all_relevant_keys(message, self._flatten_data(self.knowledge_base)) if self.knowledge_base else None,
                }

                # Generate a response using the LLM
                llm_response = self.llm_instance.generate(prompt=message, context=context, **kwargs)
                responses.append(f"**Analysis:**\n\n{llm_response}")
            except Exception as e:
                logger.error(f"Failed to generate LLM response: {e}")
                responses.append(f"An error occurred while generating the analysis: {e}")
        if not self.tools and not tool_calls:
            # If no tools were executed, proceed with the original logic
            # Retrieve relevant context using RAG
            rag_context = self.rag.retrieve(message) if self.rag else None
            # Retrieve relevant context from the knowledge base (API result)
            knowledge_base_context = None
            if self.knowledge_base:
                # Flatten the knowledge base
                flattened_data = self._flatten_data(self.knowledge_base)
                # Find all relevant key-value pairs in the knowledge base
                relevant_values = self._find_all_relevant_keys(message, flattened_data)
                if relevant_values:
                    knowledge_base_context = ", ".join(relevant_values)

            # Combine both contexts (RAG and knowledge base)
            context = {
                "rag_context": rag_context,
                "knowledge_base_context": knowledge_base_context,
            }
            # Prepare the prompt with instructions, description, and context
            prompt = self._build_prompt(message, context)

            # Generate the response using the LLM
            response = self.llm_instance.generate(prompt=prompt, context=context, **kwargs)

            # Format the response based on the json_output flag
            if self.json_output:
                response = self._format_response_as_json(response)

            # Validate the response against the expected_output
            if self.expected_output:
                response = self._validate_response(response)

            if markdown:
                return f"**Response:**\n\n{response}"
            return response
        # Combine all responses into a single string
        return "\n\n".join(responses)
    
    def _generate_team_response(self, message: str, team: List['Agent'], markdown: bool = False, **kwargs) -> str:
        """Generate a response using a team of assistants."""
        responses = []
        for agent in team:
            response = agent.print_response(message, markdown=markdown, **kwargs)
            responses.append(f"**{agent.name}:**\n\n{response}")
        return "\n\n".join(responses)

    def _build_prompt(self, message: str, context: Optional[List[Dict]]) -> str:
        """Build the prompt using instructions, description, and context."""
        prompt_parts = []

        # Add description if available
        if self.description:
            prompt_parts.append(f"Description: {self.description}")

        # Add instructions if available
        if self.instructions:
            instructions = "\n".join(self.instructions)
            prompt_parts.append(f"Instructions: {instructions}")

        # Add context if available
        if context:
            prompt_parts.append(f"Context: {context}")

        # Add the user's message
        prompt_parts.append(f"User Input: {message}")

        return "\n\n".join(prompt_parts)

    def _format_response_as_json(self, response: str) -> Union[Dict, str]:
        """Format the response as JSON if json_output is True."""
        try:
            # Use regex to extract JSON from the response (e.g., within ```json ``` blocks)
            json_match = re.search(r'```json\s*({.*?})\s*```', response, re.DOTALL)
            if json_match:
                # Extract the JSON part and parse it
                json_str = json_match.group(1)
                return json.loads(json_str)  # Return the parsed JSON object (a dictionary)
            else:
                # If no JSON block is found, try to parse the entire response as JSON
                return json.loads(response)  # Return the parsed JSON object (a dictionary)
        except json.JSONDecodeError:
            # If the response is not valid JSON, wrap it in a dictionary
            return {"response": response}  # Return a dictionary with the response as a string

    def normalize_key(self, key: str) -> str:
        """
        Normalize a key by converting it to lowercase and replacing spaces with underscores.
        """
        return key.lower().replace(" ", "_")
    
    def match_key(self, expected_key, response_keys, threshold=0.5):
        """
        Match an expected key to the closest key in the response using semantic similarity or fuzzy matching.
        """
        expected_key_norm = self.normalize_key(expected_key)
        response_keys_norm = [self.normalize_key(k) for k in response_keys]

        if hasattr(self, 'semantic_model') and self.semantic_model is not None:
            try:
                # Compute embeddings for the expected key and all response keys
                expected_embedding = self.semantic_model.encode(expected_key_norm, convert_to_tensor=True)
                response_embeddings = self.semantic_model.encode(response_keys_norm, convert_to_tensor=True)
                
                # Compute cosine similarity
                similarity_scores = util.pytorch_cos_sim(expected_embedding, response_embeddings)[0]
                
                # Find the best match
                best_score = similarity_scores.max().item()
                best_index = similarity_scores.argmax().item()
                
                if best_score > threshold:
                    return response_keys[best_index], best_score
            except Exception as e:
                logging.warning(f"Semantic matching failed: {e}. Falling back to fuzzy matching.")
        
        # Fallback to fuzzy matching
        best_match = None
        best_score = -1
        for key, key_norm in zip(response_keys, response_keys_norm):
            score = fuzz.ratio(expected_key_norm, key_norm) / 100
            if score > best_score:
                best_score = score
                best_match = key
        
        return best_match, best_score

    def _validate_response(self, response: Union[str, Dict]) -> Union[str, Dict]:
        """Validate the response against the expected_output format using semantic similarity or fallback methods."""
        if isinstance(self.expected_output, dict):
            if not isinstance(response, dict):
                return {"response": response}

            validated_response = {}
            normalized_expected_keys = {self.normalize_key(k): k for k in self.expected_output.keys()}

            for expected_key_norm, expected_key_orig in normalized_expected_keys.items():
                # Find all response keys that match the expected key (case-insensitive and normalized)
                matching_response_keys = [
                    k for k in response.keys()
                    if self.normalize_key(k) == expected_key_norm
                ]

                # If no exact match, use semantic matching to find similar keys
                if not matching_response_keys:
                    for response_key in response.keys():
                        best_match, best_score = self.match_key(expected_key_orig, [response_key])
                        if best_match and best_score > 0.5:  # Use a threshold to determine a valid match
                            matching_response_keys.append(response_key)

                # Merge values from all matching keys
                merged_values = []
                for matching_key in matching_response_keys:
                    value = response[matching_key]
                    if isinstance(value, list):
                        merged_values.extend(value)
                    else:
                        merged_values.append(value)

                # Assign the merged values to the expected key
                if merged_values:
                    validated_response[expected_key_orig] = merged_values
                else:
                    validated_response[expected_key_orig] = "NA"  # Default value for missing keys

                # Recursively validate nested dictionaries
                expected_value = self.expected_output[expected_key_orig]
                if isinstance(expected_value, dict) and isinstance(validated_response[expected_key_orig], dict):
                    validated_response[expected_key_orig] = self._validate_response(validated_response[expected_key_orig])

            return validated_response
        elif isinstance(self.expected_output, str):
            if not isinstance(response, str):
                return str(response)
        return response
    
    def cli_app(
        self,
        message: Optional[str] = None,
        exit_on: Optional[List[str]] = None,
        **kwargs,
    ):
        """Run the agent in a CLI app."""
        from rich.prompt import Prompt

        if message:
            self.print_response(message=message, **kwargs)

        _exit_on = exit_on or ["exit", "quit", "bye"]
        while True:
            message = Prompt.ask(f"[bold] {self.emoji} {self.user_name} [/bold]")
            if message in _exit_on:
                break

            self.print_response(message=message, **kwargs)

    def _generate_api(self):
        """Generate an API for the agent if api=True."""
        from .api.api_generator import APIGenerator
        self.api_generator = APIGenerator(self)
        print(f"API generated for agent '{self.name}'. Use `.run_api()` to start the API server.")

    def run_api(self):
        """Run the API server for the agent."""
        if not hasattr(self, 'api_generator'):
            raise ValueError("API is not enabled for this agent. Set `api=True` when initializing the agent.")
    
        # Get API configuration
        host = self.api_config.get("host", "0.0.0.0") if self.api_config else "0.0.0.0"
        port = self.api_config.get("port", 8000) if self.api_config else 8000

        # Run the API server
        self.api_generator.run(host=host, port=port)

    def _flatten_data(self, data: Union[Dict, List], parent_key: str = "", separator: str = "_") -> List[Dict]:
        """
        Recursively flatten a nested dictionary or list into a list of key-value pairs.

        Args:
            data (Union[Dict, List]): The nested data structure.
            parent_key (str): The parent key (used for recursion).
            separator (str): The separator used for nested keys.

        Returns:
            List[Dict]: A list of flattened key-value pairs.
        """
        items = []
        if isinstance(data, dict):
            for key, value in data.items():
                new_key = f"{parent_key}{separator}{key}" if parent_key else key
                if isinstance(value, (dict, list)):
                    items.extend(self._flatten_data(value, new_key, separator))
                else:
                    items.append({new_key: value})
                    # Include the value as a key for searching
                    if isinstance(value, str):
                        items.append({value: new_key})
        elif isinstance(data, list):
            for index, item in enumerate(data):
                new_key = f"{parent_key}{separator}{index}" if parent_key else str(index)
                if isinstance(item, (dict, list)):
                    items.extend(self._flatten_data(item, new_key, separator))
                else:
                    items.append({new_key: item})
                    # Include the value as a key for searching
                    if isinstance(item, str):
                        items.append({item: new_key})
        return items
    
    def _find_all_relevant_keys(self, query: str, flattened_data: List[Dict], threshold: float = 0.5) -> List[str]:
        """
        Find all relevant keys in the flattened data based on semantic similarity to the query.

        Args:
            query (str): The user's query.
            flattened_data (List[Dict]): The flattened key-value pairs.
            threshold (float): The similarity threshold for considering a match.

        Returns:
            List[str]: A list of relevant values.
        """
        if not flattened_data:
            return []

        # Extract keys from the flattened data
        keys = [list(item.keys())[0] for item in flattened_data]

        # Compute embeddings for the query and keys
        query_embedding = self.semantic_model.encode(query, convert_to_tensor=True)
        key_embeddings = self.semantic_model.encode(keys, convert_to_tensor=True)

        # Compute cosine similarity between the query and keys
        similarities = util.pytorch_cos_sim(query_embedding, key_embeddings)[0]

        # Find all keys with a similarity score above the threshold
        relevant_indices = [i for i, score in enumerate(similarities) if score > threshold]
        relevant_values = [flattened_data[i][keys[i]] for i in relevant_indices]

        return relevant_values