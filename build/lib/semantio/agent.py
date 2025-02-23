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
from .memory import Memory

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Agent(BaseModel):
    """
    An intelligent agent that combines LLM capabilities with dynamic knowledge base integration,
    tool usage, and conversation memory. The agent can ingest external domain-specific content (via a dynamic document loader)
    so that it answers queries based on that information.
    """
    name: Optional[str] = Field(None, description="Name of the agent.")
    description: Optional[str] = Field(None, description="Description of the agent's role.")
    instructions: Optional[List[str]] = Field(None, description="List of instructions for the agent.")
    model: Optional[str] = Field(None, description="This one is not in use.")
    show_tool_calls: bool = Field(False, description="Whether to show tool calls in the response.")
    markdown: bool = Field(False, description="Whether to format the response in markdown.")
    tools: Optional[List[BaseTool]] = Field(None, description="List of tools available to the agent.")
    user_name: Optional[str] = Field("User", description="Name of the user interacting with the agent.")
    emoji: Optional[str] = Field(":robot:", description="Emoji to represent the agent in the CLI.")
    rag: Optional[RAG] = Field(None, description="RAG instance for context retrieval.")
    knowledge_base: Optional[Any] = Field(
        None,
        description="Domain-specific knowledge base content (e.g., loaded via a dynamic document loader)."
    )
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
    memory: Memory = Field(default_factory=Memory)
    memory_config: Dict = Field(
        default_factory=lambda: {
            "max_context_length": 4000,
            "summarization_threshold": 3000
        }
    )
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize the LLM model and tools if needed.
        self._initialize_model()
        # Initialize conversation memory with configuration.
        self.memory = Memory(
            max_context_length=self.memory_config.get("max_context_length", 4000),
            summarization_threshold=self.memory_config.get("summarization_threshold", 3000)
        )
        # Initialize tools as an empty list if not provided.
        if self.tools is None:
            self.tools = []
        # Automatically discover and register tools if auto_tool is enabled.
        if self.auto_tool and not self.tools:
            self.tools = self._discover_tools()
        # Pass the LLM instance to each tool.
        for tool in self.tools:
            tool.llm = self.llm_instance
        # Initialize the SentenceTransformer model for semantic matching.
        self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        # Initialize default RAG if not provided.
        if self.rag is None:
            self.rag = self._initialize_default_rag()
        # Automatically generate API if api=True.
        if self.api:
            self._generate_api()

    def _initialize_model(self):
        """Initialize the LLM model based on the provided configuration."""
        if self.llm_instance is not None:
            return  # Already initialized.
        if self.llm is None:
            raise ValueError("llm must be specified.")
    
        # Retrieve API key from configuration or environment variable.
        api_key = getattr(self, 'api_key', None) or os.getenv(f"{self.llm.upper()}_API_KEY")
    
        # Map LLM providers to their respective classes and default models.
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

        llm_provider = self.llm.lower()
        if llm_provider not in llm_providers:
            raise ValueError(f"Unsupported LLM provider: {self.llm}. Supported providers: {list(llm_providers.keys())}")

        llm_config = llm_providers[llm_provider]
        llm_class_name = llm_config["class"]
        default_model = llm_config["default_model"]
        model_to_use = self.llm_model or default_model

        # Dynamically import and initialize the LLM class.
        module_name = f"semantio.llm.{llm_provider}"
        llm_module = importlib.import_module(module_name)
        llm_class = getattr(llm_module, llm_class_name)
        self.llm_instance = llm_class(model=model_to_use, api_key=api_key)

    def _initialize_default_rag(self) -> RAG:
        """Initialize a default RAG instance using a dummy vector store."""
        vector_store = VectorStore()
        retriever = Retriever(vector_store)
        return RAG(retriever)

    def print_response(
        self,
        message: Optional[Union[str, Image, List, Dict]] = None,
        image: Optional[Union[str, Image]] = None,
        stream: bool = False,
        markdown: bool = False,
        team: Optional[List['Agent']] = None,
        **kwargs,
    ) -> Union[str, Dict]:
        """
        Generate and print the agent's response while storing conversation history.
        If an image is provided (either via the 'image' parameter or if 'message' is a PIL.Image),
        the agent processes it accordingly.
        If a team is provided (or if self.team is set), only the aggregated final response is returned.
        """
        # Handle image input first.
        if image is not None:
            response = self._generate_response_from_image(message or "", image, markdown=markdown, **kwargs)
            print(response)
            if response:
                self.memory.add_message(role="agent", content=response)
            return response

        if isinstance(message, Image):
            response = self._generate_response_from_image("", message, markdown=markdown, **kwargs)
            print(response)
            if response:
                self.memory.add_message(role="agent", content=response)
            return response

        # For text input, add the user message to memory.
        if message and isinstance(message, str):
            self.memory.add_message(role="user", content=message)

        # If a team is provided (or if self.team exists), generate an aggregated final response.
        if team is None and self.team is not None:
            team = self.team

        if team is not None:
            # Instead of printing individual team outputs, call each agent's _generate_response
            # to capture their outputs silently.
            aggregated_responses = []
            for agent in team:
                resp = agent._generate_response(message, markdown=markdown, **kwargs)
                aggregated_responses.append(f"**{agent.name}:**\n\n{resp}")
            final_response = "\n\n".join(aggregated_responses)
            print(final_response)
            self.memory.add_message(role="agent", content=final_response)
            return final_response

        # Standard text response processing.
        if stream:
            response = ""
            for chunk in self._stream_response(message, markdown=markdown, **kwargs):
                print(chunk, end="", flush=True)
                response += chunk
            if response:
                self.memory.add_message(role="agent", content=response)
            print()
            return response
        else:
            response = self._generate_response(message, markdown=markdown, **kwargs)
            print(response)
            if response:
                self.memory.add_message(role="agent", content=response)
            return response

    def _stream_response(self, message: str, markdown: bool = False, **kwargs) -> Iterator[str]:
        """Simulate streaming of the agent's response."""
        response = self._generate_response(message, markdown=markdown, **kwargs)
        for chunk in response.split():
            yield chunk + " "

    def _generate_response_from_image(self, message: str, image: Union[str, Image], markdown: bool = False, **kwargs) -> str:
        """
        Process an image by sending it to the LLM for analysis if the LLM supports vision.
        Supports both image URLs and local PIL.Image objects.
        """
        try:
            if not self.llm_instance or not getattr(self.llm_instance, "supports_vision", False):
                raise ValueError("Vision is not supported for the current model.")
            prompt = self._build_prompt(message, context=None)
            if isinstance(image, str) and image.startswith("http"):
                return self.llm_instance.generate_from_image_url(prompt, image, **kwargs)
            elif isinstance(image, Image):
                if image.mode == "RGBA":
                    image = image.convert("RGB")
                image_bytes = io.BytesIO()
                image.save(image_bytes, format="JPEG")
                image_bytes = image_bytes.getvalue()
                return self.llm_instance.generate_from_image(prompt, image_bytes, **kwargs)
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
        for file in tools_dir.glob("*.py"):
            if file.name == "base_tool.py":
                continue  # Skip the base tool file.
            try:
                module_name = file.stem
                module = importlib.import_module(f"semantio.tools.{module_name}")
                for name, obj in module.__dict__.items():
                    if isinstance(obj, type) and issubclass(obj, BaseTool) and obj != BaseTool:
                        tools.append(obj())
                        logger.info(f"Registered tool: {obj.__name__}")
            except Exception as e:
                logger.error(f"Failed to load tool from {file}: {e}")
        return tools

    def _get_tool_descriptions(self) -> str:
        """
        Generate a description of all available tools for inclusion in the LLM prompt.
        """
        return "\n".join(f"{tool.name}: {tool.description}" for tool in self.tools)

    def register_tool(self, tool: BaseTool):
        """Register a tool for the agent."""
        if self.tools is None:
            self.tools = []
        self.tools.append(tool)
    
    def _analyze_query_and_select_tools(self, query: str) -> List[Dict[str, Any]]:
        """
        Use the LLM to analyze the query and dynamically select the most appropriate tools.
        Returns a list of tool calls (tool name and input).
        """
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
            response = self.llm_instance.generate(prompt=prompt)
            tool_calls = json.loads(response)
            return tool_calls
        except Exception as e:
            logger.error(f"Failed to analyze query and select tools: {e}")
            return []
    
    def _generate_response(self, message: str, markdown: bool = False, team: Optional[List['Agent']] = None, **kwargs) -> str:
        """Generate the agent's response, including tool execution and context retrieval."""
        if team is not None:
            return self._generate_team_response(message, team, markdown=markdown, **kwargs)
        
        tool_outputs = {}
        responses = []
        tool_calls = []

        if self.auto_tool:
            tool_calls = self._analyze_query_and_select_tools(message)
        else:
            if self.tools:
                tool_calls = [
                    {
                        "tool": tool.name,
                        "input": {"query": message, "context": None}
                    }
                    for tool in self.tools
                ]

        if tool_calls:
            for tool_call in tool_calls:
                tool_name = tool_call["tool"]
                tool_input = tool_call["input"]
                tool = next((t for t in self.tools if t.name.lower() == tool_name.lower()), None)
                if tool:
                    try:
                        tool_output = tool.execute(tool_input)
                        response_text = f"Tool '{tool_name}' executed. Output: {tool_output}"
                        if self.show_tool_calls:
                            response_text = f"**Tool Called:** {tool_name}\n\n{response_text}"
                        responses.append(response_text)
                        tool_outputs[tool_name] = tool_output
                    except Exception as e:
                        logger.error(f"Error executing tool '{tool_name}': {e}")
                        responses.append(f"An error occurred while executing the tool '{tool_name}': {e}")
                else:
                    responses.append(f"Tool '{tool_name}' not found.")

        if tool_outputs:
            try:
                context = {
                    "conversation_history": self.memory.get_context(self.llm_instance),
                    "tool_outputs": tool_outputs,
                    "rag_context": self.rag.retrieve(message) if self.rag else None,
                    "knowledge_base": self._get_knowledge_context(message) if self.knowledge_base else None,
                }
                prompt = self._build_memory_prompt(message, context)
                memory_entries = [{"role": e.role, "content": e.content} for e in self.memory.storage.retrieve()]
                llm_response = self.llm_instance.generate(prompt=prompt, context=context, memory=memory_entries, **kwargs)
                responses.append(f"**Analysis:**\n\n{llm_response}")
            except Exception as e:
                logger.error(f"Failed to generate LLM response: {e}")
                responses.append(f"An error occurred while generating the analysis: {e}")
        elif not self.tools and not tool_calls:
            context = {
                "conversation_history": self.memory.get_context(self.llm_instance),
                "rag_context": self.rag.retrieve(message) if self.rag else None,
                "knowledge_base": self._get_knowledge_context(message),
            }
            prompt = self._build_memory_prompt(message, context)
            memory_entries = [{"role": e.role, "content": e.content} for e in self.memory.storage.retrieve()]
            response = self.llm_instance.generate(prompt=prompt, context=context, memory=memory_entries, **kwargs)
            if self.json_output:
                response = self._format_response_as_json(response)
            if self.expected_output:
                response = self._validate_response(response)
            if markdown:
                return f"**Response:**\n\n{response}"
            return response
        return "\n\n".join(responses)
    
    def _generate_team_response(self, message: str, team: List['Agent'], markdown: bool = False, **kwargs) -> str:
        """
        Generate a final aggregated response using a team of assistants.
        This method calls each team member's internal _generate_response (without printing)
        and aggregates the results into a single output.
        """
        team_responses = []
        for agent in team:
            resp = agent._generate_response(message, markdown=markdown, **kwargs)
            team_responses.append(f"**{agent.name}:**\n\n{resp}")
        return "\n\n".join(team_responses)

    def _build_memory_prompt(self, user_input: str, context: dict) -> str:
        """Construct a prompt that incorporates role, instructions, conversation history, and external context."""
        prompt_parts = []
        if self.description:
            prompt_parts.append(f"# ROLE\n{self.description}")
        if self.instructions:
            prompt_parts.append("# INSTRUCTIONS\n" + "\n".join(f"- {i}" for i in self.instructions))
        if context.get('conversation_history'):
            prompt_parts.append(f"# CONVERSATION HISTORY\n{context['conversation_history']}")
        if context.get('knowledge_base'):
            prompt_parts.append(f"# KNOWLEDGE BASE\n{context['knowledge_base']}")
        prompt_parts.append(f"# USER INPUT\n{user_input}")
        return "\n\n".join(prompt_parts)
        
    def _summarize_text(self, text: str) -> str:
        """
        Summarize the provided text using the LLM.
        Adjust the prompt as needed.
        """
        prompt = f"Summarize the following text concisely:\n\n{text}\n\nSummary:"
        summary = self.llm_instance.generate(prompt=prompt)
        return summary.strip()

    def _get_knowledge_context(self, message: str) -> str:
        """
        Retrieve context from the knowledge base.
        For JSON documents, use the "flattened" field.
        For other documents (e.g., website, YouTube) use the "text" field.
        If the combined text is too long, break it into chunks and summarize each chunk.
        """
        if not self.knowledge_base:
            return ""
        texts = []
        for doc in self.knowledge_base:
            if isinstance(doc, dict):
                if "flattened" in doc:
                    # Join all values from the flattened key/value pairs.
                    flattened_text = " ".join(str(v) for item in doc["flattened"] for v in item.values())
                    texts.append(flattened_text)
                elif "text" in doc:
                    texts.append(doc["text"])
                else:
                    texts.append(" ".join(str(v) for v in doc.values()))
            else:
                texts.append(str(doc))
        combined_text = "\n".join(texts)
        
        # If the combined text is very long, break it into chunks and summarize.
        max_words = 1000
        words = combined_text.split()
        if len(words) > max_words:
            chunks = []
            for i in range(0, len(words), max_words):
                chunk = " ".join(words[i:i+max_words])
                chunks.append(chunk)
            # Summarize each chunk.
            summaries = [self._summarize_text(chunk) for chunk in chunks]
            final_context = "\n".join(summaries)
            return final_context
        else:
            return combined_text




    
    def _build_prompt(self, message: str, context: Optional[List[Dict]]) -> str:
        """Build a basic prompt including description, instructions, context, and user input."""
        prompt_parts = []
        if self.description:
            prompt_parts.append(f"Description: {self.description}")
        if self.instructions:
            prompt_parts.append("Instructions: " + "\n".join(self.instructions))
        if context:
            prompt_parts.append(f"Context: {context}")
        prompt_parts.append(f"User Input: {message}")
        return "\n\n".join(prompt_parts)

    def _format_response_as_json(self, response: str) -> Union[Dict, str]:
        """Attempt to extract and format a JSON response."""
        try:
            json_match = re.search(r'```json\s*({.*?})\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                return json.loads(json_str)
            else:
                return json.loads(response)
        except json.JSONDecodeError:
            return {"response": response}

    def normalize_key(self, key: str) -> str:
        """Normalize a key by converting to lowercase and replacing spaces with underscores."""
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
        """
        Validate and structure the response based on the expected_output using semantic matching.
        """
        if isinstance(self.expected_output, dict):
            if not isinstance(response, dict):
                return {"response": response}
            validated_response = {}
            normalized_expected_keys = {self.normalize_key(k): k for k in self.expected_output.keys()}
            for expected_key_norm, expected_key_orig in normalized_expected_keys.items():
                matching_response_keys = [k for k in response.keys() if self.normalize_key(k) == expected_key_norm]
                if not matching_response_keys:
                    for response_key in response.keys():
                        best_match, best_score = self.match_key(expected_key_orig, [response_key])
                        if best_match and best_score > 0.5:
                            matching_response_keys.append(response_key)
                merged_values = []
                for matching_key in matching_response_keys:
                    value = response[matching_key]
                    if isinstance(value, list):
                        merged_values.extend(value)
                    else:
                        merged_values.append(value)
                validated_response[expected_key_orig] = merged_values if merged_values else "NA"
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
        """Run the agent as a command-line application."""
        from rich.prompt import Prompt

        if message:
            self.print_response(message=message, **kwargs)

        _exit_on = exit_on or ["exit", "quit", "bye"]
        while True:
            try:
                user_input = Prompt.ask(f"[bold] {self.emoji} {self.user_name} [/bold]")
                if user_input in _exit_on:
                    break
                self.print_response(message=user_input, **kwargs)
            except KeyboardInterrupt:
                print("\n\nSession ended. Goodbye!")
                break

    def _generate_api(self):
        """Generate an API for the agent if API mode is enabled."""
        from .api.api_generator import APIGenerator
        self.api_generator = APIGenerator(self)
        print(f"API generated for agent '{self.name}'. Use `.run_api()` to start the API server.")

    def run_api(self):
        """Run the API server for the agent."""
        if not hasattr(self, 'api_generator'):
            raise ValueError("API is not enabled for this agent. Set `api=True` when initializing the agent.")
        host = self.api_config.get("host", "0.0.0.0") if self.api_config else "0.0.0.0"
        port = self.api_config.get("port", 8000) if self.api_config else 8000
        self.api_generator.run(host=host, port=port)

