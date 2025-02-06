from .openai import OpenAILlm
from .anthropic import AnthropicLLM
from .groq import GroqLlm
from .mistral import MistralLLM
from .deepseek import DeepSeekLLM
from .gemini import GeminiLLM

def get_llm(provider: str, **kwargs):
    provider = provider.lower()  # Convert provider name to lowercase
    if provider == "openai":
        return OpenAILlm(**kwargs)
    elif provider == "anthropic":
        return AnthropicLLM(**kwargs)
    elif provider == "groq":
        return GroqLlm(**kwargs)
    elif provider == "mistral":
        return MistralLLM(**kwargs)
    elif provider == "deepseek":
        return DeepSeekLLM(**kwargs)
    elif provider == "gemini":
        return GeminiLLM(**kwargs)
    
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")