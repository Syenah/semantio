import argparse
import warnings
from semantio.agent import Agent
from semantio.llm import get_llm
from urllib3.exceptions import NotOpenSSLWarning

# Suppress the NotOpenSSLWarning
warnings.filterwarnings("ignore", category=NotOpenSSLWarning)

def main():
    parser = argparse.ArgumentParser(description="opAi CLI")
    parser.add_argument("--message", type=str, required=True, help="Message to send to the agent")
    parser.add_argument("--provider", type=str, required=True, help="LLM provider (e.g., groq, openai)")
    parser.add_argument("--api-key", type=str, required=True, help="API key for the LLM provider")
    parser.add_argument("--model", type=str, default=None, help="Model name (e.g., mixtral-8x7b-32768)")
    args = parser.parse_args()

    # Initialize LLM
    llm_config = {"api_key": args.api_key}
    if args.model:
        llm_config["model"] = args.model

    llm = get_llm(provider=args.provider, **llm_config)

    # Create an agent
    agent = Agent(model=args.provider, llm=llm)
    agent.print_response(args.message)
    

if __name__ == "__main__":
    main()