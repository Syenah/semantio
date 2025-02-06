from setuptools import setup, find_packages

setup(
    name="semantio",
    version="0.0.4",
    description="A powerful SDK for building AI agents",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Rakesh",
    author_email="rakeshsahoo689@gmail.com",
    url="https://github.com/Syenah/semantio",
    packages=find_packages(),
    install_requires=[
        "openai",
        "anthropic",
        "groq",
        "google-genai",
        "mistralai",
        "faiss-cpu",  # For vector storage
        "pydantic",   # For data validation
        "requests",   # For web tools
        "playwright", # For web scraping
        "fastapi",    #	For creating the RESTful API
        "uvicorn",    # For running the FastAPI app
        "pillow",     # For image processing
        "slowapi",    # For rate limiting
        "sentence-transformers", # For sentence embeddings
        "fuzzywuzzy", # For fuzzy string matching
        "duckduckgo-search", # For DuckDuckGo search
        "yfinance",   # For stock/crypto prices
        "selenium",   # For web scraping
        "beautifulsoup4", # For HTML parsing
        "webdriver-manager", # For browser automation
        "validators", # For URL validation

    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "semantio=semantio.cli.main:main",
        ],
    },
)