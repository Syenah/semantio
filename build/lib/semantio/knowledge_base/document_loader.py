import os
import json
import csv
import re
from pathlib import Path
from typing import List, Dict, Any
from io import BytesIO

import requests
from bs4 import BeautifulSoup

# Optional: Import pandas for XLSX support and PyPDF2 for PDF support
try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from PyPDF2 import PdfReader
except ImportError:
    PdfReader = None


def flatten_json(data: Any, parent_key: str = "", separator: str = "_") -> List[Dict[str, Any]]:
    """
    Recursively flatten a JSON structure.
    For each key-value pair, add an entry mapping key->value.
    Additionally, if the value is a string, add an entry mapping the value to its flattened key.
    """
    items = []
    if isinstance(data, dict):
        for key, value in data.items():
            new_key = f"{parent_key}{separator}{key}" if parent_key else key
            if isinstance(value, (dict, list)):
                items.extend(flatten_json(value, new_key, separator))
            else:
                items.append({new_key: value})
                if isinstance(value, str):
                    items.append({value: new_key})
    elif isinstance(data, list):
        for index, item in enumerate(data):
            new_key = f"{parent_key}{separator}{index}" if parent_key else str(index)
            if isinstance(item, (dict, list)):
                items.extend(flatten_json(item, new_key, separator))
            else:
                items.append({new_key: item})
                if isinstance(item, str):
                    items.append({item: new_key})
    return items


class DocumentLoader:
    """
    A dynamic document loader that supports multiple source types:
    
    - Local files: CSV, TXT, JSON, XLSX, PDF
    - URL sources: HTML websites (text extraction), JSON APIs, PDF URLs
    - YouTube links: Extracts transcripts using youtube_transcript_api
    
    For JSON sources, if flatten is True (default), the returned document is a dictionary with two keys:
       "original": the raw JSON data,
       "flattened": a list of flattened key/value pairs (including reverse mappings).
    """
    def load(self, source: str, flatten: bool = True) -> List[Dict[str, Any]]:
        """
        Load documents from the given source.
        If source starts with "http", treat it as a URL; otherwise, as a local file.
        """
        if source.startswith("http"):
            return self.load_from_url(source, flatten=flatten)
        else:
            return self.load_from_file(source, flatten=flatten)

    def load_from_file(self, file_path: str, flatten: bool = True) -> List[Dict[str, Any]]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        ext = path.suffix.lower()
        if ext == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if flatten:
                return [{"original": data, "flattened": flatten_json(data)}]
            else:
                return data if isinstance(data, list) else [data]
        elif ext == ".txt":
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            return [{"text": content}]
        elif ext == ".csv":
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                return [row for row in reader]
        elif ext == ".xlsx":
            if pd is None:
                raise ImportError("pandas is required to load XLSX files")
            df = pd.read_excel(path)
            return df.to_dict(orient="records")
        elif ext == ".pdf":
            if PdfReader is None:
                raise ImportError("PyPDF2 is required to load PDF files")
            reader = PdfReader(str(path))
            content = ""
            for page in reader.pages:
                content += page.extract_text() or ""
            return [{"text": content}]
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    def load_from_url(self, url: str, flatten: bool = True) -> List[Dict[str, Any]]:
        if "youtube.com" in url or "youtu.be" in url:
            return self._load_youtube(url)
        response = requests.get(url)
        if response.status_code != 200:
            raise ValueError(f"Failed to fetch data from URL: {url}")
        content_type = response.headers.get("Content-Type", "").lower()
        if "application/json" in content_type:
            data = response.json()
            if flatten:
                return [{"original": data, "flattened": flatten_json(data)}]
            else:
                return data if isinstance(data, list) else [data]
        elif "text/html" in content_type:
            # First, try with requests + BeautifulSoup.
            soup = BeautifulSoup(response.text, "html.parser")
            text = soup.get_text(separator="\n").strip()
            # If the text seems too short (less than 50 words), assume content is loaded via JavaScript.
            if len(text.split()) < 50:
                try:
                    text = self._fetch_with_headless_browser(url)
                except Exception as e:
                    # If headless browser fails, log and fallback to the short text.
                    print(f"Headless fetch failed: {e}")
            return [{"text": text}]
        elif "application/pdf" in content_type:
            if PdfReader is None:
                raise ImportError("PyPDF2 is required to load PDF files")
            pdf_file = BytesIO(response.content)
            reader = PdfReader(pdf_file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            return [{"text": text}]
        else:
            return [{"text": response.text}]

    def _fetch_with_headless_browser(self, url: str) -> str:
        """
        Use a headless browser (Playwright) to fetch fully rendered content.
        """
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            raise ImportError("playwright is required for JS-rendered pages. Install it with 'pip install playwright' and run 'playwright install'.")
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, wait_until="networkidle")
            html = page.content()
            browser.close()
            soup = BeautifulSoup(html, "html.parser")
            text = soup.get_text(separator="\n").strip()
            return text

    def _load_youtube(self, url: str) -> List[Dict[str, Any]]:
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
        except ImportError:
            raise ImportError("youtube_transcript_api is required to load YouTube transcripts")
        
        video_id = None
        patterns = [r"v=([^&]+)", r"youtu\.be/([^?&]+)"]
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                video_id = match.group(1)
                break
        if not video_id:
            raise ValueError("Could not extract video ID from URL")
        
        # Define a prioritized list of language codes to try
        preferred_languages = ["en", "hi", "es", "fr", "de", "ru"]
        
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=preferred_languages)
            text = " ".join(segment["text"] for segment in transcript)
            return [{"text": text}]
        except Exception as e:
            # Return a fallback document indicating transcript retrieval failed
            return [{"text": f"Transcript not available for video {url}: {str(e)}"}]

