import logging
import os
import sys

from google import genai

api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not api_key:
    try:
        from dotenv import load_dotenv

        load_dotenv(os.path.join(os.getcwd(), "backend-api", ".env"))
        api_key = os.getenv("GOOGLE_API_KEY")
    except Exception as exc:
        logging.debug("Could not load .env: %s", exc)

if not api_key:
    print("No API Key found")
    sys.exit(1)

client = genai.Client(api_key=api_key)
print("Listing models...")
try:
    for m in client.models.list(config={"page_size": 100}):
        print(f"Model: {m.name}")
except Exception as e:
    print(f"Error listing models: {e}")
