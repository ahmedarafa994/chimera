import os

from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("CHIMERA_API_KEY")
print(f"CHIMERA_API_KEY: {api_key}")
