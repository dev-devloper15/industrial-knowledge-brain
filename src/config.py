import os
from dotenv import load_dotenv

# for load keys from .env file
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    print("❌ Error: API key not found. please check .env file")
else:
    print(f"✅ Success: API Key load successfully! ({GROQ_API_KEY[:6]}...)")
