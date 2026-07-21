import os
from dotenv import load_dotenv

# .env file se keys ko load karne ke liye
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    print("❌ Error: API Key nahi mili! Apni .env file check karein.")
else:
    # Key ka sirf shuruat ka hissa print karenge security ke liye
    print(f"✅ Success: API Key sahi se load ho gayi hai! ({GOOGLE_API_KEY[:6]}...)")
