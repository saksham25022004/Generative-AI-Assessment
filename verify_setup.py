import os
from dotenv import load_dotenv
import openai

load_dotenv()
print("Verifying environment...")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("❌ OPENAI_API_KEY not set in environment (.env).")
else:
    try:
        openai.api_key = OPENAI_API_KEY
        resp = openai.Embedding.create(model="text-embedding-3-small", input="test")
        print("✅ OpenAI embeddings OK. Model returned vector of length", len(resp['data'][0]['embedding']))
    except Exception as e:
        print("❌ OpenAI call failed:", str(e))