# list_models.py -- Show currently available models from each provider.
# Usage: uv run python list_models.py

import os
from google import genai
from openai import OpenAI

def list_gemini(api_key):
    print("\n=== GEMINI ===")
    try:
        client = genai.Client(api_key=api_key)
        models = client.models.list()
        for m in models:
            name = getattr(m, 'name', str(m))
            print("  " + name)
    except Exception as e:
        print("  Error: " + str(e))

def list_openai_compat(label, api_key, base_url):
    print("\n=== " + label.upper() + " ===")
    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        models = client.models.list()
        for m in models.data:
            print("  " + m.id)
    except Exception as e:
        print("  Error: " + str(e))

if __name__ == "__main__":
    gemini_key = os.environ.get("GEMINI_API_KEY")
    groq_key = os.environ.get("GROQ_API_KEY")
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")

    if gemini_key:
        list_gemini(gemini_key)
    if groq_key:
        list_openai_compat("Groq", groq_key, "https://api.groq.com/openai/v1")
    if openrouter_key:
        list_openai_compat("OpenRouter (free only)", openrouter_key, "https://openrouter.ai/api/v1/models?supported_parameters=free")
