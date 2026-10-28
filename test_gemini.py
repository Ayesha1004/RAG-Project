import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY")
)

response = client.models.generate_content(
    model=os.getenv(
        "GEMINI_EVAL_MODEL",
        "gemini-2.5-flash"
    ),
    contents="Reply with: Gemini API working"
)

print(response.text)