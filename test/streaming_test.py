
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY")
)

response = client.chat.completions.create(
    model=os.getenv("MODEL"),
    messages = [{"role": "user", "content": "What is an API explain me in 100 words?"}],
    stream = True,
)

full_response = ""
for chunk in response:
    if not chunk.choices:
        continue
    piece = chunk.choices[0].delta.content
    if piece:
        print(piece, end = "",flush=True)
        full_response += piece

print("\n\nFull response:", full_response)