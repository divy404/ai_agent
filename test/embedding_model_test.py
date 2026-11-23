import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY")
)

response = client.embeddings.create(
    model="nomic-embed-text",
    input="I am Divyansh Jain"
)

print(len(response.data[0].embedding))