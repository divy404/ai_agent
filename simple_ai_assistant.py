
import json

from openai import OpenAI
from dotenv import load_dotenv
import os

from tools import TOOLS, FUNCTION_NAME_TO_FUNCTION_MAP

load_dotenv()  # By default, it looks for a .env file in the current directory.

client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY")
)

messages = []

messages.append({
    "role": "system",
    "content": "Always respond in a fun way and keep the response within 27 words."
})

while True:
    user_input = input("\nUser: ")

    if user_input.lower() in ["exit", "quit", "bye", "goodbye"]:
        print("Exiting the assistant. Byeee!")
        break

    messages.append({
        "role": "user",
        "content": user_input
    })
# chained tool calling 
    while True:
        response = client.chat.completions.create(
                model=os.getenv("MODEL"),
                messages=messages,
                tools=TOOLS
            )
        msg = response.choices[0].message

        if not msg.tool_calls:
            break

        messages.append(msg)
             
        for tool_call in msg.tool_calls:
            func_name = tool_call.function.name
            func_args = json.loads(tool_call.function.arguments or "{}")

            fn = FUNCTION_NAME_TO_FUNCTION_MAP[func_name]
            result = fn(**func_args)  # Unpack dictionary into function arguments

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(result)
            })
            
    messages.append({
        "role": "assistant",
        "content": msg.content
    })

    print("Assistant: " + (msg.content or ""))
