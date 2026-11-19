# standard library imports
import os
# third-party imports
from openai import OpenAI
from dotenv import load_dotenv
# local imports 
from Utilities.utilities import log
from tools import TOOLS, run_tool

load_dotenv()  # By default, it looks for a .env file in the current directory.

client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY")
)

messages = []

MAX_STEPS = 5 # maximum steps for the agent for tool calling to avoid infinite loops 
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
    for step in range(MAX_STEPS): 
        response = client.chat.completions.create(
                model=os.getenv("MODEL"),
                messages=messages,
                tools=TOOLS
            )
        msg = response.choices[0].message

        if not msg.tool_calls:
            break

        log(f"Step {step+1}: model requested {len(msg.tool_calls)} tool calls.")

        messages.append(msg)
             
        for tool_call in msg.tool_calls:
            result = run_tool(tool_call)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(result)
            })
    else:
    # using else after for loops is used to check if the loop was exited normally and not using break
        fallback = "Sorry, I couldn't finish this within step limit."
        messages.append({
        "role":"assistant",
        "content": fallback
        })
        print("Assistant:"+ fallback)
        continue

    messages.append({
        "role": "assistant",
        "content": msg.content
    })

    print("Assistant: " + (msg.content or ""))
