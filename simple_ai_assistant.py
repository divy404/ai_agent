
# standard library imports
import os
from types import SimpleNamespace

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
def stream_chat(history):
    stream = client.chat.completions.create(
    model=os.getenv("MODEL"),
    messages=history,
    tools=TOOLS,
    stream=True,
)
    full_response = ""
    calls = {}
    label_printed = False
    try:
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta 

            # 1 Content streaming do we have messages from the llm 
            if delta.content:
                if not label_printed:
                    print("Assistant: ",end="",flush=True)
                    label_printed=True
                print(delta.content,end="",flush=True)
                full_response+=delta.content
            #2 For Tool Calls
            for tc in (delta.tool_calls or []):
                obj = calls.setdefault(
                            tc.index,
                            SimpleNamespace(
                                id = None,
                                function=SimpleNamespace(
                                    name="",
                                    arguments=""
                                )
                            )
                        )
                if tc.id:
                    obj.id = tc.id
                if tc.function and tc.function.name:
                    obj.function.name = tc.function.name
                if tc.function and tc.function.arguments:
                    obj.function.arguments = tc.function.arguments

    except KeyboardInterrupt:
        print("Stopped by You")
        calls={}

    if(label_printed):
        print()

    calls_list = []
    for key in sorted(calls):
        calls_list.append(calls[key])

    return full_response,calls_list 

def build_tool_calls_payload(tool_calls):
    # converts our collected tools calls into the dict format that the API expects inside an assistant message
    payload = [] 
    for tool_call in tool_calls:
        tool_call_dict = {
            "id": tool_call.id,
            "type":"function",
            "function": {
                "name": tool_call.function.name,
                "arguments": tool_call.function.arguments
            }
        }
        payload.append(tool_call_dict)
    return payload

    
MAX_STEPS = 5  # Maximum steps for the agent to avoid infinite tool-calling loops.

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

    # Chained tool calling
    for step in range(MAX_STEPS):
        full_response, tool_calls = stream_chat(messages)

        if not tool_calls:
            break

        log(f"Step {step + 1}: model requested {len(tool_calls)} tool calls.")

        messages.append({
            "role":"assistant",
            "content":full_response,
            "tool_calls":build_tool_calls_payload(tool_calls)
        })

        for tool_call in tool_calls:
            result = run_tool(tool_call)

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(result)
            })

    else:
        # The else block executes if the loop finishes without a break.
        fallback = "Sorry, I couldn't finish this within step limit."

        messages.append({
            "role": "assistant",
            "content": fallback
        })

        print("Assistant: " + fallback)
        continue

