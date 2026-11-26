from datetime import datetime
import json
import requests

from Utilities.utilities import log
from rag import search

def get_current_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def get_city_weather(city):
    url = f"https://wttr.in/{city}?format=3"
    response = requests.get(url, timeout=10)
    response.raise_for_status() # Raise an error for bad responses
    return response.text if response.status_code == 200 else "Could not fetch weather data."

def get_user_city():
    return "Ujjain"

def search_knowledge_base(query):
    results = search(query,top_k=3)

    log(f"📚 Knowledge base search: {query}")
    for r in results:
        log(f"   {r['score']:.2f} - {r['source']} #{r['chunk_id']}")

    blocks = []

    for r in results:
        sourceFileName = r['source']
        chunkId = r['chunk_id']
        chunkText = r['text']

        header = f"[{sourceFileName} #{chunkId}]"
        block  = f"{header}\n{chunkText}"
        blocks.append(block)

    return "\n\n".join(blocks)
    




TOOLS = [
        {
            "type" : "function",
            "function" : {
                "name" : "get_current_time",
                "description" : "Returns the current date and time as a formatted string.",
                "parameters" : {"type": "object", "properties" : {}},
            }
        },
        {
            "type" : "function",
            "function" : {
                "name" : "get_city_weather",
                "description" : "Tells us the current weather of a city.",
                "parameters" : {
                    "type": "object",
                    "properties" : {
                        "city" : {
                            "type" : "string",
                            "description" : "The name of the city to get the weather for."
                        }
                    },
                    "required" : ["city"]
                }
            }
        },
        {
            "type":"function",
            "function" : {
                "name":"get_user_city",
                "description":"Tells us the city of the user, where the user lives or my city.",
                "parameters" : {"type":"object", "properties":{}},
            }
        },
        {
            "type": "function",
            "function": {
                "name": "search_knowledge_base",
                "description" : "Searches the Zorvex Technologies company knowledge base: leave, medical, HR, hiring, tech, incident "
                                "reports and travel expense policies. Use it for any question about company policies or documents. "
                                "Returns the most relevant passages with their source.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The user's question, rewritten as a short, self-contained search query.",
                        }
                    },
                    "required": ["query"],
                },
            },
        },
    ]


# Our llm will just send funcation name so how we will call it , so lets make a map for it

FUNCTION_NAME_TO_FUNCTION_MAP = {
    "get_current_time" : get_current_time,
    "get_city_weather" : get_city_weather,
    "get_user_city" : get_user_city,
    "search_knowledge_base" : search_knowledge_base,


}

def run_tool(tool_call):
    func_name = tool_call.function.name
    if func_name not in FUNCTION_NAME_TO_FUNCTION_MAP:
        raise ValueError(f"Function '{func_name}' is not defined in the function map.")
    
    try:
        func_args = json.loads(tool_call.function.arguments or "{}")
        log(f"Calling function '{func_name}' with arguments: {func_args}")
        fn = FUNCTION_NAME_TO_FUNCTION_MAP[func_name]
        result = fn(**func_args)  # Unpack dictionary into function arguments
    except Exception as e:
        result = f"Error while executing function '{func_name}': {str(e)}"

    log(f"Function '{func_name}' returned: {result}")
    return result