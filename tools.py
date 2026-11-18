from datetime import datetime
import requests

def get_current_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def get_city_weather(city):
    url = f"https://wttr.in/{city}?format=3"
    response = requests.get(url, timeout=10)
    return response.text if response.status_code == 200 else "Could not fetch weather data."

def get_user_city():
    return "Ujjain"

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
                }
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

]


# Our llm will just send funcation name so how we will call it , so lets make a map for it

FUNCTION_NAME_TO_FUNCTION_MAP = {
    "get_current_time" : get_current_time,
    "get_city_weather" : get_city_weather,
    "get_user_city" : get_user_city

}