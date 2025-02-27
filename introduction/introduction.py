# %%

import os
import json
import requests
from openai import OpenAI
from typing import Optional, Literal
from pydantic import BaseModel, Field


client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    organization=os.getenv("OPENAI_ORGANIZATION_ID"), 
    project=os.getenv("OPENAI_PROJECT_ID")
)

# %%
# limerick output
completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the purpose of life?"},
    ],
)
response = completion.choices[0].message.content
print(response)

# %%
# structured output - still in beta
class CalendarEvent(BaseModel):
    name: str
    date: str
    participants: list[str]
    location: Optional[str] = None

system_prompt = "Extract the event information in a piece of text."
completion = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Alice and Bob are going to the park on Saturday to play chess."},
    ],
    response_format=CalendarEvent
)

event = completion.choices[0].message.parsed
event.name
event.date
event.participants

# %%
# weather extraction tool with structured output
def get_weather(latitude, longitude):

    response = requests.get(
        f'https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,wind_speed_10m&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m'
        # f"https://api.openweathermap.org/data/2.5/weather?lat={latitude}&lon={longitude}&appid={os.getenv('OPENWEATHER_API_KEY')}"
    )
    data = response.json()
    return data['current']

def call_function(name, args):
    if name == "get_weather":
        return get_weather(**args)
    # elif name == "get_news":
    #     return get_news(**args)
    
    
class WeatherResponse(BaseModel):
    temperature: float = Field(..., description="The current temperature in Celsius for the given location.")
    wind_speed: float = Field(..., description="The current wind speed in m/s for the given location.")
    response: str = Field(..., description="A natural language response to the user's question.")
    weather_type: Optional[Literal["sunny", "rainy", "cloudy", "snowy"]] = Field(None, description="The type of weather expected.")

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current temperature for a given location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "latitude": {"type": "number"},
                    "longitude": {"type": "number"}
                    # "location": {
                    #     "type": "string",
                    #     "description": "City and country e.g. Bogotá, Colombia"
                    # }
                },
                "required": [
                    # "location"
                    "latitude",
                    "longitude"
                ],
                "additionalProperties": False
            },
            "strict": True
        }
    }
]

system_prompt = "You are a helpful weather assistant."
messages=[
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "What is the weather like in Paris today?"}
]

completion = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    tools=tools
)

completion.model_dump()

for tool_call in completion.choices[0].message.tool_calls:
    name = tool_call.function.name
    # convert string to dict - alternative to eval or ast.literal_eval
    args = json.loads(tool_call.function.arguments)
    
    # To give the agent more context, we can add the messages from the completion
    messages.append(completion.choices[0].message)

    # Call the function and include the result in the messages for more context
    result = call_function(name, args)
    messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(result)})

completion_2 = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=messages,
    tools=tools,
    response_format=WeatherResponse
)

final_response = completion_2.choices[0].message.parsed
final_response.temperature
final_response.wind_speed
final_response.weather_type
final_response.response

# %%
def search_kb(question: str):
    """
    Load the whole knowledge base from the JSON file.
    (This is a mock function for demonstration purposes, we don't search)
    """
    with open("kb.json", "r") as f:
        return json.load(f)

def call_function(name, args):
    if name == "search_kb":
        return search_kb(**args)

tools = [
    {
        "type": "function",
        "function": {
            "name": "search_kb",
            "description": "Search the knowledge base for relevant information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"}
                },
                "required": ["question"],
                "additionalProperties": False
            },
            "strict": True
        }
        
    }
]

system_prompt = "You are a helpful assistant that answers question from the knowledge base about our e-commerce store."
messages=[
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "What is the return policy for your store?"}
]

completion = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    tools=tools
)

for tool_call in completion.choices[0].message.tool_calls:
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)
    messages.append(completion.choices[0].message)
    
    result = call_function(name, args)
    messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(result)})
    
completion_2 = client.chat.completions.create(
    model="gpt-4o",
    messages=messages
)

print(completion_2.choices[0].message.content)

class KBResponse(BaseModel):
    answer: str = Field(..., description="The response from the knowledge base.")
    source: int = Field(..., description="The record id of the information in the knowledge base.")

completion_3 = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=messages,
    tools=tools,
    response_format=KBResponse
)

final_response = completion_3.choices[0].message.parsed
final_response.answer
final_response.source


# %%
