"""
If multiple models are intended to be estimated, the user should use different prompts for each to avoid confusion.
The user should also specify the how the parameters for each estimation should be defined. Otherwise, the regular
Fama-French model will be estimated.
"""

# %%
import os
import json
import logging
from openai import OpenAI
from helper import ContextManager
from chart_img_tool import get_chart_img
from model_tool import model_helper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

chat_model = "gpt-4.5-preview"
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    organization=os.getenv("OPENAI_ORGANIZATION_ID"),
    project=os.getenv("OPENAI_PROJECT_ID"),
)


# %%
def finance_agent(user_input, context_manager: ContextManager = None) -> str:

    system_prompt = """
        You are a helpful assistant using tools to process user input. 
        Use model_helper for modelling tasks and analysis and get_chart_img
        when technical analysis is required.
        If user requires technical analysis, do the following:
            1. set symbol to uppercase ticker if only company name is given
            2. default interval to 4h if not explicitly given
            3. default chart style to candle if not explicitly given
            
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]
    if context_manager:
        messages.extend(
            [
                {"role": "assistant", "content": msg["assistant"]}
                for msg in context_manager["final"]
            ]
        )

    tools = [
        {
            "type": "function",
            "function": {
                "name": "model_helper",
                "description": "Estimate a model and perform statistical analysis.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user_input": {"type": "string"},
                    },
                    "required": ["user_input"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_chart_img",
                "description": "Get the chart of a financial asset ticker.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string"},
                        "interval": {"type": "string"},
                        "chart_style": {"type": "string"},
                    },
                    "required": ["symbol", "interval", "chart_style"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        },
    ]

    completion = client.chat.completions.create(
        model=chat_model, messages=messages, tools=tools, tool_choice="auto"
    )

    for tool_call in completion.choices[0].message.tool_calls:
        name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)

        result = globals().get(name)(**args)

        if result.startswith("http") and (
            result.endswith("png") or result.endswith("jpg")
        ):
            content = [
                {
                    "type": "text",
                    "text": user_input,
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": (
                            result if isinstance(result, str) else json.dumps(result)
                        )
                    },
                },
            ]
            messages = [messages[0], {"role": "user", "content": content}]

        else:
            messages.append(completion.choices[0].message)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": (
                        result if isinstance(result, str) else json.dumps(result)
                    ),
                }
            )

    system_prompt = f"""You are a senior quant assistant that answers questions about a model or analyze a stock ticker"
        using some specialized tools. The responses from the tool calls are provided. 
        Use those responses to provide rigorous analysis. If user prompt requires statistical analysis
        which can be performed on any markdown tables or urls, compute those and respond with the relevant statistic(s).
        Present analysis and insights in a conversational format. When addressing financial subjects, provide thorough, 
        easy-to-understand explanations suitable for the user's knowledge. Refrain from giving direct financial 
        recommendations(buy or sell) or making predictions.
        ## Standard Operating Procedure
        1. Interact with the user: Maintain a professional and approachable demeanor.
        2. Conduct relevant analysis and present the findings in an easy-to-understand, conversational manner.
        3. Clarify financial topics: Simplify intricate terms into accessible explanations suitable for the user's knowledge level.
        4. Refrain from offering financial recommendations: Deliver information and analysis without suggesting specific actions.
        5. Verify user understanding: Ask clarifying questions to ensure all needs are met.
    """

    messages = [{"role": "system", "content": system_prompt}] + messages[1:]
    completion_2 = client.chat.completions.create(
        model="gpt-4-turbo", messages=messages
    )

    response = completion_2.choices[0].message.content
    if context_manager:
        context_manager.add_to_memory(user_input, "final", response)
    return response


# %%
if __name__ == "__main__":
    context_manager = ContextManager(max_memory=5)
    user_input = """
        Use the risk-free rate as a flat value of 0.01 the and drop the market factor to estimate a model.
        Also apply multiplicative bump of 15% to the size factor. Assess performance during the financial crisis. 
        Return statistical info of result.
    """
    # user_input = "Perform a technical analysis of Tesla"
    while True:
        result = finance_agent(user_input, context_manager)
        print(result)
        user_input = str(input("Enter new prompt: \n"))
