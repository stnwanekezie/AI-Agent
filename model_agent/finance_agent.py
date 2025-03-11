"""
If multiple models are intended to be estimated, the user should use different prompts for each to avoid confusion.
The user should also specify the how the parameters for each estimation should be defined. Otherwise, the regular
Fama-French model will be estimated.
"""

# %%
import os
import json
import logging
from typing import Union
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

chat_model = "gpt-4o"
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    organization=os.getenv("OPENAI_ORGANIZATION_ID"),
    project=os.getenv("OPENAI_PROJECT_ID"),
)


# %%
def finance_agent(
    user_input, context_manager: Union[None, ContextManager] = None
) -> str:

    system_prompt = """
        You are a helpful assistant using tools to process user input. 
        Use model_helper for modelling tasks or related analysis and use get_chart_img
        when technical analysis is required. Where relevant, use context to refine the user prompt to
        ensure that details which are left to be inferred from conversation history are captured.
        If user requires technical analysis, do the following:
            1. if only company name is given, get the company ticker in uppercase
            2. set symbol as the chart-img.com representation of EXCHANGE:TICKER
            3. if exchange is not specified by user and cannot be inferred for ticker, default to SP 
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

        result = globals().get(name)(**args, context_manager=context_manager)

        if (
            isinstance(result, str)
            and result.startswith("http")
            and (result.endswith("png") or result.endswith("jpg"))
        ):
            content = [
                {
                    "type": "text",
                    "text": user_input,
                },
                {
                    "type": "image_url",
                    "image_url": {"url": result},
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
        6. Make sure that analysis are performed to completion.
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
        Estimate a model for the returns of Tesla and Microsoft. Use a 20% test split and assess
        the forecasting performance of the models between 2020 and 2022.
    """
    user_input = """
        Reestimate models for both stocks using a flat risk-free rate of 0.01 dropping the market factor
        and applying a 15% multiplicative bump to the size factor. Assess the performance during the
        financial crisis and return summary statistics.
    """
    # user_input = "Perform a technical analysis of the nasdaq inc"
    while True:
        result = finance_agent(user_input, context_manager)
        print(result)
        user_input = str(input("Enter new prompt: \n"))
