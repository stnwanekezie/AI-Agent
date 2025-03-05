# %%
import os
import json
import pickle
import logging
from pathlib import Path
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Literal, Union, List
from collections import deque, defaultdict
from regression_model import StockReturnsModel


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

root = Path(__file__).parent.resolve()

# %%


class ContextManager:
    def __init__(self, max_memory=10):
        self.memory = defaultdict(lambda: deque(maxlen=max_memory))

    def __getitem__(self, key):
        return self.memory[key]

    def add_to_memory(self, user_input, slot, model_response):
        self.memory[slot].append({"user": user_input, "assistant": model_response})

    def get_context(self, slot):
        return "\n".join(
            [
                f"User: {msg['user']}\nAssistant: {msg['assistant']}"
                for msg in self.memory[slot]
            ]
        )


class SummaryActions(BaseModel):
    actions: List[str] = Field(
        ...,
        description="""
            This is a list constructed from user input. The list elements is one or more
            of ['full-estimation', 'out-of-sample', and 'simulation']. Key values may include 
            'full-estimation' or 'out-of-sample' but not both and 'forecasting', if applicable
        """,
    )
    confidence_scores: List[float] = Field(
        ...,
        description="Degree of confidence in a prediction. Value should be between 0 and 1, 1 being the highest.",
    )


class UserPrompts(BaseModel):
    user_prompt: str = Field(
        ...,
        description="""
            The extracted subprompts should exclude extraneous context and only contain information essential for 
            effective execution. Ensure no relevant information is lost.
        """,
    )


class EstimationArgs(BaseModel):
    stock_ticker: str = Field(
        ...,
        description="Ticker of stock for which a model or analysis is required. Default is NVDA.",
    )

    risk_free_rate: Union[bool, float, Literal["flat"]] = Field(
        ...,
        description=(
            "If boolean, denotes if the risk-free factor should be used for model estimation. Default is True."
            "If float, denotes a flat value to use to control variable in model estimation."
            "If a string 'flat', denotes using a flat value from data to control variable in model estimation."
        ),
    )
    excess_market_return: Union[bool, float, Literal["flat"]] = Field(
        ...,
        description=(
            "If boolean, denotes if the market factor should be used for model estimation. Default is True."
            "If float, denotes a flat value to use to control variable in model estimation."
            "If a string 'flat', denotes using a flat value from data to control variable in model estimation."
        ),
    )
    size_factor: Union[bool, float, Literal["flat"]] = Field(
        ...,
        description=(
            "If boolean, denotes if the small minus big (SMB) factor should be used for model estimation. Default is True. "
            "If float, denotes a flat value to use to control variable in model estimation. "
            "If a string 'flat', denotes using a flat value from data to control variable in model estimation."
        ),
    )
    value_factor: Union[bool, float, Literal["flat"]] = Field(
        ...,
        description=(
            "If boolean, denotes if the high minus low (HML) factor should be used for model estimation. Default is True. "
            "If float, denotes a flat value to use to control variable in model estimation. "
            "If a string 'flat', denotes using a flat value from data to control variable in model estimation."
        ),
    )

    train_test_split: float = Field(
        ...,
        description=(
            "Specifies the proportion of the data to use for training/testing when action is action is out-of-sample. "
            "If training size is specified, set to 1 - training size. If training/testing is mentioned "
            "but a numerical value is not specified, set to 0.2. "
            "Otherwise, strictly default to 0."
        ),
    )

    add_constant: bool = Field(
        ...,
        description=(
            "Specifies if a constant term should be added to the model. "
            "Default is True."
        ),
    )


class SimulationArgs(BaseModel):
    simulation_horizon: Union[str, None] = Field(
        ...,
        description=(
            "Specifies the time period to use for simulation or performance analysis."
            "Default is None. If reference is made to an action or event, get the start date to end date of the action "
            "or the start and end dates of the most important period of the action or event. "
            "If string, the string form should be of the form '<start_date>-<end_date>' where each date has the format 'YYYYMM'."
        ),
    )


# %%
def extract_action_summary(
    user_input: str, context_manager: ContextManager = None
) -> dict:
    logger.info("Starting extraction of action summaries...")

    system_prompt = """
        You are a careful assistant that given a model estimation or performance evaluation prompt, exercises great care to 
        decompose it into a list of clear summary action steps including one or more of the following:
        [full estimation, out-of-sample (OOS) estimation, simulation]. Unless estimation is specified as OOS, assume full-estimation.
        If full estimation and OOS are simultaneously desired, this will be clearly communicated.
        Reference to forecasting or performance assessment imply that action is simulation.
        Assign confidence scores to predictions.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]
    if context_manager:
        messages.extend(
            [
                {"role": "assistant", "content": msg["assistant"]}
                for msg in context_manager.memory["actions"]
            ]
        )
    completion = client.beta.chat.completions.parse(
        model=chat_model,
        messages=messages,
        response_format=SummaryActions,
    )

    result = completion.choices[0].message.parsed
    if context_manager:
        context_manager.add_to_memory(user_input, "actions", str(result.model_dump()))
    logger.info(f"Model action summaries: {', '.join(result.actions)}")

    return dict(zip(result.actions, result.confidence_scores))


def extract_subprompt(
    user_input: str, action: str, context_manager: ContextManager = None
) -> UserPrompts:
    logger.info(f"Creating subprompt for {action}...")

    system_prompt = f"""
        Given a user prompt related to modelling-related processes, extract and generate a new  
        subprompt that strictly includes the details relevant to the {action} step.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]
    if context_manager:
        messages.extend(
            [
                {"role": "assistant", "content": msg["assistant"]}
                for msg in context_manager.memory["subprompts"]
            ]
        )
    completion = client.beta.chat.completions.parse(
        model=chat_model,
        messages=messages,
        response_format=UserPrompts,
    )

    result = completion.choices[0].message.parsed
    if context_manager:
        context_manager.add_to_memory(
            user_input, "subprompts", str(result.model_dump())
        )
    return result.user_prompt


# %%


def extract_model_args(
    user_input: str, action: str, context_manager: ContextManager = None
) -> Union[EstimationArgs, SimulationArgs]:

    logger.info("Starting model parameters extraction...")

    system_prompt = "You are a helpful model parameter extractor."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]
    if context_manager:
        messages.extend(
            [
                {"role": "assistant", "content": msg["assistant"]}
                for msg in context_manager.memory["args"]
            ]
        )

    completion = client.beta.chat.completions.parse(
        model=chat_model,
        messages=messages,
        response_format=SimulationArgs if action == "simulation" else EstimationArgs,
    )

    result = completion.choices[0].message.parsed
    result_dump = result.model_dump()
    if context_manager:
        context_manager.add_to_memory(user_input, "args", str(result_dump))

    log_info = "Parameter extraction complete with the following:\n"
    fixed_params = f"{'; '.join(['%s=%.2f' % (k, v) for k, v in result_dump.items() if isinstance(v, float)])}\n"

    params_to_drop = [
        attr
        for attr in result_dump.keys()
        if isinstance(r := getattr(result, attr), bool) and not r
    ]
    if fixed_params:
        log_info += "Fixed params: " + fixed_params
    if params_to_drop:
        log_info += f"Parameters to drop: {'; '.join(params_to_drop)}\n"
    if result_dump.get("performance_horizon"):
        log_info += f"Performance Horizon: {result.performance_horizon}\n"
    if result_dump.get("train_test_split"):
        log_info += f"Train-Test Split: {result.train_test_split}\n"

    logger.info(log_info)

    return result_dump


def response_processor(
    user_input, responses, context_manager: ContextManager = None
) -> str:
    system_prompt = (
        "You are a senior quant assistant that answers questions about a model. "
        f"The model returned the following response: {responses}. "
        "Use the model responses to inform your answers. "
        "If user prompt requires statistical analysis which can be performed on any markdown tables "
        "in the response, compute and respond with the relevant statistic(s) in an informative way."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]
    if context_manager:
        messages.extend(
            [
                {"role": "assistant", "content": msg["assistant"]}
                for msg in context_manager.memory["final"]
            ]
        )

    completion = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=messages,
    )

    response = completion.choices[0].message.content
    if context_manager:
        context_manager.add_to_memory(user_input, "final", response)
    return response


def model_helper(user_input: str, context_manager: ContextManager = None) -> dict:
    logger.debug(f"Input text: {user_input}")

    model_actions = {}
    actions = extract_action_summary(user_input, context_manager)
    for action, confidence_score in actions.items():
        if confidence_score >= 0.7:
            sub_prompt = extract_subprompt(user_input, action, context_manager)
            args = extract_model_args(sub_prompt, action, context_manager)
            model_actions[action] = {"sub_prompt": sub_prompt, "args": args}
        else:
            logger.warning(
                f"Confidence score for action: \n{action.title()} is below threshold. Skipping action."
            )

    estimation_actions = ["full-estimation", "out-of-sample"]
    if any([action in model_actions.keys() for action in estimation_actions]):
        for action in estimation_actions:
            try:
                ordered_actions = {action: model_actions.pop(action)}
                model_actions = {**ordered_actions, **model_actions}
            except KeyError:
                continue

    responses = []
    model_cache = root.joinpath("model_cache")
    model_name = None
    for action, subdict in model_actions.items():
        if action in ["full-estimation", "out-of-sample"]:
            logger.info("Model is being estimated...")

            args_hash = hash(str(subdict["args"]))
            model_name = model_cache / f"{args_hash}.pkl"
            if not (model_name).exists():
                model_obj = StockReturnsModel(**subdict["args"])
                with open(model_name, "wb") as f:
                    pickle.dump(model_obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            logger.info("Model performance analysis ongoing...")

            with open(model_name, "rb") as f:
                model_obj = pickle.load(f)
            prediction = model_obj.predict(**subdict["args"])
            responses.append(
                {
                    "sub_prompt": subdict["sub_prompt"],
                    "prediction": prediction.to_markdown(),
                }
            )

    return responses
