# %%
import os
import pickle
import logging
import hashlib
from copy import deepcopy
from pathlib import Path
from openai import OpenAI
from helper import ContextManager
from collections import defaultdict
from pydantic import BaseModel, Field
from typing import Literal, Union, List
from regression_model import FamaFrenchModel


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
    stock_tickers: List[str] = Field(
        ...,
        description="List of stock tickers for which a model or analysis is required. Default is [NVDA].",
    )

    risk_free_rate: Union[bool, float, Literal["flat"]] = Field(
        ...,
        description=(
            "If float, denotes a flat value to use to control variable in model estimation."
            "If a string 'flat', denotes using a flat value from data to control variable in model estimation. "
            "Default is True and this value cannot be False if boolean. "
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
    cols_to_bump: List[str] = Field(
        ...,
        description=(
            "Each string will incorporate 3 elements separated by '-'. "
            "The first element will be the name of columns including the following ['risk_free_rate', 'excess_market_return', 'size_factor', 'value_factor'] "
            r"which the user wants to bump. The second is the desired factor to bump a column by - default is 10% or 0.1. The last element denotes the type of bump to be applied "
            "including ['additive', 'multiplicative']. Default value of this attribute is None."
        ),
    )
    train_test_split: float = Field(
        ...,
        description=(
            "Specifies the proportion of the data to use for training/testing when action is action is out-of-sample. "
            "If training size is specified, set to 1 - training size. Unless a numerical value is specified for "
            "training/testing, strictly default to 0."
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
            "Default is None object. If reference is made to an action or event, get the start date to end date of the action "
            "or the start and end dates of the most important period of the action or event. "
            "If string, the string form should be of the form '<start_date>-<end_date>' where each date has the format 'YYYYMM'."
        ),
    )


# %%
def extract_action_summary(
    user_input: str, context_manager: Union[None, ContextManager] = None
) -> dict:
    logger.info("Starting extraction of action summaries...")

    system_prompt = """
        You are a precise and analytical assistant tasked with interpreting user prompts and identifying the desired actions. 
        Your goal is to produce a clear list of required actions based on the prompt. Possible actions include:
            1. Full Estimation: Assume this by default for modeling tasks unless specified otherwise.
            2. Out-of-Sample (OOS) Estimation: Only include this if it is explicitly requested. If both full estimation and OOS are required, this will be clearly communicated.
            3. Simulation: Trigger this action if the prompt includes references to forecasting or performance assessment.
        
        Guidelines for Accurate Interpretation:
            1. Carefully analyze the prompt to determine the exact actions required.
            2. If the estimation type is unspecified, default to full estimation.
            3. Only select OOS estimation if explicitly stated.
            4. Treat any mention of forecasting or performance evaluation as an indication that simulation is required.
            5. Clearly indicate any uncertainty in your interpretation.
            6. Assign confidence scores to predictions.
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

    system_prompt = """
        You are an analytical assistant tasked with decomposing user prompts into clear, distinct subprompts for each required action. 
        The actions will be from the following list:
            1. Full Estimation
            2. Out-of-Sample (OOS) Estimation
            3. Simulation
        
        Guidelines for Decomposition:
            1. Clearly separate subprompts for different actions.
            2. For each identified action, generate a concise and specific subprompt that clearly describes the task along with all the information provided to achieve it.
            3. Ensure each subprompt is actionable, unambiguous, and tailored to the corresponding action.
            4. Use the provided user action to guide the breakdown.
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
    user_input: str, action: str, context_manager: Union[None, ContextManager] = None
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
    fixed_params = f"{'; '.join(['%s=%.2f' % (k, v) for k, v in result_dump.items() if isinstance(v, float)])}"

    params_to_drop = [
        attr
        for attr in result_dump.keys()
        if isinstance(r := getattr(result, attr), bool) and not r
    ]
    if fixed_params:
        log_info += f"Fixed params: {fixed_params}\n"
    if params_to_drop:
        log_info += f"Parameters to drop: {'; '.join(params_to_drop)}\n"
    if result_dump.get("performance_horizon"):
        log_info += f"Performance Horizon: {result.performance_horizon}\n"
    if result_dump.get("train_test_split"):
        log_info += f"Train-Test Split: {result.train_test_split}\n"

    logger.info(log_info)

    return result_dump


def model_helper(
    user_input: str, context_manager: Union[None, ContextManager] = None
) -> dict:
    logger.debug(f"Input text: {user_input}")

    model_actions = {}
    estimation_actions = ["full-estimation", "out-of-sample"]
    actions = extract_action_summary(user_input, context_manager)

    for action, confidence_score in actions.items():
        if confidence_score >= 0.7:
            sub_prompt = extract_subprompt(user_input, action, context_manager)
            args = extract_model_args(sub_prompt, action, context_manager)
            if action in estimation_actions:
                filename = hashlib.md5(
                    str(str(sorted(list(args.items())))).encode()
                ).hexdigest()
                model_actions[action] = {
                    "sub_prompt": sub_prompt,
                    "args": args,
                    "filename": filename,
                }
            else:
                model_actions[action] = {"sub_prompt": sub_prompt, "args": args}
        else:
            logger.warning(
                f"Confidence score for action: \n{action.title()} is below threshold. Skipping action."
            )

    responses = defaultdict(dict)
    model_cache = root.joinpath("model_cache")
    if not model_cache.is_dir():
        model_cache.mkdir(parents=True)
    simulation_params = model_actions.pop("simulation", None)
    for action, subdict in model_actions.items():
        filename = subdict.get("filename")
        model_path = model_cache / f"{filename}.pkl"

        if model_path.exists():
            with open(model_path, "rb") as f:
                model_obj = pickle.load(f)

        elif action in estimation_actions:
            logger.info("Model is being estimated...")
            model_obj = FamaFrenchModel(**subdict["args"])
            with open(model_path, "wb") as f:
                pickle.dump(model_obj, f, protocol=pickle.HIGHEST_PROTOCOL)

        responses[action] = {
            "sub_prompt": subdict["sub_prompt"],
            "prediction": model_obj.params_df.to_markdown(),
        }

    if simulation_params:
        logger.info("Model performance analysis ongoing...")

        for subdict in model_actions.values():
            filename = subdict.get("filename")
            model_path = model_cache / f"{filename}.pkl"

            with open(model_path, "rb") as f:
                model_obj = pickle.load(f)

            prediction = model_obj.predict(**simulation_params["args"])
            responses["simulation"][action] = {
                "sub_prompt": subdict["sub_prompt"],
                "prediction": prediction.to_markdown(),
            }

    return responses
