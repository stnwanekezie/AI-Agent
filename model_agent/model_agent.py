"""
If multiple models are intended to be estimated, the user should use different prompts for each to avoid confusion.
The user should also specify the how the parameters for each estimation should be defined. Otherwise, the regular
Fama-French model will be estimated.
"""

# %%
import os
import pickle
import logging
from pathlib import Path
from openai import OpenAI
from typing import Literal, Union, List
from pydantic import BaseModel, Field
from regression_model import StockReturnsModel

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

root = Path(__file__).parent.resolve()

# %%


class SummaryActions(BaseModel):
    actions: List[str] = Field(
        ...,
        description="""
            This is a list constructed from user input. The list elements is one or more
            of ['full-estimation', 'out-of-sample', and 'forecasting']. Key values may include 
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


class ModelParameters(BaseModel):
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

    performance_horizon: str = Field(
        ...,
        description=(
            "Specifies the time period to use for analysis."
            "Default is an empty string. If reference is made to an action, get the start date to end date of the action "
            "or the start and end dates of the most important period of the action. "
            "The string form should be of the form '<start_date>-<end_date>' where each date has the format 'YYYYMM'."
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


# %%
def extract_action_summary(user_input: str) -> dict:
    logger.info("Starting extraction of action summaries...")

    system_prompt = """
        Given a model estimation or performance evaluation prompt, decompose it into clear summary action steps 
        including one or more of [full estimation, out-of-sample estimation, performance assessment] in a list. 
        The prompt will specify either a full estimation or an out-of-sample estimation (but not both), as well as 
        performance assessment, if applicable. 
        Assign confidence scores to predictions.
    """
    completion = client.beta.chat.completions.parse(
        model=chat_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ],
        response_format=SummaryActions,
    )

    result = completion.choices[0].message.parsed
    logger.info(f"Model action summaries: {', '.join(result.actions)}")

    return dict(zip(result.actions, result.confidence_scores))


def extract_subprompt(user_input: str, action: str) -> UserPrompts:
    logger.info(f"Creating subprompt for {action}...")

    system_prompt = f"""
        Given a user prompt related to modelling-related processes, extract and generate a new  
        subprompt that strictly includes the details relevant to the {action} step.
    """

    completion = client.beta.chat.completions.parse(
        model=chat_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ],
        response_format=UserPrompts,
    )

    result = completion.choices[0].message.parsed
    return result.user_prompt


# %%


def extract_model_parameters(user_input: str) -> ModelParameters:

    logger.info("Starting model parameters extraction...")

    system_prompt = "You are a helpful model parameter extractor."
    completion = client.beta.chat.completions.parse(
        model=chat_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ],
        response_format=ModelParameters,
    )

    result = completion.choices[0].message.parsed
    result_dump = result.model_dump()

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
    if result.performance_horizon:
        log_info += f"Performance Horizon: {result.performance_horizon}\n"
    if result.train_test_split:
        log_info += f"Train-Test Split: {result.train_test_split}\n"

    logger.info(log_info)

    return result_dump


def response_processor(user_input, responses):
    system_prompt = (
        "You are a senior quant analyst assistant that answers questions about a model. "
        f"The model returned the following response: {responses}. "
        "Use the model responses to inform your answers. "
        "If user prompt requires statistical analysis which can be performed on any markdown tables "
        "in the response, compute and respond with the relevant statistic(s) in an informative way."
    )

    completion = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ],
    )

    return completion.choices[0].message.content


def model_helper(user_input: str) -> dict:
    logger.debug(f"Input text: {user_input}")

    model_actions = {}
    actions = extract_action_summary(user_input)
    for action, confidence_score in actions.items():
        if confidence_score >= 0.7:
            sub_prompt = extract_subprompt(user_input, action)
            args = extract_model_parameters(sub_prompt)
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
            prediction = model_obj.predict()
            responses.append(
                {
                    "sub_prompt": subdict["sub_prompt"],
                    "prediction": prediction.to_markdown(),
                }
            )

    final_response = response_processor(user_input, responses)

    return final_response


# %%
if __name__ == "__main__":
    user_input = """
        Use the risk-free rate as a flat value of 0.05 the and drop the market factor to estimate a model. 
        Assess performance during the financial crisis.
    """
    result = model_helper(user_input)
    print(result)
