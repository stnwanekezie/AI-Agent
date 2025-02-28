# Model Agent

The `model_agent.py` script is designed to assist with the estimation and performance evaluation of financial models. It uses the OpenAI API to parse user prompts and extract relevant actions and parameters for model estimation and simulation. Model implementation is 
contained in `regression_model.py`.

## Features

- **Model Estimation**: Supports full estimation and out-of-sample estimation.
- **Simulation**: Allows for forecasting and backtesting over specified time periods.
- **Parameter Extraction**: Extracts model parameters from user prompts.
- **Response Processing**: Processes model responses and provides statistical analysis.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/stnwanekezie/AI-Agent.git
    cd AI-Agent
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Set up your OpenAI API credentials as environment variables:
    ```bash
    export OPENAI_API_KEY="your-api-key"
    export OPENAI_ORGANIZATION_ID="your-organization-id"
    export OPENAI_PROJECT_ID="your-project-id"
    ```

2. Run the script with a user input prompt. Example user input:
    ```plaintext
    Use the risk-free rate as a flat value of 0.01 and drop the market factor to estimate a model. Assess performance during the financial crisis. Return statistical info of result.
    ```

## Functions

### [extract_action_summary(user_input: str) -> dict]
Extracts action summaries from the user input.

### [extract_subprompt(user_input: str, action: str) -> UserPrompts]
Generates a subprompt for a specific action.

### [extract_model_args(user_input: str, action: str) -> Union[EstimationArgs, SimulationArgs]]
Extracts model parameters for a specific action.

### [response_processor(user_input, responses)]
Processes model responses and provides statistical analysis.

### [model_helper(user_input: str) -> dict]
Main function that handles the entire workflow from extracting actions to processing responses.


## Pydantic Classes for Structured Outputs

### [SummaryActions]
Defines the structure for action summaries.

### [UserPrompts]
Defines the structure for user subprompts.

### [EstimationArgs]
Defines the structure for model estimation parameters.

### [SimulationArgs]
Defines the structure for simulation parameters.

## Logging

The script uses the [logging] module to log information, warnings, and errors. Logs are formatted with timestamps and log levels.

## Example

```python
if __name__ == "__main__":
    user_input = """
        Use the risk-free rate as a flat value of 0.01 and drop the market factor to estimate a model. 
        Assess performance during the financial crisis. Return statistical info of result.
    """
    result = model_helper(user_input)
    print(result)