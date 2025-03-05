# Finance Agent

A Python-based AI assistant for financial analysis, combining quantitative modeling with technical analysis capabilities.

## Overview

The Finance Agent is an AI-powered tool that helps analyze financial data using:
- Fama-French model estimation
- Technical analysis with chart generation
- Natural language processing for user interactions
- Context-aware responses with memory management

## Tech Stack

- **Core Language**: Python 3.7+
- **AI/ML**: OpenAI GPT-4 Turbo
- **Data Processing**: pandas, statsmodels, scikit-learn
- **Visualization**: Pillow, ChartImg API
- **Other**: pydantic, requests

## Prerequisites

```bash
# Required environment variables
OPENAI_API_KEY=your-api-key
OPENAI_ORGANIZATION_ID=your-org-id
OPENAI_PROJECT_ID=your-project-id
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/finance-agent.git
cd finance-agent

# Install dependencies
pip install -r requirements.txt
```

## Usage Examples

### Quantitative Analysis
```python
user_input = """
Use the risk-free rate as a flat value of 0.01 and drop the market factor to estimate a model.
Assess performance during the financial crisis.
"""
result = finance_agent(user_input)
```

### Technical Analysis
```python
user_input = "Show me a technical analysis of Tesla's stock performance"
result = finance_agent(user_input)
```

## Architecture

### Core Components

1. **ContextManager**
   - Manages conversation history
   - Implements memory with configurable size
   - Organizes responses by categories

2. **Model Helper**
   - Handles Fama-French model estimation
   - Performs statistical analysis
   - Manages model caching

3. **Chart Image Tool**
   - Generates technical analysis charts
   - Supports multiple timeframes
   - Customizable chart styles

## Workflow

1. User input processing
2. Tool selection (model_helper or chart_img)
3. Analysis execution
4. Response formatting
5. Context management

## Limitations

- Does not provide investment advice
- No real-time data processing
- Limited to supported financial instruments
- API rate limits may apply

## API Reference

### `finance_agent(user_input: str, context_manager: ContextManager = None) -> str`

Main entry point for the application.

**Parameters:**
- `user_input`: User's query or command
- `context_manager`: Optional context management instance

**Returns:**
- Formatted response with analysis results

### `ContextManager`

```python
manager = ContextManager(max_memory=10)
manager.add_to_memory(user_input, slot, response)
```

## License

MIT License - see [LICENSE](LICENSE) for details

## Dependencies

- OpenAI for GPT-4 API
- Financial data providers
- Open-source community

## Support

Create an issue for bugs or feature requests.