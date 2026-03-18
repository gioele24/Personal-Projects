# Mathematical Reasoning Agent

This project implements a local AI agent specialized in mathematical reasoning and symbolic computation.  
The agent uses LangChain, an Ollama‑powered LLM, and custom tools for executing WolframScript code and generating interactive Plotly visualizations.

## Features

- **Stateful agent** with conversational memory  
- **WolframScript integration** for symbolic and numerical computation  
- **Plotly tool** for generating interactive mathematical plots  
- **Custom system prompt** to guide reasoning and tool usage  
- **Logging system** to track tool calls and agent decisions  
- **CLI interface** for interactive mathematical queries

## Usage

1. Ensure WolframScript is installed and accessible from your system.  
2. Run the agent script to start the interactive CLI.  
3. Ask mathematical questions; the agent will decide whether to use WolframScript, Plotly, or pure LLM reasoning.

Example queries:
- `Integrate x^2 * sin(x)`  
- `Solve x^3 - 2x + 1 = 0`  
- `Plot sin(t^2) from 0 to 10`

## Notes

- Plotly outputs are saved as HTML files in the `plots/` directory.  
- The agent maintains conversational state across turns.  
- Logs are written to `log.txt` for debugging and transparency.

## Technologies

- LangChain  
- Ollama (local LLMs)  
- WolframScript  
- Plotly  
- Python
