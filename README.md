# Eval Framework Sandbox

A simple Q&A bot for technical documentation designed to test and compare different LLM evaluation frameworks including DeepEval, LangChain Evaluation, RAGAS, and OpenAI Evals.

## Purpose

This project serves as a testbed for comparing how different evaluation frameworks assess the same RAG (Retrieval-Augmented Generation) system.

## Quick Start

1. Clone the repository
```bash
git clone https://github.com/LiteObject/eval-framework-sandbox.git
cd eval-framework-sandbox
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Set up environment variables
```bash
cp .env.example .env
# Edit .env with your API keys (optional unless running remote evals)
```

4. Ask a question
```bash
python -m src.main "How do you install the Python requests library?"
```
The bot will print a synthesized answer and list matching documents.

5. Run the unit tests
```bash
pytest
```

6. (Optional) Try an evaluation framework
    - Update `.env` with the relevant API keys *or* enable the Ollama flag for a local model (details below).
    - Install extras: `pip install -r requirements.txt` already includes optional libs, or `pip install .[eval]` after editable install.
    - Use the runner scripts in `evaluations/` as starting points; each script writes results into `results/`.

### Using a local Ollama model with LangChain evaluation

The core QA bot already runs fully offline using TF-IDF retrieval. If you also want LangChain's evaluators to call a local Ollama model instead of OpenAI:

1. [Install Ollama](https://ollama.ai/download) and pull a model, e.g. `ollama pull llama3`.
2. Set the following environment variables (via `.env` or your shell):
   - `LANGCHAIN_USE_OLLAMA=true`
   - `OLLAMA_MODEL=llama3` (or any other pulled model)
   - Optionally `OLLAMA_BASE_URL=http://localhost:11434` if you're running Ollama on a non-default host/port.
3. Leave `OPENAI_API_KEY` blank; the LangChain evaluator will detect the Ollama flag and use `ChatOllama`.

If `LANGCHAIN_USE_OLLAMA` is `false`, the evaluator falls back to `ChatOpenAI` and expects a valid `OPENAI_API_KEY` plus `LANGCHAIN_OPENAI_MODEL` (defaults to `gpt-3.5-turbo`).

## Evaluation Frameworks

- **DeepEval**: Unit testing for LLMs
- **RAGAS**: RAG-specific evaluation metrics
- **LangChain Evaluation**: Native LangChain eval tools
- **OpenAI Evals**: OpenAI's evaluation framework

## Project Structure

- `data/`: Test questions, ground truth, and source documents
- `src/`: Core Q&A bot implementation
- `evaluations/`: Framework-specific evaluation scripts
- `results/`: Evaluation results and comparisons (gitignored except for `.gitkeep`)

## Metrics Evaluated

- Answer Correctness
- Context Relevance
- Faithfulness
- Answer Similarity
- Response Time
- Hallucination Rate