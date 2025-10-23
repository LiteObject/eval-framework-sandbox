# Eval Framework Sandbox

A simple Q&A bot for technical documentation designed to test and compare different LLM evaluation frameworks including DeepEval, LangChain Evaluation, RAGAS, and OpenAI Evals.

## Purpose

This project serves as a testbed for comparing how different evaluation frameworks assess the same RAG (Retrieval-Augmented Generation) system.

## Key Concepts

This project evaluates a RAG (Retrieval-Augmented Generation) system. Here are a few key concepts to help you understand the components:

-   **RAG (Retrieval-Augmented Generation)**: This is a technique where a large language model's knowledge is supplemented with information retrieved from other sources (in this case, our local documents). The process has two main steps:
    1.  **Retrieval**: A search algorithm (like TF-IDF) finds relevant documents based on the user's query.
    2.  **Generation**: A language model takes the retrieved documents and the original query to generate a comprehensive answer.

-   **Ground Truth**: In the context of evaluation, "ground truth" refers to the ideal or perfect answer to a given question. We use the ground truth dataset (`data/ground_truth.json`) as a benchmark to measure how accurate and relevant the Q&A bot's answers are.

-   **TF-IDF (Term Frequency-Inverse Document Frequency)**: This is the retrieval algorithm used by the Q&A bot to find relevant documents. It works by assigning a score to each word in a document based on two factors:
    -   **Term Frequency (TF)**: How often a word appears in a specific document.
    -   **Inverse Document Frequency (IDF)**: How rare or common the word is across all documents.
    
    This allows the system to prioritize words that are important to a specific document over common words that appear everywhere (like "the" or "and").

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

These integrations are opt-in. Install the additional dependencies with:

```bash
pip install .[eval]
```

Each runner expects the dataset built from the JSON files in `data/questions.json`
and `data/ground_truth.json`. The helper below mirrors what the runners use
internally:

```python
from pathlib import Path
from evaluations.utils import load_dataset_from_files

dataset = load_dataset_from_files(
     Path("data/questions.json"),
     Path("data/ground_truth.json"),
)
```

### DeepEval

1. Set `DEEPEVAL_API_KEY` in `.env` if you plan to submit results to the hosted
    DeepEval service (local scoring works without it).
2. Run the runner programmatically:

    ```python
    from evaluations.deepeval_runner import DeepEvalRunner

    runner = DeepEvalRunner()
    result = runner.evaluate(dataset)
    print(result.score, result.details)
    ```

    The report is also written to `results/deepeval_result.json`.

### LangChain Evaluation

1. Choose your backend:
    - Remote OpenAI models: set `OPENAI_API_KEY` and optionally
      `LANGCHAIN_OPENAI_MODEL` (defaults to `gpt-3.5-turbo`).
    - Local Ollama: set `LANGCHAIN_USE_OLLAMA=true`, `OLLAMA_MODEL`, and
      optionally `OLLAMA_BASE_URL`; no OpenAI key required.
2. Invoke the runner:

    ```python
    from evaluations.langchain_eval_runner import LangChainEvalRunner

    runner = LangChainEvalRunner()
    result = runner.evaluate(dataset)
    print(result.score, result.details)
    ```

    LangChain will call the configured chat model to grade responses and store
    the output at `results/langchain_result.json`.

### RAGAS

1. Install the `ragas` extras (already included in `.[eval]`). Some metrics call
    an LLM; set `OPENAI_API_KEY` or configure RagAS to use a local model before
    running.
2. Evaluate the dataset:

    ```python
    from evaluations.ragas_runner import RagasRunner

    runner = RagasRunner()
    result = runner.evaluate(dataset)
    print(result.score, result.details)
    ```

    The raw metric results are saved to `results/ragas_result.json`.

### OpenAI Evals

This repository only prepares the dataset and relies on OpenAI's CLI for the
actual evaluation. Ensure `evals` is installed and `OPENAI_API_KEY` is set, then
use `evaluations/openai_eval_runner.py` to export a dataset and follow the
[OpenAI Evals documentation](https://github.com/openai/evals) to launch the
experiments with `oaieval`.

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