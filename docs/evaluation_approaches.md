# Evaluation Approaches for LLM Systems

This document summarizes common evaluation approaches for LLM and RAG systems, how they work, and when to use each. It complements the main `README.md` and the `evaluation_comparison.ipynb` notebook.

## 1. LLM as a Judge

**Idea:** Use a language model to grade or score the outputs of another AI system.

- **How it works:**
  - The judge LLM sees the **question**, the **model prediction**, and often the **ground-truth answer**.
  - It decides how correct, relevant, or helpful the prediction is and returns a score (e.g., 0–1) or a label ("correct" / "incorrect").
- **Example in this repo:** `LangChainEvalRunner` (LangChain Evaluation).
- **Pros:**
  - Captures **semantic similarity** and paraphrasing.
  - Can consider **tone**, **completeness**, and **reasoning quality**.
  - Easy to adapt across domains by changing the grading prompt.
- **Cons:**
  - Requires access to an LLM (API or local model).
  - Non‑deterministic: repeated runs may vary slightly.
  - Inherits **biases** and failure modes of the judge model.
- **Best for:**
  - When you care about **human‑like judgment** and semantic correctness.
  - Comparing prompts or model variants for realistic answer quality.

## 2. Rule‑Based String Metrics

**Idea:** Evaluate answers using simple string‑level comparisons against ground truth.

Common metrics:

- **Exact Match (EM):** Prediction must exactly equal the ground truth.
- **Word Overlap / Jaccard Similarity:**
  - $\text{score} = \frac{|W_{pred} \cap W_{ref}|}{|W_{pred} \cup W_{ref}|}$ where $W$ are word sets.
- **Edit Distance (Levenshtein):** Minimum number of edits (insert/delete/replace) to turn prediction into ground truth.
- **BLEU / ROUGE:** N‑gram overlap metrics originally from MT/summarization.

- **Examples in this repo:**
  - `DeepEvalRunner`: offline **word‑overlap** Jaccard.
  - `RagasRunner`: offline **token‑overlap** Jaccard.
- **Pros:**
  - Very fast and **fully offline**.
  - Deterministic and easy to implement.
- **Cons:**
  - Penalizes paraphrases and synonyms.
  - Does not understand semantics or factual correctness.
- **Best for:**
  - **Sanity checks** and quick regression tests.
  - Simple tasks with short, formulaic answers.

## 3. Embedding‑Based Similarity

**Idea:** Compare prediction and ground truth in a **vector space** using embeddings.

- **How it works:**
  - Encode prediction and ground truth into vectors using an embedding model.
  - Compute a similarity metric (usually **cosine similarity**).
- **Example in this repo:** `EmbeddingEvalRunner` using `sentence-transformers` (`all-MiniLM-L6-v2`).
- **Pros:**
  - Captures **semantic similarity** better than raw word overlap.
  - Still cheaper and more deterministic than full LLM‑as‑judge.
- **Cons:**
  - Requires an embedding model and some numeric tooling.
  - Can still miss nuances like factual grounding or subtle errors.
- **Best for:**
  - Middle ground between rule‑based and LLM‑based evaluation.
  - Comparing high‑level semantic similarity of answers.

## 4. Reference‑Free / No‑Ground‑Truth Metrics

**Idea:** Evaluate outputs **without** a labeled ground‑truth answer.

Examples:

- **Fluency / Coherence:** Is the answer grammatically correct and well‑structured?
- **Faithfulness / Grounding:** Does the answer stay consistent with the provided context documents?
- **Perplexity:** How “surprising” is the answer to a language model?

- **Pros:**
  - Works when you **don’t have ground truth labels**.
  - Useful for checking hallucinations and style.
- **Cons:**
  - Harder to connect directly to “correctness”.
  - Often requires LLM‑based or heuristic judges.
- **Best for:**
  - Early‑stage systems or open‑ended tasks where labels are expensive.

## 5. Human Evaluation

**Idea:** Have humans rate or compare model outputs directly.

- **How it works:**
  - Annotators score answers on scales like **correctness**, **helpfulness**, **harmlessness**.
  - Or perform **pairwise comparisons**: “Is answer A or B better?”
- **Pros:**
  - Gold standard for nuanced and domain‑specific quality.
  - Can capture preferences and safety concerns better than automated metrics.
- **Cons:**
  - Expensive and slow.
  - Requires careful rubric design and annotator training.
- **Best for:**
  - Final evaluations before deployment.
  - Tuning systems to human preferences.

## 6. Task‑Specific Automated Metrics

**Idea:** Use **domain‑specific signals** rather than generic text metrics.

Examples:

- **Code generation:** Does the code compile? Do unit tests pass? (`pass@k`)
- **Math / reasoning:** Is the numeric answer correct? Does a verifier model accept the chain of thought?
- **Retrieval:** Precision@K, Recall@K, MRR for retrieved documents.
- **Classification:** Accuracy, F1, ROC‑AUC.

- **Pros:**
  - Highly interpretable and strongly tied to task success.
- **Cons:**
  - Not always available for free‑form text tasks.
- **Best for:**
  - Structured tasks where you can **run the output** or check it programmatically.

## 7. Multi‑Metric Frameworks

**Idea:** Combine multiple metrics to get a richer view of system behavior.

Examples (often in RAG‑oriented tools):

- **Context Precision / Recall:** Are retrieved documents relevant and sufficient?
- **Faithfulness:** Does the answer stay within retrieved evidence?
- **Answer Relevance:** Does the answer actually address the question?

- **Pros:**
  - More holistic: can distinguish retrieval issues from generation issues.
- **Cons:**
  - Harder to summarize into a single score.
- **Best for:**
  - Diagnosing **where** a RAG pipeline fails (retrieval vs. generation).

## Choosing an Approach

There is no single “best” evaluation method. A practical strategy is to **mix several**:

- Use **rule‑based** or **embedding‑based** metrics for fast regression tests and CI.
- Use **LLM as a judge** for semantic quality and user‑facing comparisons.
- Run **task‑specific checks** wherever you can (tests, verifiers, constraints).
- Periodically validate with **human evaluation** to catch gaps and biases.

In this sandbox:

- **LangChainEvalRunner** demonstrates **LLM as a judge**.
- **DeepEvalRunner** and **RagasRunner** illustrate **rule‑based overlap metrics**.
- **EmbeddingEvalRunner** adds a **semantic embedding similarity** baseline.

You can extend this pattern by adding new runners that implement other metrics or domain‑specific checks, while keeping the same `EvaluationInput` / `EvaluationResult` interface.
