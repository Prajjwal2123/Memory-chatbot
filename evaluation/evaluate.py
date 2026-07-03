"""
Step 7: Evaluation framework.

Evaluates the RAG pipeline on a held-out question set using RAGAS metrics:
- context_relevance / context_precision: is retrieved context relevant?
- faithfulness: is the answer grounded in the retrieved context (no hallucination)?
- answer_correctness: does the answer match the ground truth?

Run from the project root:
    python evaluation/evaluate.py
"""
import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_correctness, context_precision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from rag.rag_pipeline import generate_rag_answer
from models import get_llm, get_embeddings


def load_test_set(path: str = None) -> list[dict]:
    path = path or os.path.join(os.path.dirname(__file__), "test_questions.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_eval_dataset(test_questions: list[dict]) -> Dataset:
    rows = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    for item in test_questions:
        result = generate_rag_answer(item["question"])
        rows["question"].append(item["question"])
        rows["answer"].append(result["answer"])
        rows["contexts"].append([result["context"]] if result["context"] else [""])
        rows["ground_truth"].append(item["ground_truth"])
    return Dataset.from_dict(rows)


def main():
    test_questions = load_test_set()
    print(f"Running evaluation on {len(test_questions)} questions...")

    dataset = build_eval_dataset(test_questions)

    judge_llm = LangchainLLMWrapper(get_llm(temperature=0))
    judge_embeddings = LangchainEmbeddingsWrapper(get_embeddings())

    results = evaluate(
        dataset,
        metrics=[faithfulness, answer_correctness, context_precision],
        llm=judge_llm,
        embeddings=judge_embeddings,
    )

    print("\n=== Evaluation Results ===")
    print(results)

    df = results.to_pandas()
    out_path = os.path.join(os.path.dirname(__file__), "eval_results.csv")
    df.to_csv(out_path, index=False)
    print(f"\nDetailed per-question results saved to {out_path}")


if __name__ == "__main__":
    main()
