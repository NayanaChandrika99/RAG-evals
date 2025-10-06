"""
RAGAS Evaluation Script for RAG Ops Framework

This script evaluates the RAG system using RAGAS metrics and a top-tier judge model.

KEY COST DECISION: We use GPT-4 as the judge model for reliable evaluation.
Cost: ~$0.10-0.20 per run (20 questions). This is a worthwhile investment for
accurate quality assessment.
"""

import json
import sys
import argparse
from pathlib import Path

from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextRecall,
    ContextPrecision
)
from langchain_openai import ChatOpenAI
from datasets import Dataset

# Import our RAG chain
from src.rag_chain import ask_question_with_context


def load_golden_dataset(dataset_path: str = "data/golden_dataset.json"):
    """Load the golden dataset for evaluation"""
    if not Path(dataset_path).exists():
        print(f"❌ Error: Golden dataset not found at {dataset_path}")
        print("\nPlease create a golden dataset with this format:")
        print("""
[
  {
    "question": "Your question here",
    "ground_truth_answer": "The expected answer",
    "ground_truth_context": "Relevant context from document"
  }
]
        """)
        sys.exit(1)

    with open(dataset_path, "r") as f:
        return json.load(f)


def run_evaluation(
    dataset_path: str = "data/golden_dataset.json",
    output_path: str = "evaluation_report.json",
    judge_model: str = "gpt-4-turbo"
):
    """
    Run RAGAS evaluation on the golden dataset.

    Args:
        dataset_path: Path to golden dataset JSON
        output_path: Path to save evaluation results
        judge_model: Model to use for evaluation judgments
    """

    print("="*60)
    print("🔬 RAG System Evaluation")
    print("="*60)

    # Load golden dataset
    print(f"\n📥 Loading golden dataset from {dataset_path}...")
    golden_data = load_golden_dataset(dataset_path)
    print(f"✅ Loaded {len(golden_data)} test cases")

    # Generate answers using RAG chain with context tracking
    print(f"\n🤖 Generating answers for {len(golden_data)} questions...")
    eval_data = []

    for i, item in enumerate(golden_data, 1):
        question = item["question"]
        print(f"   {i}/{len(golden_data)}: {question[:60]}...")

        try:
            # Get both answer and contexts using the enhanced chain
            result = ask_question_with_context(question)

            eval_data.append({
                "question": question,
                "answer": result["answer"],
                "contexts": result["contexts"],
                "ground_truth": item["ground_truth_answer"]
            })

        except Exception as e:
            print(f"   ❌ Error on question {i}: {e}")
            # Add placeholder to maintain dataset size
            eval_data.append({
                "question": question,
                "answer": "ERROR",
                "contexts": [""],
                "ground_truth": item["ground_truth_answer"]
            })

    print(f"✅ Generated {len(eval_data)} answers")

    # Convert to RAGAS dataset format
    print(f"\n🔄 Converting to RAGAS format...")
    eval_dataset = Dataset.from_list(eval_data)

    # Use a top-tier model for reliable evaluation judgments
    print(f"\n🧠 Initializing judge model: {judge_model}")
    print(f"   (This is our strategic investment in quality assessment)")
    judge_llm = ChatOpenAI(
        model=judge_model,
        temperature=0  # Deterministic for consistent evaluation
    )

    # Run the evaluation
    print(f"\n⚡ Running RAGAS evaluation...")
    print(f"   Metrics: faithfulness, answer_relevancy, context_recall, context_precision")
    print(f"   This may take a few minutes...")

    try:
        result = evaluate(
            dataset=eval_dataset,
            metrics=[Faithfulness(), AnswerRelevancy(), ContextRecall(), ContextPrecision()],
            llm=judge_llm
        )
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Print results
    print("\n" + "="*60)
    print("📊 Evaluation Results")
    print("="*60)

    # Extract scores - handle both old (dict) and new (EvaluationResult) RAGAS formats
    def extract_score(value):
        """Extract numeric score from RAGAS result (handles list or float)"""
        if isinstance(value, list):
            return sum(value) / len(value) if value else 0.0
        return float(value)

    # Convert EvaluationResult to dict if needed
    if hasattr(result, 'to_pandas'):
        # New RAGAS format - convert to dict, only select numeric columns for metrics
        df = result.to_pandas()
        # Get only the metric columns we care about
        metric_cols = ['faithfulness', 'answer_relevancy', 'context_recall', 'context_precision']
        result_dict = df[metric_cols].mean().to_dict()
    elif hasattr(result, '__dict__'):
        # Try getting attributes
        result_dict = vars(result)
    else:
        # Assume it's already a dict
        result_dict = result

    scores = {
        "faithfulness": extract_score(result_dict.get("faithfulness", 0)),
        "answer_relevancy": extract_score(result_dict.get("answer_relevancy", 0)),
        "context_recall": extract_score(result_dict.get("context_recall", 0)),
        "context_precision": extract_score(result_dict.get("context_precision", 0))
    }

    # Print formatted results
    for metric, score in scores.items():
        # Determine status based on thresholds from CLAUDE.md
        thresholds = {
            "faithfulness": 0.80,
            "answer_relevancy": 0.80,
            "context_recall": 0.75,
            "context_precision": 0.70
        }

        status = "✅" if score >= thresholds.get(metric, 0.75) else "⚠️"
        print(f"{status} {metric:20s}: {score:.3f} (target: ≥{thresholds.get(metric, 0.75):.2f})")

    print("="*60)

    # Calculate overall pass rate
    passes = sum(1 for metric, score in scores.items()
                 if score >= thresholds.get(metric, 0.75))
    pass_rate = (passes / len(scores)) * 100

    print(f"\n📈 Overall: {passes}/{len(scores)} metrics passed ({pass_rate:.0f}%)")

    # Save detailed results
    print(f"\n💾 Saving detailed results to {output_path}...")
    output_data = {
        "scores": scores,
        "pass_rate": pass_rate,
        "num_test_cases": len(golden_data),
        "judge_model": judge_model,
        "thresholds": thresholds
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"✅ Results saved!")

    # Return exit code based on pass rate
    if pass_rate < 75:
        print(f"\n⚠️  Warning: Pass rate {pass_rate:.0f}% below 75% threshold")
        return 1

    print(f"\n✅ All quality gates passed!")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation on RAG system")
    parser.add_argument(
        "--dataset",
        default="data/golden_dataset.json",
        help="Path to golden dataset"
    )
    parser.add_argument(
        "--output",
        default="evaluation_report.json",
        help="Path to save evaluation results"
    )
    parser.add_argument(
        "--judge-model",
        default="gpt-4-turbo",
        help="Model to use for evaluation"
    )

    args = parser.parse_args()

    try:
        exit_code = run_evaluation(
            dataset_path=args.dataset,
            output_path=args.output,
            judge_model=args.judge_model
        )
        sys.exit(exit_code)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
