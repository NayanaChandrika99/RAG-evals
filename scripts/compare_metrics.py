"""
Metrics Comparison Script for CI/CD Quality Gates

This script compares evaluation metrics between a baseline and current run.
It's designed to be used in CI/CD pipelines to prevent regressions.

DEFENSIVE CODING: Uses .get() with defaults to handle missing metrics gracefully,
preventing pipeline crashes from simple key errors.
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Dict


def load_report(path: str) -> Dict:
    """Load evaluation report from JSON file"""
    if not Path(path).exists():
        print(f"❌ Error: Report not found at {path}")
        sys.exit(1)

    with open(path, "r") as f:
        return json.load(f)


def compare_metrics(
    baseline_path: str,
    current_path: str,
    threshold: float = 0.05
) -> int:
    """
    Compare evaluation metrics between baseline and current run.

    Args:
        baseline_path: Path to baseline evaluation report
        current_path: Path to current evaluation report
        threshold: Maximum allowed degradation (default: 5%)

    Returns:
        int: 0 if all metrics passed, 1 if any regressions detected
    """

    print("="*60)
    print("🔍 Comparing Evaluation Metrics")
    print("="*60)

    # Load reports
    print(f"\n📥 Loading baseline from: {baseline_path}")
    baseline = load_report(baseline_path)

    print(f"📥 Loading current from: {current_path}")
    current = load_report(current_path)

    # Extract scores (defensive coding: use .get() with defaults)
    baseline_scores = baseline.get("scores", {})
    current_scores = current.get("scores", {})

    # Metrics to compare (updated for RAGAS latest API)
    metrics = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]

    print(f"\n📊 Metric Comparison (threshold: {threshold*100:.0f}% degradation)")
    print("-"*60)

    failed = False
    results = []

    for metric in metrics:
        # Use .get() with default value to handle missing metrics gracefully
        # This prevents pipeline crashes if a metric is missing from either report
        baseline_score = baseline_scores.get(metric, 0.0)
        current_score = current_scores.get(metric, 0.0)

        # Calculate difference
        diff = current_score - baseline_score
        degradation = baseline_score - current_score

        # Check if regression occurred
        if degradation > threshold:
            status = "❌ REGRESSION"
            failed = True
        elif current_score < baseline_score:
            status = "⚠️  DEGRADED"
        elif current_score > baseline_score:
            status = "✅ IMPROVED"
        else:
            status = "✅ MAINTAINED"

        # Print result
        print(f"{status:15s} {metric:20s}: {current_score:.3f} (was {baseline_score:.3f}, Δ {diff:+.3f})")

        results.append({
            "metric": metric,
            "baseline": baseline_score,
            "current": current_score,
            "diff": diff,
            "passed": degradation <= threshold
        })

    print("-"*60)

    # Summary
    passes = sum(1 for r in results if r["passed"])
    total = len(results)

    print(f"\n📈 Summary: {passes}/{total} metrics passed")

    # Print overall result
    if failed:
        print("\n❌ QUALITY GATE FAILED: One or more metrics regressed beyond threshold")
        print(f"   Maximum allowed degradation: {threshold*100:.0f}%")
        print("\n💡 Actions:")
        print("   1. Review recent changes that might affect quality")
        print("   2. Check if test dataset has changed")
        print("   3. Validate evaluation LLM connectivity")
        print("   4. Compare full evaluation reports for details")
        return 1
    else:
        print("\n✅ QUALITY GATE PASSED: All metrics within acceptable range")
        return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare evaluation metrics for CI/CD quality gates"
    )
    parser.add_argument(
        "baseline",
        help="Path to baseline evaluation report"
    )
    parser.add_argument(
        "current",
        help="Path to current evaluation report"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="Maximum allowed degradation (default: 0.05 = 5%%)"
    )

    args = parser.parse_args()

    try:
        exit_code = compare_metrics(
            baseline_path=args.baseline,
            current_path=args.current,
            threshold=args.threshold
        )
        sys.exit(exit_code)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
