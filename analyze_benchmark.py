#!/usr/bin/env python3
"""
Post-process benchmark CSV files to compute metrics on a balanced sample.

Takes 100% of positive samples and randomly samples an equal number of negatives
to create a balanced dataset for fair metric computation. Averages over multiple
random balanced samples to provide robust estimates with confidence intervals.

Usage:
    python analyze_benchmark.py <csv_file> [--seed SEED] [--threshold THRESHOLD] [--n_samples N]

Examples:
    python analyze_benchmark.py benchmark/model/CasualConv_A3_Test_20260112_131334.csv
    python analyze_benchmark.py benchmark/mm_run_20260111_1904/multimodal_perf_benchmark_20260112_133225.csv --threshold 0.45
    python analyze_benchmark.py benchmark/model/test.csv --n_samples 200
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def compute_metrics(labels: np.ndarray, preds: np.ndarray) -> dict:
    """Compute classification metrics."""
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    total = len(labels)

    return {
        "total_samples": total,
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "accuracy": accuracy_score(labels, preds) * 100,
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "false_positive_rate": (fp / (fp + tn) * 100) if (fp + tn) > 0 else 0,
        "false_negative_rate": (fn / (fn + tp) * 100) if (fn + tp) > 0 else 0,
    }


def aggregate_metrics(metrics_list: list) -> dict:
    """Aggregate metrics from multiple runs, computing mean, std, and 95% CI."""
    keys = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "false_positive_rate",
        "false_negative_rate",
    ]

    aggregated = {
        "n_samples": len(metrics_list),
        "total_samples": metrics_list[0]["total_samples"],  # Same for all
    }

    for key in keys:
        values = np.array([m[key] for m in metrics_list])
        mean = np.mean(values)
        std = np.std(values, ddof=1)  # Sample std
        # 95% CI using t-distribution approximation (for large n, ~1.96)
        ci_half = 1.96 * std / np.sqrt(len(values))
        aggregated[key] = {
            "mean": mean,
            "std": std,
            "ci_low": mean - ci_half,
            "ci_high": mean + ci_half,
            "min": np.min(values),
            "max": np.max(values),
        }

    return aggregated


def print_metrics(metrics: dict, title: str = "Metrics"):
    """Pretty print metrics."""
    print(f"\n{'=' * 60}")
    print(f" {title}")
    print("=" * 60)
    print(f"  Total Samples:       {metrics['total_samples']:,}")
    print(f"  True Positives:      {metrics['true_positives']:,}")
    print(f"  True Negatives:      {metrics['true_negatives']:,}")
    print(f"  False Positives:     {metrics['false_positives']:,}")
    print(f"  False Negatives:     {metrics['false_negatives']:,}")
    print()
    print(f"  Accuracy:            {metrics['accuracy']:.2f}%")
    print(f"  Precision:           {metrics['precision']:.3f}")
    print(f"  Recall:              {metrics['recall']:.3f}")
    print(f"  F1 Score:            {metrics['f1']:.3f}")
    print(f"  False Positive Rate: {metrics['false_positive_rate']:.2f}%")
    print(f"  False Negative Rate: {metrics['false_negative_rate']:.2f}%")
    print("=" * 60)


def print_aggregated_metrics(agg: dict, title: str = "Aggregated Metrics"):
    """Pretty print aggregated metrics with confidence intervals."""
    print(f"\n{'=' * 70}")
    print(f" {title}")
    print(
        f" (n={agg['n_samples']} balanced samples, {agg['total_samples']:,} samples each)"
    )
    print("=" * 70)
    print(f"  {'Metric':<20} {'Mean':>10} {'± Std':>10} {'95% CI':>20}")
    print(f"  {'-' * 20} {'-' * 10} {'-' * 10} {'-' * 20}")

    def fmt_row(name, m, is_pct=False):
        if is_pct:
            return (
                f"  {name:<20} {m['mean']:>9.2f}% {m['std']:>9.2f}% "
                f"[{m['ci_low']:>7.2f}%, {m['ci_high']:>7.2f}%]"
            )
        else:
            return (
                f"  {name:<20} {m['mean']:>10.3f} {m['std']:>10.3f} "
                f"[{m['ci_low']:>8.3f}, {m['ci_high']:>8.3f}]"
            )

    print(fmt_row("Accuracy", agg["accuracy"], is_pct=True))
    print(fmt_row("Precision", agg["precision"]))
    print(fmt_row("Recall", agg["recall"]))
    print(fmt_row("F1 Score", agg["f1"]))
    print(
        fmt_row("False Positive Rate", agg["false_positive_rate"], is_pct=True)
    )
    print(
        fmt_row("False Negative Rate", agg["false_negative_rate"], is_pct=True)
    )
    print("=" * 70)


def generate_markdown_report(
    csv_path: str,
    threshold: float,
    original_metrics: dict,
    balanced_metrics: dict,
    pos_count: int,
    neg_count: int,
    balanced_probs: np.ndarray,
    balanced_labels: np.ndarray,
    threshold_results: list,
    best_thresh: float,
    best_f1: float,
    n_bootstrap: int = 1,
) -> str:
    """Generate a markdown report.

    Args:
        balanced_metrics: Either a single metrics dict (n_bootstrap=1) or
                         an aggregated metrics dict from aggregate_metrics().
        n_bootstrap: Number of bootstrap samples used.
    """
    lines = []
    is_aggregated = n_bootstrap > 1

    # Header
    lines.append("# Balanced Benchmark Analysis Report")
    lines.append(f"\n**Source CSV:** `{csv_path}`")
    lines.append(f"\n**Analysis Threshold:** {threshold}")
    if is_aggregated:
        lines.append(f"\n**Bootstrap Samples:** {n_bootstrap}")

    # Original dataset summary
    lines.append("\n## Original Dataset")
    lines.append(f"\n- **Total Samples:** {pos_count + neg_count:,}")
    lines.append(
        f"- **Positives (end of turn):** {pos_count:,} ({pos_count / (pos_count + neg_count) * 100:.1f}%)"
    )
    lines.append(
        f"- **Negatives (not end of turn):** {neg_count:,} ({neg_count / (pos_count + neg_count) * 100:.1f}%)"
    )
    if pos_count > 0:
        lines.append(f"- **Imbalance Ratio:** 1:{neg_count / pos_count:.1f}")

    # Original metrics table
    lines.append("\n### Original Dataset Performance")
    lines.append("\n| Metric | Value |")
    lines.append("|--------|------:|")
    lines.append(f"| Accuracy | {original_metrics['accuracy']:.2f}% |")
    lines.append(f"| Precision | {original_metrics['precision']:.3f} |")
    lines.append(f"| Recall | {original_metrics['recall']:.3f} |")
    lines.append(f"| F1 Score | {original_metrics['f1']:.3f} |")
    lines.append(
        f"| False Positive Rate | {original_metrics['false_positive_rate']:.2f}% |"
    )
    lines.append(
        f"| False Negative Rate | {original_metrics['false_negative_rate']:.2f}% |"
    )

    # Balanced dataset
    lines.append("\n## Balanced Dataset")
    n_balanced = balanced_metrics["total_samples"]
    lines.append(
        f"\n- **Total Samples per bootstrap:** {n_balanced:,} ({n_balanced // 2:,} positives + {n_balanced // 2:,} negatives)"
    )
    lines.append(
        "- **Sampling Method:** 100% of positives, random sample of equal negatives"
    )
    if is_aggregated:
        lines.append(f"- **Bootstrap Samples:** {n_bootstrap}")

    # Balanced metrics table
    lines.append("\n### Balanced Dataset Performance")
    if is_aggregated:
        lines.append("\n| Metric | Mean | ± Std | 95% CI |")
        lines.append("|--------|-----:|------:|-------:|")

        def fmt_agg(m, is_pct=False):
            if is_pct:
                return f"{m['mean']:.2f}% | {m['std']:.2f}% | [{m['ci_low']:.2f}%, {m['ci_high']:.2f}%]"
            else:
                return f"{m['mean']:.3f} | {m['std']:.3f} | [{m['ci_low']:.3f}, {m['ci_high']:.3f}]"

        lines.append(
            f"| Accuracy | {fmt_agg(balanced_metrics['accuracy'], is_pct=True)} |"
        )
        lines.append(
            f"| Precision | {fmt_agg(balanced_metrics['precision'])} |"
        )
        lines.append(f"| Recall | {fmt_agg(balanced_metrics['recall'])} |")
        lines.append(f"| F1 Score | {fmt_agg(balanced_metrics['f1'])} |")
        lines.append(
            f"| False Positive Rate | {fmt_agg(balanced_metrics['false_positive_rate'], is_pct=True)} |"
        )
        lines.append(
            f"| False Negative Rate | {fmt_agg(balanced_metrics['false_negative_rate'], is_pct=True)} |"
        )
    else:
        lines.append("\n| Metric | Value |")
        lines.append("|--------|------:|")
        lines.append(f"| Accuracy | {balanced_metrics['accuracy']:.2f}% |")
        lines.append(f"| Precision | {balanced_metrics['precision']:.3f} |")
        lines.append(f"| Recall | {balanced_metrics['recall']:.3f} |")
        lines.append(f"| F1 Score | {balanced_metrics['f1']:.3f} |")
        lines.append(
            f"| False Positive Rate | {balanced_metrics['false_positive_rate']:.2f}% |"
        )
        lines.append(
            f"| False Negative Rate | {balanced_metrics['false_negative_rate']:.2f}% |"
        )

        # Confusion matrix only for single sample
        lines.append("\n### Confusion Matrix (Balanced)")
        lines.append("\n| | Predicted Negative | Predicted Positive |")
        lines.append("|---|---:|---:|")
        lines.append(
            f"| **Actual Negative** | {balanced_metrics['true_negatives']:,} (TN) | {balanced_metrics['false_positives']:,} (FP) |"
        )
        lines.append(
            f"| **Actual Positive** | {balanced_metrics['false_negatives']:,} (FN) | {balanced_metrics['true_positives']:,} (TP) |"
        )

    # Probability distribution
    lines.append("\n## Probability Distribution (Balanced)")
    pos_probs = balanced_probs[balanced_labels == 1]
    neg_probs = balanced_probs[balanced_labels == 0]

    lines.append("\n| Class | Mean | Std | Min | Max |")
    lines.append("|-------|-----:|----:|----:|----:|")
    lines.append(
        f"| Positive (should be high) | {pos_probs.mean():.4f} | {pos_probs.std():.4f} | {pos_probs.min():.4f} | {pos_probs.max():.4f} |"
    )
    lines.append(
        f"| Negative (should be low) | {neg_probs.mean():.4f} | {neg_probs.std():.4f} | {neg_probs.min():.4f} | {neg_probs.max():.4f} |"
    )

    # Threshold sweep
    lines.append("\n## Threshold Analysis (Balanced)")
    if is_aggregated:
        lines.append(
            f"\n**Best Threshold:** {best_thresh:.2f} (Mean F1: {best_f1:.3f})"
        )
        lines.append("\n| Threshold | Accuracy | Precision | Recall | F1 |")
        lines.append("|----------:|---------:|----------:|-------:|---:|")
        for t_result in threshold_results:
            marker = " ⭐" if t_result["threshold"] == best_thresh else ""
            lines.append(
                f"| {t_result['threshold']:.2f} | {t_result['accuracy']:.2f}% ± {t_result['accuracy_std']:.2f}% | "
                f"{t_result['precision']:.3f} ± {t_result['precision_std']:.3f} | "
                f"{t_result['recall']:.3f} ± {t_result['recall_std']:.3f} | "
                f"{t_result['f1']:.3f} ± {t_result['f1_std']:.3f}{marker} |"
            )
    else:
        lines.append(
            f"\n**Best Threshold:** {best_thresh:.2f} (F1: {best_f1:.3f})"
        )
        lines.append("\n| Threshold | Accuracy | Precision | Recall | F1 |")
        lines.append("|----------:|---------:|----------:|-------:|---:|")
        for t_result in threshold_results:
            marker = " ⭐" if t_result["threshold"] == best_thresh else ""
            lines.append(
                f"| {t_result['threshold']:.2f} | {t_result['accuracy']:.2f}% | "
                f"{t_result['precision']:.3f} | {t_result['recall']:.3f} | "
                f"{t_result['f1']:.3f}{marker} |"
            )

    return "\n".join(lines)


def analyze_benchmark(
    csv_path: str, threshold: float = 0.5, seed: int = 42, n_bootstrap: int = 1
):
    """Analyze benchmark CSV with balanced sampling.

    Args:
        csv_path: Path to the benchmark CSV file.
        threshold: Decision threshold for predictions.
        seed: Base random seed for reproducibility.
        n_bootstrap: Number of balanced samples to average over.
                     If 1, uses single sample (legacy behavior).
                     If >1, averages metrics over multiple random samples.
    """
    print(f"\n📊 Analyzing: {csv_path}")
    print(f"   Threshold: {threshold}")
    print(f"   Random seed: {seed}")
    print(f"   Bootstrap samples: {n_bootstrap}")

    # Load CSV
    df = pd.read_csv(csv_path)
    print(f"   Loaded {len(df):,} samples")

    # Check required columns
    required_cols = ["true_label", "prob"]
    for col in required_cols:
        if col not in df.columns:
            print(f"❌ Error: Missing required column '{col}'")
            print(f"   Available columns: {list(df.columns)}")
            sys.exit(1)

    # Extract data
    labels = df["true_label"].values
    probs = df["prob"].values

    # Generate predictions at specified threshold
    preds = (probs > threshold).astype(int)

    # --- Original (imbalanced) metrics ---
    pos_count = (labels == 1).sum()
    neg_count = (labels == 0).sum()
    print(f"\n📈 Original dataset:")
    print(
        f"   Positives (end of turn):     {pos_count:,} ({pos_count / len(labels) * 100:.1f}%)"
    )
    print(
        f"   Negatives (not end of turn): {neg_count:,} ({neg_count / len(labels) * 100:.1f}%)"
    )
    print(
        f"   Imbalance ratio:             1:{neg_count / pos_count:.1f}"
        if pos_count > 0
        else ""
    )

    original_metrics = compute_metrics(labels, preds)
    print_metrics(
        original_metrics, f"Original Dataset Metrics (threshold={threshold})"
    )

    # --- Balanced sampling ---
    pos_indices = np.where(labels == 1)[0]
    neg_indices = np.where(labels == 0)[0]
    n_samples = len(pos_indices)

    # Check if we have enough negatives
    if len(neg_indices) < n_samples:
        print(
            f"\n   ⚠️  Not enough negatives ({len(neg_indices)}) to match positives ({n_samples})"
        )
        print(
            f"   Using all {len(neg_indices)} negatives and sampling {len(neg_indices)} positives"
        )
        n_samples = len(neg_indices)

    if n_bootstrap == 1:
        # --- Single balanced sample (legacy behavior) ---
        print(f"\n⚖️  Creating balanced sample...")
        np.random.seed(seed)

        if len(neg_indices) >= len(pos_indices):
            sampled_pos_indices = pos_indices  # All positives
            sampled_neg_indices = np.random.choice(
                neg_indices, size=n_samples, replace=False
            )
        else:
            sampled_pos_indices = np.random.choice(
                pos_indices, size=n_samples, replace=False
            )
            sampled_neg_indices = neg_indices

        balanced_indices = np.concatenate(
            [sampled_pos_indices, sampled_neg_indices]
        )
        np.random.shuffle(balanced_indices)

        balanced_labels = labels[balanced_indices]
        balanced_preds = preds[balanced_indices]
        balanced_probs = probs[balanced_indices]

        print(
            f"   Balanced sample size: {len(balanced_indices):,} ({n_samples:,} pos + {n_samples:,} neg)"
        )

        balanced_metrics = compute_metrics(balanced_labels, balanced_preds)
        print_metrics(
            balanced_metrics,
            f"Balanced Dataset Metrics (threshold={threshold})",
        )

        # Threshold sweep on single balanced set
        best_f1 = 0
        best_thresh = 0.5
        threshold_results = []
        for thresh in np.linspace(0.1, 0.9, 17):
            t_preds = (balanced_probs > thresh).astype(int)
            t_metrics = compute_metrics(balanced_labels, t_preds)
            threshold_results.append(
                {
                    "threshold": round(thresh, 2),
                    "accuracy": t_metrics["accuracy"],
                    "precision": t_metrics["precision"],
                    "recall": t_metrics["recall"],
                    "f1": t_metrics["f1"],
                }
            )
            if t_metrics["f1"] > best_f1:
                best_f1 = t_metrics["f1"]
                best_thresh = thresh

    else:
        # --- Multiple bootstrap samples ---
        print(f"\n⚖️  Creating {n_bootstrap} balanced samples...")

        all_metrics = []
        # Store threshold-level metrics for aggregation
        threshold_metrics_per_sample = {
            round(t, 2): [] for t in np.linspace(0.1, 0.9, 17)
        }

        # We'll keep the last sample for probability distribution reporting
        last_balanced_labels = None
        last_balanced_probs = None

        for i in range(n_bootstrap):
            np.random.seed(seed + i)

            if len(neg_indices) >= len(pos_indices):
                sampled_pos_indices = pos_indices  # All positives
                sampled_neg_indices = np.random.choice(
                    neg_indices, size=n_samples, replace=False
                )
            else:
                sampled_pos_indices = np.random.choice(
                    pos_indices, size=n_samples, replace=False
                )
                sampled_neg_indices = neg_indices

            balanced_indices = np.concatenate(
                [sampled_pos_indices, sampled_neg_indices]
            )
            np.random.shuffle(balanced_indices)

            b_labels = labels[balanced_indices]
            b_preds = preds[balanced_indices]
            b_probs = probs[balanced_indices]

            # Compute metrics at specified threshold
            metrics = compute_metrics(b_labels, b_preds)
            all_metrics.append(metrics)

            # Threshold sweep
            for thresh in np.linspace(0.1, 0.9, 17):
                t_preds = (b_probs > thresh).astype(int)
                t_metrics = compute_metrics(b_labels, t_preds)
                threshold_metrics_per_sample[round(thresh, 2)].append(t_metrics)

            last_balanced_labels = b_labels
            last_balanced_probs = b_probs

        balanced_labels = last_balanced_labels
        balanced_probs = last_balanced_probs

        print(
            f"   Balanced sample size: {n_samples * 2:,} ({n_samples:,} pos + {n_samples:,} neg) x {n_bootstrap} samples"
        )

        # Aggregate metrics
        balanced_metrics = aggregate_metrics(all_metrics)
        print_aggregated_metrics(
            balanced_metrics,
            f"Balanced Dataset Metrics (threshold={threshold}, n={n_bootstrap})",
        )

        # Aggregate threshold sweep results
        best_f1 = 0
        best_thresh = 0.5
        threshold_results = []
        print(
            f"\n🎯 Threshold sweep (averaged over {n_bootstrap} balanced samples):"
        )
        print(
            f"   {'Threshold':>10} | {'Accuracy':>14} | {'Precision':>14} | {'Recall':>14} | {'F1':>14}"
        )
        print(
            f"   {'-' * 10} | {'-' * 14} | {'-' * 14} | {'-' * 14} | {'-' * 14}"
        )

        for thresh in np.linspace(0.1, 0.9, 17):
            thresh_rounded = round(thresh, 2)
            t_metrics_list = threshold_metrics_per_sample[thresh_rounded]
            t_agg = aggregate_metrics(t_metrics_list)
            threshold_results.append(
                {
                    "threshold": thresh_rounded,
                    "accuracy": t_agg["accuracy"]["mean"],
                    "accuracy_std": t_agg["accuracy"]["std"],
                    "precision": t_agg["precision"]["mean"],
                    "precision_std": t_agg["precision"]["std"],
                    "recall": t_agg["recall"]["mean"],
                    "recall_std": t_agg["recall"]["std"],
                    "f1": t_agg["f1"]["mean"],
                    "f1_std": t_agg["f1"]["std"],
                }
            )
            marker = " *" if thresh_rounded == round(threshold, 2) else ""
            print(
                f"   {thresh:>10.2f} | {t_agg['accuracy']['mean']:>6.2f}% ± {t_agg['accuracy']['std']:>4.2f}% | "
                f"{t_agg['precision']['mean']:>5.3f} ± {t_agg['precision']['std']:>5.3f} | "
                f"{t_agg['recall']['mean']:>5.3f} ± {t_agg['recall']['std']:>5.3f} | "
                f"{t_agg['f1']['mean']:>5.3f} ± {t_agg['f1']['std']:>5.3f}{marker}"
            )
            if t_agg["f1"]["mean"] > best_f1:
                best_f1 = t_agg["f1"]["mean"]
                best_thresh = thresh_rounded

        print(
            f"\n   ✅ Best threshold: {best_thresh:.2f} (Mean F1: {best_f1:.3f})"
        )

    # --- Probability distribution on balanced set ---
    print(f"\n📉 Probability Distribution (balanced set):")
    print(f"   Min:  {balanced_probs.min():.4f}")
    print(f"   Max:  {balanced_probs.max():.4f}")
    print(f"   Mean: {balanced_probs.mean():.4f}")
    print(f"   Std:  {balanced_probs.std():.4f}")

    # Breakdown by class
    pos_probs = balanced_probs[balanced_labels == 1]
    neg_probs = balanced_probs[balanced_labels == 0]
    print(f"\n   Positive samples (should be high):")
    print(f"     Mean: {pos_probs.mean():.4f}, Std: {pos_probs.std():.4f}")
    print(f"   Negative samples (should be low):")
    print(f"     Mean: {neg_probs.mean():.4f}, Std: {neg_probs.std():.4f}")

    # Print threshold sweep for single sample case
    if n_bootstrap == 1:
        print(f"\n🎯 Threshold sweep (balanced set):")
        print(
            f"   {'Threshold':>10} | {'Accuracy':>10} | {'Precision':>10} | {'Recall':>10} | {'F1':>10}"
        )
        print(
            f"   {'-' * 10} | {'-' * 10} | {'-' * 10} | {'-' * 10} | {'-' * 10}"
        )
        for t_result in threshold_results:
            marker = (
                " *" if t_result["threshold"] == round(threshold, 2) else ""
            )
            print(
                f"   {t_result['threshold']:>10.2f} | {t_result['accuracy']:>9.2f}% | "
                f"{t_result['precision']:>10.3f} | {t_result['recall']:>10.3f} | "
                f"{t_result['f1']:>10.3f}{marker}"
            )
        print(f"\n   ✅ Best threshold: {best_thresh:.2f} (F1: {best_f1:.3f})")

    # --- Save balanced CSV (only for single sample) ---
    if n_bootstrap == 1:
        balanced_indices_final = np.concatenate(
            [sampled_pos_indices, sampled_neg_indices]
        )
        balanced_df = df.iloc[balanced_indices_final].copy()
        output_path = csv_path.replace(".csv", "_balanced.csv")
        balanced_df.to_csv(output_path, index=False)
        print(f"\n💾 Balanced CSV saved to: {output_path}")

    # --- Generate and save markdown report ---
    md_output_path = csv_path.replace(".csv", "_balanced_analysis.md")
    md_report = generate_markdown_report(
        csv_path=csv_path,
        threshold=threshold,
        original_metrics=original_metrics,
        balanced_metrics=balanced_metrics,
        pos_count=pos_count,
        neg_count=neg_count,
        balanced_probs=balanced_probs,
        balanced_labels=balanced_labels,
        threshold_results=threshold_results,
        best_thresh=best_thresh,
        best_f1=best_f1,
        n_bootstrap=n_bootstrap,
    )
    with open(md_output_path, "w") as f:
        f.write(md_report)
    print(f"📄 Markdown report saved to: {md_output_path}")

    return {
        "original": original_metrics,
        "balanced": balanced_metrics,
        "best_threshold": best_thresh,
        "best_f1": best_f1,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Analyze benchmark CSV with balanced sampling"
    )
    parser.add_argument("csv_file", help="Path to benchmark CSV file")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold for predictions (default: 0.5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling (default: 42)",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=100,
        help="Number of balanced samples to average over (default: 100). "
        "Use 1 for single sample (legacy behavior).",
    )

    args = parser.parse_args()

    if not Path(args.csv_file).exists():
        print(f"❌ Error: File not found: {args.csv_file}")
        sys.exit(1)

    analyze_benchmark(
        args.csv_file,
        threshold=args.threshold,
        seed=args.seed,
        n_bootstrap=args.n_samples,
    )


if __name__ == "__main__":
    main()
