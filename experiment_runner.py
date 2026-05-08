"""
Phase 3 — Experiment runner.

Executes all four experiments and prints formatted result tables.
Does NOT retrain the model; all runs use fixed weights (or MockModel).

Usage
-----
  python experiment_runner.py              # mock mode, no GPU needed
  python experiment_runner.py --real       # real model (edit MODEL_NAME below)
"""

from evaluation import run_inference, compute_metrics, save_results

# ── Dataset ───────────────────────────────────────────────────────────────────

EVAL_DATASET = [
    {"question": "What is 12 * 8?",   "answer": "96"},
    {"question": "What is 15 + 27?",  "answer": "42"},
    {"question": "What is 100 - 37?", "answer": "63"},
    {"question": "What is 7 * 9?",    "answer": "63"},
    {"question": "What is 144 / 12?", "answer": "12"},
    {"question": "What is 25 * 4?",   "answer": "100"},
    {"question": "What is 81 / 9?",   "answer": "9"},
    {"question": "What is 13 + 19?",  "answer": "32"},
    {"question": "What is 56 / 7?",   "answer": "8"},
    {"question": "What is 11 * 11?",  "answer": "121"},
    {"question": "What is 200 - 75?", "answer": "125"},
    {"question": "What is 6 * 7?",    "answer": "42"},
    {"question": "What is 48 / 6?",   "answer": "8"},
    {"question": "What is 33 + 44?",  "answer": "77"},
    {"question": "What is 9 * 9?",    "answer": "81"},
    {"question": "What is 120 / 8?",  "answer": "15"},
    {"question": "What is 17 + 28?",  "answer": "45"},
    {"question": "What is 5 * 13?",   "answer": "65"},
    {"question": "What is 72 / 8?",   "answer": "9"},
    {"question": "What is 14 * 6?",   "answer": "84"},
]

TOKEN_BUDGETS  = [50, 100, 200, 400]
METHODS        = ["sft", "simple_pg", "grpo"]
METHOD_LABELS  = {"sft": "SFT Baseline", "simple_pg": "Simple PG", "grpo": "GRPO"}


# ── Table printer ─────────────────────────────────────────────────────────────

def _table(headers: list, rows: list) -> None:
    col_widths = [
        max(len(str(h)), max((len(str(row[i])) for row in rows), default=0))
        for i, h in enumerate(headers)
    ]
    sep = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
    fmt = lambda cells: "| " + " | ".join(str(c).ljust(w) for c, w in zip(cells, col_widths)) + " |"

    print(sep)
    print(fmt(headers))
    print(sep)
    for row in rows:
        print(fmt(row))
    print(sep)


_pct = lambda v: f"{v:.1%}"
_f3  = lambda v: f"{v:.4f}"
_i   = lambda v: f"{v:.0f}"


# ── Experiment 1: Inference Scaling ───────────────────────────────────────────

def run_inference_scaling(model=None, tokenizer=None, device="cpu") -> dict:
    """
    Vary max_tokens in TOKEN_BUDGETS; measure accuracy and reward (GRPO model).

    Why more tokens → better accuracy
    -----------------------------------
    At small budgets (50 tokens) the model can't emit all three required
    structure tags before hitting the limit — every response is a format_error.
    As the budget grows, the model can (1) complete the format, (2) verify its
    intermediate steps, and (3) detect and correct early mistakes.
    Accuracy rises steeply at first then flattens as "easy" errors are already
    fixed at moderate budgets — the classic diminishing-returns curve.
    """
    print("\n" + "=" * 65)
    print("  Experiment 1: Inference Scaling  (GRPO profile, all budgets)")
    print("=" * 65)

    by_budget: dict = {}
    for budget in TOKEN_BUDGETS:
        results = run_inference(
            EVAL_DATASET, budget,
            model=model, tokenizer=tokenizer, device=device,
            mock_profile="grpo",
        )
        metrics = compute_metrics(results)
        by_budget[budget] = {"metrics": metrics, "results": results}

    _table(
        ["Token Budget", "Accuracy", "Avg Reward", "Correction Rate", "Avg Reasoning Len"],
        [
            [
                budget,
                _pct(by_budget[budget]["metrics"]["accuracy"]),
                _f3(by_budget[budget]["metrics"]["avg_reward"]),
                _pct(by_budget[budget]["metrics"]["correction_rate"]),
                _i( by_budget[budget]["metrics"]["avg_reasoning_length"]),
            ]
            for budget in TOKEN_BUDGETS
        ],
    )
    return by_budget


# ── Experiment 2: Baseline Comparison ────────────────────────────────────────

def run_baseline_comparison(model=None, tokenizer=None, device="cpu") -> dict:
    """
    Compare SFT / Simple PG / GRPO at maximum token budget (400 tokens).

    How GRPO changes reasoning behaviour
    --------------------------------------
    SFT produces correct answers but rarely self-corrects — it only learned
    what the demonstrations showed.  Simple PG improves accuracy but training
    is noisy without a stable baseline.  GRPO uses the group mean as a
    variance-reducing baseline, which means more consistent gradient updates
    and higher tolerance for exploration — leading to more correction attempts
    and ultimately higher accuracy.
    """
    print("\n" + "=" * 65)
    print("  Experiment 2: Baseline Comparison  (400-token budget)")
    print("=" * 65)

    by_method: dict = {}
    for method in METHODS:
        results = run_inference(
            EVAL_DATASET, max_tokens=400,
            model=model, tokenizer=tokenizer, device=device,
            mock_profile=method,
        )
        metrics = compute_metrics(results)
        by_method[method] = {"metrics": metrics, "results": results}

    _table(
        ["Method", "Accuracy", "Avg Reward", "Reward Variance", "Correction Rate"],
        [
            [
                METHOD_LABELS[m],
                _pct(by_method[m]["metrics"]["accuracy"]),
                _f3(by_method[m]["metrics"]["avg_reward"]),
                _f3(by_method[m]["metrics"]["reward_variance"]),
                _pct(by_method[m]["metrics"]["correction_rate"]),
            ]
            for m in METHODS
        ],
    )
    return by_method


# ── Experiment 3: Reasoning Quality ──────────────────────────────────────────

def run_reasoning_quality(by_method: dict) -> dict:
    """Analyse reasoning structure and per-component reward for each method."""
    print("\n" + "=" * 65)
    print("  Experiment 3: Reasoning Quality Analysis")
    print("=" * 65)

    quality: dict = {}
    for method in METHODS:
        m  = by_method[method]["metrics"]
        rc = m["reward_components_mean"]
        quality[method] = {
            "avg_reasoning_length": m["avg_reasoning_length"],
            "correction_rate":      m["correction_rate"],
            "format_score":         rc.get("format",      0.0),
            "reasoning_score":      rc.get("reasoning",   0.0),
            "correctness_score":    rc.get("correctness", 0.0),
        }

    _table(
        ["Method", "Avg Reasoning Len", "Correction Rate",
         "Format Score", "Reasoning Score", "Correctness Score"],
        [
            [
                METHOD_LABELS[m],
                _i( quality[m]["avg_reasoning_length"]),
                _pct(quality[m]["correction_rate"]),
                _f3(quality[m]["format_score"]),
                _f3(quality[m]["reasoning_score"]),
                _f3(quality[m]["correctness_score"]),
            ]
            for m in METHODS
        ],
    )
    return quality


# ── Experiment 4: Failure Analysis ───────────────────────────────────────────

def run_failure_analysis(by_method: dict) -> dict:
    """Log incorrect predictions and error type distribution per method."""
    print("\n" + "=" * 65)
    print("  Experiment 4: Failure Analysis")
    print("=" * 65)

    analysis: dict = {}
    all_error_types: set = set()

    for method in METHODS:
        results  = by_method[method]["results"]
        failures = [r for r in results if not r["is_correct"]]
        err_dist = by_method[method]["metrics"]["error_distribution"]
        all_error_types.update(err_dist.keys())

        analysis[method] = {
            "n_failures":        len(failures),
            "error_distribution": err_dist,
            "failure_examples":  [
                {"question": r["question"], "pred": r["pred"],
                 "gt": r["gt"], "error": r["error_type"]}
                for r in failures[:3]
            ],
        }

        print(f"\n  {METHOD_LABELS[method]}  --  {len(failures)}/{len(results)} failures")
        print(f"  Error breakdown: {err_dist}")
        for ex in analysis[method]["failure_examples"]:
            print(f"    [{ex['error']:15s}]  Q: {ex['question']:<28}  "
                  f"pred={str(ex['pred'])!r:>8}  gt={ex['gt']!r}")

    # Error distribution summary table
    sorted_errors = sorted(all_error_types)
    print()
    _table(
        ["Method"] + sorted_errors,
        [
            [METHOD_LABELS[m]] + [
                analysis[m]["error_distribution"].get(et, 0)
                for et in sorted_errors
            ]
            for m in METHODS
        ],
    )
    return analysis


# ── Master runner ─────────────────────────────────────────────────────────────

def run_all(
    model=None,
    tokenizer=None,
    device: str = "cpu",
    save_dir: str = "results",
) -> dict:
    """
    Run all four experiments, print result tables, and save JSON.

    Pass model + tokenizer to use a real HF model.
    Leave both as None to run in deterministic mock mode (no GPU required).
    """
    print("\n" + "#" * 65)
    print("  Phase 3: Inference Scaling and Evaluation")
    print(f"  Mode    : {'REAL MODEL' if model else 'MockModel (deterministic)'}")
    print(f"  Dataset : {len(EVAL_DATASET)} questions")
    print("#" * 65)

    scaling_results = run_inference_scaling(model, tokenizer, device)
    method_results  = run_baseline_comparison(model, tokenizer, device)
    quality_results = run_reasoning_quality(method_results)
    failure_results = run_failure_analysis(method_results)

    all_results = {
        "inference_scaling":   {b: scaling_results[b]["metrics"] for b in TOKEN_BUDGETS},
        "baseline_comparison": {m: method_results[m]["metrics"]  for m in METHODS},
        "reasoning_quality":   quality_results,
        "failure_analysis": {
            m: {k: v for k, v in failure_results[m].items() if k != "failure_examples"}
            for m in METHODS
        },
    }
    save_results(all_results, f"{save_dir}/experiment_results.json")
    return all_results


if __name__ == "__main__":
    run_all()
