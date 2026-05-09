"""
Phase 3 — Evaluation utilities.

Works in two modes:
  Mock mode  (model=None) — uses MockModel; no GPU required
  Real mode  (model + tokenizer) — greedy inference on a loaded HF model

Response template contract
--------------------------
build_full_response() in sampling_utils.py prepends "<think>\\n" before
calling verify().  Every template here therefore starts MID-think-block
(after the opening tag) and must include the closing </think> plus all
subsequent structure tags.
"""

import json
import re
import statistics
import hashlib
from pathlib import Path
from typing import List, Optional

from verifier import verify
from sampling_utils import build_full_response


# ── Response templates ────────────────────────────────────────────────────────
# Text generated AFTER the prompt's "<think>\n" opener.

_T_CORRECT = (
    "The problem asks us to compute the answer.\n"
    "Step 1: Identify the operation.\n"
    "Step 2: Apply it — result is {answer}.\n"
    "Step 3: Verify by reverse-checking.\n"
    "</think>\n"
    "<verify>\nCross-check confirms: {answer}.\n</verify>\n"
    "<answer>{answer}</answer>"
)

# Two separate <think> blocks signal genuine self-correction to correction_reward()
_T_CORRECTION = (
    "First pass: I think the answer might be {wrong}.\n"
    "</think>\n"
    "<think>\nWait, I made an error. Let me redo this.\n"
    "Corrected computation gives {answer}.\n"
    "</think>\n"
    "<verify>\nRechecked: {answer} is correct. Initial estimate was off.\n</verify>\n"
    "<answer>{answer}</answer>"
)

_T_NEAR_MISS = (
    "Quick estimate: approximately {near}.\n"
    "</think>\n"
    "<verify>\nSeems close enough.\n</verify>\n"
    "<answer>{near}</answer>"
)

_T_WRONG = (
    "I compute the answer as {wrong}.\n"
    "</think>\n"
    "<verify>\nConfirmed: {wrong}.\n</verify>\n"
    "<answer>{wrong}</answer>"
)

# No structural tags at all — simulates a model that ignores the format prompt
_T_FORMAT_ERROR = "I need to think about this problem more carefully before answering."


# ── MockModel ─────────────────────────────────────────────────────────────────

class MockModel:
    """
    Deterministic synthetic model for running all experiments without a GPU.

    Three profiles simulate SFT, simple policy-gradient, and GRPO behaviour:
      accuracy_at_max    -- accuracy ceiling at maximum token budget (400 tokens)
      correction_rate    -- fraction of correct responses featuring self-correction
      format_error_rate  -- fraction of responses with missing structure tags

    Token budget scales effective accuracy via TOKEN_SCALE, reproducing the
    real inference-scaling effect: more tokens -> higher accuracy.

    Scores are rank-normalised within each dataset call so that the accuracy
    numbers exactly match the profile settings regardless of which questions
    happen to have high or low MD5 hashes.
    """

    PROFILES = {
        "sft": {
            "accuracy_at_max":   0.45,
            "correction_rate":   0.05,
            "format_error_rate": 0.15,
        },
        "simple_pg": {
            "accuracy_at_max":   0.60,
            "correction_rate":   0.10,
            "format_error_rate": 0.10,
        },
        "grpo": {
            "accuracy_at_max":   0.75,
            "correction_rate":   0.20,
            "format_error_rate": 0.05,
        },
    }

    # Fraction of max-budget accuracy achieved at each token budget
    _TOKEN_SCALE = {50: 0.00, 100: 0.30, 200: 0.65, 400: 1.00}

    def __init__(self, profile: str = "grpo"):
        if profile not in self.PROFILES:
            raise ValueError(f"Unknown profile '{profile}'. Choose from {list(self.PROFILES)}")
        p = self.PROFILES[profile]
        self._acc_max  = p["accuracy_at_max"]
        self._corr_r   = p["correction_rate"]
        self._fmt_err  = p["format_error_rate"]

    # ── internals ──────────────────────────────────────────────────────────────

    def _q_hash(self, question: str) -> float:
        """Raw deterministic float in [0, 1) from question text."""
        return int(hashlib.md5(question.encode()).hexdigest(), 16) % 10000 / 10000.0

    def rank_scores(self, questions: list[str]) -> dict[str, float]:
        """
        Return rank-normalised scores for a list of questions.
        Rank normalisation maps raw hashes to i/N so the score distribution
        is perfectly uniform — accuracy numbers match profile settings exactly.
        """
        order = sorted(questions, key=self._q_hash)
        n = len(order)
        return {q: i / n for i, q in enumerate(order)}

    def _effective_accuracy(self, max_tokens: int) -> float:
        return self._acc_max * self._TOKEN_SCALE.get(max_tokens, 1.0)

    @staticmethod
    def _near(gt: str) -> str:
        try:
            return str(int(float(gt.replace(",", ""))) + 1)
        except ValueError:
            return gt + "_approx"

    @staticmethod
    def _wrong(gt: str) -> str:
        try:
            return str(int(float(gt.replace(",", ""))) - 3)
        except ValueError:
            return "unknown"

    # ── public ────────────────────────────────────────────────────────────────

    def generate_from_score(self, score: float, gt: str, max_tokens: int) -> str:
        """
        Core generation logic given a pre-computed rank-normalised score.
        Returns generated text; build_full_response() prepends '<think>\\n'.
        """
        acc   = self._effective_accuracy(max_tokens)
        t_fmt  = self._fmt_err
        t_corr = t_fmt + acc * self._corr_r
        t_ok   = t_fmt + acc
        t_near = t_ok  + 0.12

        if max_tokens <= 50 or score < t_fmt:
            return _T_FORMAT_ERROR.format(answer=gt)
        if score < t_corr:
            return _T_CORRECTION.format(answer=gt, wrong=self._wrong(gt))
        if score < t_ok:
            return _T_CORRECT.format(answer=gt)
        if score < t_near:
            return _T_NEAR_MISS.format(near=self._near(gt))
        return _T_WRONG.format(wrong=self._wrong(gt))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _reasoning_length(response: str) -> int:
    """Total characters inside <think> and <verify> blocks."""
    blocks  = re.findall(r'<think>(.*?)</think>',   response, re.DOTALL | re.IGNORECASE)
    blocks += re.findall(r'<verify>(.*?)</verify>', response, re.DOTALL | re.IGNORECASE)
    return sum(len(b) for b in blocks)


def _has_correction(response: str) -> bool:
    """True when the response contains two or more separate <think> blocks."""
    return len(re.findall(r'<think>.*?</think>', response, re.DOTALL | re.IGNORECASE)) >= 2


# ── Core inference ─────────────────────────────────────────────────────────────

def run_inference(
    dataset: List[dict],
    max_tokens: int,
    model=None,
    tokenizer=None,
    device: str = "cpu",
    mock_profile: str = "grpo",
) -> List[dict]:
    """
    Run evaluation on a dataset at a fixed token budget.

    Parameters
    ----------
    dataset      : list of {"question": str, "answer": str}
    max_tokens   : generation budget (max new tokens)
    model        : HF CausalLM, or None → uses MockModel
    tokenizer    : HF tokenizer (required when model is not None)
    device       : "cuda" or "cpu"
    mock_profile : MockModel profile when model is None

    Returns
    -------
    List of result dicts.  Each dict contains all fields from verify() plus:
      token_count       — generation budget used
      reasoning_length  — total characters in <think>+<verify> blocks
      has_correction    — True if response has ≥2 <think> blocks
    """
    mock = MockModel(mock_profile) if model is None else None

    # Pre-compute rank-normalised scores so that accuracy matches the profile
    # settings exactly, regardless of which questions have high/low MD5 hashes.
    rank_scores: dict = {}
    if mock is not None:
        rank_scores = mock.rank_scores([item["question"] for item in dataset])

    results = []
    for item in dataset:
        question, gt = item["question"], item["answer"]

        if mock is not None:
            score = rank_scores[question]
            generated_text = mock.generate_from_score(score, gt, max_tokens)
        else:
            generated_text = _infer_real_model(model, tokenizer, question, max_tokens, device)

        full_response = build_full_response(generated_text)
        result = verify(full_response, gt=gt, question=question)
        result["token_count"]      = max_tokens
        result["reasoning_length"] = _reasoning_length(full_response)
        result["has_correction"]   = _has_correction(full_response)
        results.append(result)

    return results


def _infer_real_model(model, tokenizer, question: str, max_tokens: int, device: str) -> str:
    """Greedy single-sample inference on a real HF model."""
    import torch
    from sampling_utils import build_prompt

    inputs    = tokenizer(build_prompt(question), return_tensors="pt",
                          truncation=True, max_length=512).to(device)
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,            # greedy for reproducibility
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0, prompt_len:], skip_special_tokens=True)


# ── Metrics aggregation ───────────────────────────────────────────────────────

def compute_metrics(results: List[dict]) -> dict:
    """
    Aggregate a list of per-sample result dicts into summary statistics.

    Returns
    -------
    {
        accuracy, avg_reward, reward_variance,
        avg_reasoning_length, correction_rate, avg_token_count,
        error_distribution       : {error_type: count},
        reward_components_mean   : {component: mean_value},
        n_samples
    }
    """
    if not results:
        return {"n_samples": 0}

    rewards    = [r["reward"]     for r in results]
    is_correct = [r["is_correct"] for r in results]

    # Per-component means
    comp_keys  = list(results[0]["reward_components"].keys())
    comp_means = {
        k: statistics.mean(r["reward_components"][k] for r in results)
        for k in comp_keys
    }

    # Error type counts
    err_dist: dict = {}
    for r in results:
        et = r["error_type"]
        err_dist[et] = err_dist.get(et, 0) + 1

    return {
        "accuracy":             sum(is_correct) / len(is_correct),
        "avg_reward":           statistics.mean(rewards),
        "reward_variance":      statistics.variance(rewards) if len(rewards) > 1 else 0.0,
        "avg_reasoning_length": statistics.mean(r["reasoning_length"] for r in results),
        "correction_rate":      sum(r["has_correction"] for r in results) / len(results),
        "avg_token_count":      statistics.mean(r["token_count"] for r in results),
        "error_distribution":   err_dist,
        "reward_components_mean": comp_means,
        "n_samples":            len(results),
    }


# ── Persistence ───────────────────────────────────────────────────────────────

def save_results(data: dict, path: str) -> None:
    """Serialise experiment results to JSON, creating parent directories."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  Saved -> {out}")
