"""
Dense reward verifier for RLVR self-correction training.
Backward-compatible: verify_sparse() preserves original sparse interface.
"""

import re
import math

# ── Error taxonomy ─────────────────────────────────────────────────────────────
ERROR_CORRECT   = "correct"
ERROR_PARSE     = "parse_error"
ERROR_FORMAT    = "format_error"
ERROR_NUMERIC   = "numeric_error"
ERROR_REASONING = "reasoning_error"

# ── Reward weights ─────────────────────────────────────────────────────────────
W_CORRECTNESS = 1.0
W_CLOSENESS   = 0.3
W_FORMAT      = 0.2
W_CORRECTION  = 0.3
MAX_REASONING_REWARD = 0.2   # hard cap; reasoning is already 1× weighted
GLOBAL_REWARD_CAP    = 2.0

# ── Filler detection ──────────────────────────────────────────────────────────
_FILLER_WORD_RE   = re.compile(r'(\b\w+\b)(\s+\1){4,}', re.IGNORECASE)
_ELLIPSIS_RE      = re.compile(r'\.{5,}')
_BLANK_LINES_RE   = re.compile(r'(\n\s*){6,}')


def _detect_filler(text: str) -> bool:
    """Return True if text contains reward-hacking filler patterns."""
    if _FILLER_WORD_RE.search(text):
        return True
    if _ELLIPSIS_RE.search(text):
        return True
    if _BLANK_LINES_RE.search(text):
        return True
    # Repeated-line ratio: ≥4 lines with <40% uniqueness is filler
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if len(lines) >= 4:
        unique_ratio = len(set(lines)) / len(lines)
        if unique_ratio < 0.4:
            return True
    # Very long text with extremely low vocabulary diversity
    words = text.split()
    if len(words) > 80:
        unique_ratio = len(set(w.lower() for w in words)) / len(words)
        if unique_ratio < 0.12:
            return True
    return False


# ── Tag utilities ──────────────────────────────────────────────────────────────

def _extract_tags(text: str, tag: str) -> list[str]:
    pattern = rf'<{tag}>(.*?)</{tag}>'
    return re.findall(pattern, text, re.DOTALL | re.IGNORECASE)


def _has_required_format(response: str) -> bool:
    """All three structural tags must be present and closed."""
    return (
        bool(re.search(r'<think>.*?</think>',   response, re.DOTALL | re.IGNORECASE)) and
        bool(re.search(r'<verify>.*?</verify>', response, re.DOTALL | re.IGNORECASE)) and
        bool(re.search(r'<answer>.*?</answer>', response, re.DOTALL | re.IGNORECASE))
    )


def _parse_answer(response: str) -> str | None:
    """Return stripped content of the last <answer> tag, or None."""
    hits = _extract_tags(response, 'answer')
    return hits[-1].strip() if hits else None


def _try_float(text: str | None) -> float | None:
    if text is None:
        return None
    try:
        return float(text.strip().replace(',', ''))
    except (ValueError, AttributeError):
        return None


# ── Individual reward components ───────────────────────────────────────────────

def correctness_reward(pred: str | None, gt: str) -> float:
    """Exact-match: 1.0 if correct, 0.0 otherwise. Preserves original behavior."""
    if pred is None:
        return 0.0
    return 1.0 if pred.strip() == gt.strip() else 0.0


def closeness_reward(pred: str | None, gt: str) -> float:
    """
    Numeric partial credit for near-miss answers.
    Formula: max(0, 1 - |pred - gt| / max(|gt|, 1e-6))
    Returns 0.0 for non-numeric predictions.
    """
    pred_f = _try_float(pred)
    gt_f   = _try_float(gt)
    if pred_f is None or gt_f is None:
        return 0.0
    denom = max(abs(gt_f), 1e-6)
    return max(0.0, 1.0 - abs(pred_f - gt_f) / denom)


def format_reward(response: str) -> float:
    """
    1.0 for full format compliance; partial credit per present tag (~0.33 each).
    """
    if _has_required_format(response):
        return 1.0
    score = 0.0
    if re.search(r'<think>.*?</think>',   response, re.DOTALL | re.IGNORECASE):
        score += 1 / 3
    if re.search(r'<verify>.*?</verify>', response, re.DOTALL | re.IGNORECASE):
        score += 1 / 3
    if re.search(r'<answer>.*?</answer>', response, re.DOTALL | re.IGNORECASE):
        score += 1 / 3
    return round(score, 4)


def reasoning_reward(response: str) -> float:
    """
    Reward structured multi-step reasoning.
    Gated behind full format compliance.
    Capped at MAX_REASONING_REWARD to prevent verbosity gaming.
    Filler text collapses reward to 0.
    """
    if not _has_required_format(response):
        return 0.0

    think_blocks  = _extract_tags(response, 'think')
    verify_blocks = _extract_tags(response, 'verify')
    if not think_blocks or not verify_blocks:
        return 0.0

    think_text  = '\n'.join(think_blocks)
    verify_text = '\n'.join(verify_blocks)

    if _detect_filler(think_text) or _detect_filler(verify_text):
        return 0.0

    # Count distinct non-empty lines as a proxy for reasoning steps
    think_lines  = [l.strip() for l in think_text.split('\n')  if l.strip()]
    verify_lines = [l.strip() for l in verify_text.split('\n') if l.strip()]
    steps = len(think_lines) + len(verify_lines)

    # Logarithmic scaling: grows slowly, hard-capped
    score = min(MAX_REASONING_REWARD, 0.05 * math.log1p(steps))
    return round(score, 4)


_CORRECTION_SIGNALS = frozenset([
    'wait', 'actually', 'let me reconsider', 'i made an error', 'i made a mistake',
    'correction', 'mistake', 'wrong', 'recalculate', 'let me redo', 'that was wrong',
    'let me re-check', 'that is incorrect', 'that was incorrect',
])


def correction_reward(response: str, pred: str | None, gt: str) -> float:
    """
    Reward genuine self-correction:
    - Requires ≥2 separate <think> blocks (structural signal for revision)
    - Earlier and later reasoning must differ
    - Final answer must be correct
    Returns 1.0 with an explicit correction signal, 0.5 for implicit correction,
    0.0 otherwise.
    """
    think_blocks = _extract_tags(response, 'think')
    if len(think_blocks) < 2:
        return 0.0
    if correctness_reward(pred, gt) < 1.0:
        return 0.0

    first = think_blocks[0].strip()
    last  = think_blocks[-1].strip()
    if first == last:
        return 0.0  # identical blocks — not genuine correction

    combined = (first + ' ' + last).lower()
    has_signal = any(s in combined for s in _CORRECTION_SIGNALS)
    return 1.0 if has_signal else 0.5


# ── Error taxonomy ─────────────────────────────────────────────────────────────

def classify_error(response: str, pred: str | None, gt: str) -> str:
    """Deterministic classification into the five error categories."""
    if pred is None:
        return ERROR_FORMAT if not _has_required_format(response) else ERROR_PARSE
    if pred.strip() == gt.strip():
        return ERROR_CORRECT
    if _try_float(pred) is not None and _try_float(gt) is not None:
        return ERROR_NUMERIC
    return ERROR_REASONING if _has_required_format(response) else ERROR_FORMAT


# ── Dense verifier (primary interface) ────────────────────────────────────────

def verify(response: str, gt: str, question: str = "") -> dict:
    """
    Full dense reward evaluation.

    Returns
    -------
    {
        reward           : float  — weighted composite reward
        reward_components: dict   — per-component breakdown
        error_type       : str    — one of the ERROR_* constants
        pred             : str|None
        gt               : str
        is_correct       : bool
        question         : str
    }
    """
    pred = _parse_answer(response)
    fmt_ok = _has_required_format(response)

    c_correct   = correctness_reward(pred, gt)
    # Closeness only contributes when exact match fails (no double-counting)
    c_close     = closeness_reward(pred, gt) if c_correct < 1.0 else 0.0
    c_format    = format_reward(response)
    # Auxiliary rewards gated behind valid formatting
    c_reason    = reasoning_reward(response)   if fmt_ok else 0.0
    c_correction = correction_reward(response, pred, gt) if fmt_ok else 0.0

    reward = (
        W_CORRECTNESS * c_correct
        + W_CLOSENESS  * c_close
        + W_FORMAT     * c_format
        + 1.0          * c_reason      # already capped at MAX_REASONING_REWARD
        + W_CORRECTION * c_correction
    )
    reward = round(min(reward, GLOBAL_REWARD_CAP), 4)

    return {
        "reward": reward,
        "reward_components": {
            "correctness": round(c_correct,    4),
            "closeness":   round(c_close,      4),
            "format":      round(c_format,     4),
            "reasoning":   round(c_reason,     4),
            "correction":  round(c_correction, 4),
        },
        "error_type": classify_error(response, pred, gt),
        "pred":       pred,
        "gt":         gt,
        "is_correct": c_correct == 1.0,
        "question":   question,
    }


# ── Backward-compatible sparse verifier ───────────────────────────────────────

def verify_sparse(response: str, gt: str) -> float:
    """
    Legacy interface: exact-match only. Returns 1.0 or 0.0.
    Behaviour is identical to the original sparse reward system.
    """
    pred = _parse_answer(response)
    return 1.0 if (pred is not None and pred.strip() == gt.strip()) else 0.0
