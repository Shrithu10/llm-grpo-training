"""
Reward math utilities for RLVR.

Legacy functions (sparse_reward, binary_correct, scale_reward, clip_reward)
are preserved exactly for backward compatibility.

New helpers (numeric_closeness, weighted_sum) are thin utilities used by
verifier.py and available to callers that want to compose custom rewards.
"""


# ── Legacy helpers (original sparse interface) ─────────────────────────────────

def sparse_reward(is_correct: bool, reward_value: float = 1.0) -> float:
    """Original sparse reward: reward_value if correct, else 0."""
    return reward_value if is_correct else 0.0


def binary_correct(pred, gt) -> bool:
    """Return True if pred matches gt (string comparison)."""
    return str(pred).strip() == str(gt).strip()


def scale_reward(reward: float, scale: float = 1.0) -> float:
    """Scale a reward by a multiplier."""
    return reward * scale


def clip_reward(reward: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Clip reward to [min_val, max_val]."""
    return max(min_val, min(max_val, reward))


# ── New dense reward helpers ───────────────────────────────────────────────────

def numeric_closeness(pred_str, gt_str) -> float:
    """
    Deterministic numeric partial credit.
    Formula: max(0, 1 - |pred - gt| / max(|gt|, 1e-6))
    Returns 0.0 for non-numeric inputs.
    """
    try:
        pred = float(str(pred_str).strip().replace(',', ''))
        gt   = float(str(gt_str).strip().replace(',', ''))
    except (ValueError, TypeError):
        return 0.0
    denom = max(abs(gt), 1e-6)
    return max(0.0, 1.0 - abs(pred - gt) / denom)


def weighted_sum(components: dict, weights: dict) -> float:
    """
    Compute a weighted sum of reward components.
    Keys present in components but absent in weights contribute 0.
    """
    return sum(v * weights.get(k, 0.0) for k, v in components.items())
