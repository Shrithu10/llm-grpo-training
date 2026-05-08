"""
Dense reward test suite.
Covers: exact match, near-miss numeric, malformed format,
self-correction, verbose exploit, and legacy backward-compat checks.
"""

from verifier import verify, verify_sparse
from reward_math import sparse_reward, numeric_closeness

_SEP  = "-" * 60
_WIDE = "=" * 60


def _print_result(label: str, result: dict) -> None:
    print(f"\n{_WIDE}")
    print(f"TEST : {label}")
    print(_SEP)
    print(f"  Question   : {result['question']}")
    print(f"  Ground Truth: {result['gt']!r}")
    print(f"  Prediction  : {result['pred']!r}")
    print(f"  Is Correct  : {result['is_correct']}")
    print(f"  Error Type  : {result['error_type']}")
    print(f"  Dense Reward: {result['reward']}")
    print(f"  Components  :")
    for k, v in result["reward_components"].items():
        bar = "#" * int(v * 20)
        print(f"    {k:12s}: {v:.4f}  {bar}")


# ── Test 1: Exact correct answer ───────────────────────────────────────────────
R_CORRECT = """
<think>
The problem asks for 12 * 8.
Step 1: 12 * 8 = 96.
Step 2: Double-check: 8 * 10 = 80, 8 * 2 = 16, total = 96.
</think>
<verify>
Confirmed: 12 * 8 = 96.
All steps are consistent.
</verify>
<answer>96</answer>
"""

# ── Test 2: Near-miss numeric answer ──────────────────────────────────────────
R_NEAR_MISS = """
<think>
12 times 8 should be around 95, I think.
Let me approximate: 10 * 8 = 80, then 2 * 8 = 15. Total ≈ 95.
</think>
<verify>
Close enough based on my estimate.
</verify>
<answer>95</answer>
"""

# ── Test 3: Malformed formatting ──────────────────────────────────────────────
R_BAD_FORMAT = """
The answer is 96. I calculated it by doing 12 times 8.
No structured reasoning here.
"""

# ── Test 4: Self-correction example ───────────────────────────────────────────
R_SELF_CORRECT = """
<think>
The problem asks for 12 * 8.
First attempt: 12 * 4 = 48, double it → 12 * 8 = 84.
</think>
<think>
Wait, I made an error. Let me redo this.
12 * 8: break it as (10 + 2) * 8 = 80 + 16 = 96.
</think>
<verify>
Confirming: 12 * 8 = 96. The second calculation is correct.
The first block contained an arithmetic mistake (48 * 2 = 96, not 84).
</verify>
<answer>96</answer>
"""

# ── Test 5: Verbose reasoning exploit attempt ─────────────────────────────────
R_VERBOSE_EXPLOIT = """
<think>
The answer is 96 the answer is 96 the answer is 96 the answer is 96.
The answer is 96 the answer is 96 the answer is 96 the answer is 96.
The answer is 96 the answer is 96 the answer is 96 the answer is 96.
The answer is 96 the answer is 96 the answer is 96 the answer is 96.
The answer is 96 the answer is 96 the answer is 96 the answer is 96.
</think>
<verify>
96 96 96 96 96 96 96 96 96 96 96 96 96 96 96 96 96 96 96 96.
</verify>
<answer>96</answer>
"""

CASES = [
    ("Exact Correct Answer",          R_CORRECT,        "96", "What is 12 * 8?"),
    ("Near Miss Numeric Answer",       R_NEAR_MISS,      "96", "What is 12 * 8?"),
    ("Malformed Formatting",           R_BAD_FORMAT,     "96", "What is 12 * 8?"),
    ("Self-Correction Example",        R_SELF_CORRECT,   "96", "What is 12 * 8?"),
    ("Verbose Reasoning Exploit",      R_VERBOSE_EXPLOIT,"96", "What is 12 * 8?"),
]

print(f"\n{'#' * 60}")
print("  DENSE REWARD TEST SUITE")
print(f"{'#' * 60}")

for label, response, gt, question in CASES:
    result = verify(response, gt=gt, question=question)
    _print_result(label, result)

# ── Backward-compat checks ─────────────────────────────────────────────────────
print(f"\n{_WIDE}")
print("BACKWARD COMPATIBILITY CHECKS")
print(_SEP)

print(f"  verify_sparse (correct)   : {verify_sparse(R_CORRECT,        '96')}")
print(f"  verify_sparse (near miss) : {verify_sparse(R_NEAR_MISS,      '96')}")
print(f"  verify_sparse (bad format): {verify_sparse(R_BAD_FORMAT,     '96')}")
print(f"  verify_sparse (self-corr) : {verify_sparse(R_SELF_CORRECT,   '96')}")
print()
print(f"  sparse_reward(True)       : {sparse_reward(True)}")
print(f"  sparse_reward(False)      : {sparse_reward(False)}")
print(f"  sparse_reward(True, 2.0)  : {sparse_reward(True, 2.0)}")
print()
print(f"  numeric_closeness(95, 96) : {numeric_closeness('95', '96'):.4f}")
print(f"  numeric_closeness(90, 96) : {numeric_closeness('90', '96'):.4f}")
print(f"  numeric_closeness(0,  96) : {numeric_closeness('0',  '96'):.4f}")
print(f"  numeric_closeness('a',96) : {numeric_closeness('a',  '96'):.4f}")
print(_WIDE)
