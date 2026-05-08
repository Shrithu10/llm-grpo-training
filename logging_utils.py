"""
Lightweight logging for GRPO training.
No external dependencies — plain print output only.
"""

import statistics
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class StepStats:
    step:       int
    rewards:    List[float]
    is_correct: List[bool]
    loss:       float

    @property
    def avg_reward(self) -> float:
        return statistics.mean(self.rewards) if self.rewards else 0.0

    @property
    def reward_variance(self) -> float:
        return statistics.variance(self.rewards) if len(self.rewards) > 1 else 0.0

    @property
    def accuracy(self) -> float:
        return sum(self.is_correct) / len(self.is_correct) if self.is_correct else 0.0


# SampleOutput = (reward, is_correct, response_text)
SampleOutput = Tuple[float, bool, str]


def log_step(
    stats: StepStats,
    sample_outputs: Optional[List[SampleOutput]] = None,
    verbose: bool = True,
) -> None:
    """Print a one-line training summary and optional response previews."""
    print(
        f"step {stats.step:4d} | "
        f"loss={stats.loss:+.4f} | "
        f"avg_r={stats.avg_reward:.4f} | "
        f"var={stats.reward_variance:.4f} | "
        f"acc={stats.accuracy:.0%}"
    )

    if verbose and sample_outputs:
        for i, (reward, correct, text) in enumerate(sample_outputs[:2]):
            mark    = "[ok]" if correct else "[--]"
            snippet = text[:100].replace("\n", " ")
            print(f"  [{i}] {mark} r={reward:.3f}  {snippet!r}")
    print()


def log_epoch_summary(epoch: int, all_stats: List[StepStats]) -> None:
    """Print an end-of-epoch aggregate over all steps."""
    all_rewards = [r for s in all_stats for r in s.rewards]
    all_correct = [c for s in all_stats for c in s.is_correct]

    avg_r = statistics.mean(all_rewards)
    std_r = statistics.stdev(all_rewards) if len(all_rewards) > 1 else 0.0
    acc   = sum(all_correct) / len(all_correct) if all_correct else 0.0

    print("=" * 60)
    print(f"  Epoch {epoch} | steps={len(all_stats)} | "
          f"avg_reward={avg_r:.4f} | std={std_r:.4f} | acc={acc:.1%}")
    print("=" * 60)
