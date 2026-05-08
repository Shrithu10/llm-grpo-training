# LLM GRPO Training — Self-Correction Reasoning

A lightweight, reproducible framework for training small language models to improve structured reasoning using **Group Relative Policy Optimization (GRPO)** and verifiable dense rewards.

Designed to run on a single GPU (Google Colab compatible).

---

## Overview

This project implements a three-phase RLVR (Reinforcement Learning with Verifiable Rewards) pipeline:

| Phase | Description |
|-------|-------------|
| **Phase 1** | Dense reward system with error taxonomy and reward decomposition |
| **Phase 2** | GRPO training loop with group sampling and policy gradient updates |
| **Phase 3** | Inference scaling experiments, baseline comparison, and evaluation |

---

## Key Features

- **Dense reward decomposition** — correctness, closeness, format, reasoning, and self-correction components
- **Reward hacking protection** — filler detection, auxiliary reward gating, logarithmic reasoning scaling
- **GRPO training** — no critic network; group mean serves as the variance-reducing baseline
- **Inference scaling analysis** — accuracy vs token budget curves
- **Deterministic mock model** — full experiment pipeline runs without a GPU

---

## File Structure

```
.
├── verifier.py           # Dense reward verifier + error taxonomy
├── reward_math.py        # Reward math utilities (legacy + dense helpers)
├── test_env.py           # Phase 1 test suite
│
├── sampling_utils.py     # Group sampling and prompt formatting
├── logging_utils.py      # Training step logging
├── train_grpo.py         # GRPO training loop
│
├── evaluation.py         # Inference runner, MockModel, metrics
├── experiment_runner.py  # Four experiment suite
├── plotting.py           # matplotlib visualisations
│
└── results/
    └── experiment_results.json
```

---

## Quick Start

### Requirements

```bash
pip install torch transformers
pip install matplotlib   # optional, for plots
```

### Phase 1 — Test the reward system

```bash
python test_env.py
```

### Phase 2 — GRPO training (requires GPU)

```bash
python train_grpo.py
```

Default model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`. Edit `GRPOConfig` in `train_grpo.py` to change model, group size, learning rate, etc.

### Phase 3 — Run experiments (mock mode, no GPU needed)

```bash
python experiment_runner.py
```

Pass a loaded HuggingFace model to `run_all()` to evaluate a real checkpoint.

Generate plots (requires matplotlib):

```bash
python plotting.py
```

---

## Reward Components

| Component | Weight | Description |
|-----------|--------|-------------|
| `correctness` | 1.0 | Exact-match on `<answer>` tag |
| `closeness` | 0.3 | Numeric partial credit (only when incorrect) |
| `format` | 0.2 | Presence of `<think>`, `<verify>`, `<answer>` |
| `reasoning` | ≤0.2 | Log-scaled step count; filler-penalised |
| `correction` | 0.3 | Self-correction across two `<think>` blocks |

---

## Expected Response Format

```
<think>
Step-by-step reasoning...
</think>
<verify>
Cross-check...
</verify>
<answer>96</answer>
```

---

## GRPO Algorithm

```
for each prompt:
    generate N responses          # group sampling, no grad
    compute rewards via verify()  # verifiable reward
    A_i = (r_i - mean(r)) / std(r)   # relative advantage
    loss = -mean(A_i * log π(response_i | prompt))
    backprop + grad clip + AdamW step
```

No value network required. The group mean is the baseline.

---

## Experiment Results (MockModel, 20 questions)

### Inference Scaling (GRPO)

| Token Budget | Accuracy | Avg Reward | Correction Rate |
|-------------|----------|------------|-----------------|
| 50          | 0.0%     | 0.000      | 0.0%            |
| 100         | 25.0%    | 0.702      | 5.0%            |
| 200         | 50.0%    | 0.909      | 10.0%           |
| 400         | 75.0%    | 1.116      | 15.0%           |

### Baseline Comparison (400-token budget)

| Method       | Accuracy | Avg Reward | Correction Rate |
|-------------|----------|------------|-----------------|
| SFT Baseline | 45.0%   | 0.805      | 5.0%            |
| Simple PG    | 60.0%   | 0.961      | 10.0%           |
| GRPO         | 75.0%   | 1.116      | 15.0%           |

---

## License

MIT
