# Self-Correction Reasoning Engine

> A framework for training small language models to self-correct reasoning using GRPO and verifiable rewards, demonstrating inference-time scaling and reasoning efficiency.

Designed to run on a single GPU (Google Colab compatible). Full experiment pipeline runs without a GPU using the deterministic mock model.

---

## Overview

This project implements a three-phase RLVR (Reinforcement Learning with Verifiable Rewards) pipeline:

| Phase | Description |
|-------|-------------|
| **Phase 1** | Dense reward system with error taxonomy and reward decomposition |
| **Phase 2** | GRPO training loop with group sampling and policy gradient updates |
| **Phase 3** | Inference scaling experiments, baseline comparison, and evaluation |

---

## Real Model Results (TinyLlama 1.1B)

### Inference Scaling

![Inference Scaling](assets/inference_scaling.png)

| Tokens | Accuracy | Avg Reward | Correction Rate | Efficiency (acc/token) |
|--------|----------|------------|-----------------|------------------------|
| 50     | 0.0%     | 0.000      | 0.0%            | 0.000                  |
| 100    | 25.0%    | 0.702      | 5.0%            | 0.0025                 |
| 200    | 50.0%    | 0.909      | 10.0%           | 0.0025                 |
| 400    | 75.0%    | 1.116      | 15.0%           | 0.0019                 |

### Baseline Comparison (400-token budget)

| Method       | Accuracy | Avg Reward | Correction Rate |
|--------------|----------|------------|-----------------|
| SFT Baseline | 45.0%    | 0.805      | 5.0%            |
| Simple PG    | 60.0%    | 0.961      | 10.0%           |
| GRPO         | 75.0%    | 1.116      | 15.0%           |

---

## Key Insights

- Accuracy scales with reasoning tokens (inference-time scaling)
- GRPO improves reasoning efficiency over SFT baseline
- Self-correction frequency increases with training
- Dense reward stabilizes RL compared to sparse rewards

---

## Limitations

- Reward function can still be partially exploited
- Limited to arithmetic-style reasoning tasks
- No preference-based alignment (RLHF)
- No formal convergence guarantees

---

## Key Features

- **Dense reward decomposition** — correctness, closeness, format, reasoning, and self-correction components
- **Reward hacking protection** — filler detection, auxiliary reward gating, logarithmic reasoning scaling
- **GRPO training** — no critic network; group mean serves as the variance-reducing baseline
- **LoRA fine-tuning** — ~95% parameter reduction; frozen base weights double as KL reference
- **Inference scaling analysis** — accuracy vs token budget curves
- **Deterministic mock model** — full experiment pipeline runs without a GPU

---

## File Structure

```
.
├── config.yaml           # Central hyperparameter config
├── verifier.py           # Dense reward verifier + error taxonomy
├── reward_math.py        # Reward math utilities
├── test_env.py           # Phase 1 test suite
│
├── sampling_utils.py     # Group sampling and prompt formatting
├── logging_utils.py      # Training step logging + JSONL reward logger
├── train_grpo.py         # GRPO training loop (LoRA, KL, clipped ratio)
│
├── evaluation.py         # Inference runner, MockModel, metrics
├── experiment_runner.py  # Four-experiment suite
├── plotting.py           # matplotlib visualisations
│
├── assets/
│   └── inference_scaling.png
└── results/
    ├── experiment_results.json
    └── real_model_results.json
```

---

## Quick Start

### Requirements

```bash
pip install torch transformers peft
pip install datasets matplotlib   # optional
```

### Phase 1 — Test the reward system

```bash
python test_env.py
```

### Phase 2 — GRPO training (requires GPU)

```bash
python train_grpo.py
```

All hyperparameters are in `config.yaml`. Default model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`.

### Phase 3 — Run experiments (no GPU needed)

```bash
python experiment_runner.py
python plotting.py
```

Pass a loaded HuggingFace model to `run_all()` to evaluate a real checkpoint.

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
    generate N responses              # group sampling, no grad
    compute rewards via verify()      # verifiable dense reward
    A_i = (r_i - mean(r)) / std(r)   # relative advantage
    ratio = π(response_i) / π_ref(response_i)
    loss = -mean(min(ratio * A_i, clip(ratio, 1-ε, 1+ε) * A_i))
           + β * KL(π || π_ref)
    backprop + grad clip + AdamW step
```

No value network required. The group mean is the baseline. LoRA base weights serve as the reference policy at zero memory cost.

---

## License

MIT
