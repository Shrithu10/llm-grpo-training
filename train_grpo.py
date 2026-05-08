"""
Phase 2: GRPO Training Loop
===========================
Trains a small causal LM to produce structured reasoning using
Group Relative Policy Optimization (GRPO).

Algorithm (one step):
  1. Generate N responses for a prompt   (no grad)
  2. Score each response via verifier    (no grad)
  3. Compute relative advantages: A_i = r_i - mean(r)  [normalised]
  4. Re-run forward pass to get log-probs               (grad)
  5. Loss = -mean(A_i * log π(response_i | prompt))
  6. Backprop + gradient clip + optimiser step

Why GRPO instead of PPO:
  No critic / value network is needed.  The group mean reward serves as a
  self-contained baseline.  This halves memory and removes value-network
  bias.  It works well when rewards are verifiable (exact or near-exact).

Run:
  python train_grpo.py
  # or inside Colab after `pip install transformers torch`
"""

import statistics
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from verifier import verify
from sampling_utils import build_prompt, build_full_response, generate_group
from logging_utils import StepStats, log_step, log_epoch_summary


# ── Config ─────────────────────────────────────────────────────────────────────

@dataclass
class GRPOConfig:
    model_name:          str   = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    group_size:          int   = 4       # N completions per prompt
    max_new_tokens:      int   = 256
    temperature:         float = 0.9     # >0 required for exploration
    top_k:               int   = 50
    learning_rate:       float = 5e-6    # keep small; RL updates can be noisy
    grad_clip:           float = 1.0
    num_epochs:          int   = 3
    log_every:           int   = 1       # steps between console prints
    normalize_advantages:bool  = True    # divide by std; stabilises training
    device:              str   = "cuda" if torch.cuda.is_available() else "cpu"


# ── Demo dataset ───────────────────────────────────────────────────────────────
# Replace with a real dataset (e.g. GSM8K) for production use.

DEMO_QUESTIONS: List[dict] = [
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
]


# ── Model loading ──────────────────────────────────────────────────────────────

def load_model(model_name: str, device: str):
    """Load model and tokenizer. Use bfloat16 on GPU for memory efficiency."""
    print(f"Loading {model_name} …")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map=device,
    )
    model.train()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded. Parameters: {n_params:,}  Device: {device}")
    return model, tokenizer


# ── Log-probability computation ────────────────────────────────────────────────

def get_response_log_probs(
    model,
    tokenizer,
    prompt_text: str,
    response_text: str,
    device: str,
) -> torch.Tensor:
    """
    Compute the sum of log π(token | context) over all generated tokens.

    We tokenise prompt and response separately to avoid cross-boundary
    tokenisation artefacts, then concatenate and do a single forward pass.
    Gradients flow only through this forward pass — the generation step
    (which produced response_text) was done under torch.no_grad().

    Shape: scalar tensor with grad.
    """
    # Tokenise separately: prompt keeps BOS; response suppresses it.
    prompt_ids   = tokenizer(
        prompt_text,   return_tensors="pt", add_special_tokens=True
    ).input_ids.to(device)

    response_ids = tokenizer(
        response_text, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(device)

    # Guard: empty generation (e.g. model hit max_length immediately)
    if response_ids.shape[1] == 0:
        return torch.zeros(1, device=device, requires_grad=True).squeeze()

    full_ids      = torch.cat([prompt_ids, response_ids], dim=1)  # [1, L]
    attention_mask = torch.ones_like(full_ids)

    outputs = model(full_ids, attention_mask=attention_mask)
    logits  = outputs.logits   # [1, L, vocab]

    prompt_len = prompt_ids.shape[1]
    resp_len   = response_ids.shape[1]

    # logits[0, i] → distribution over token at position i+1
    # Response tokens occupy positions [prompt_len : prompt_len+resp_len]
    # Their predicting logits are at    [prompt_len-1 : prompt_len+resp_len-1]
    pred_logits = logits[0, prompt_len - 1 : prompt_len + resp_len - 1, :]  # [R, V]
    log_probs   = F.log_softmax(pred_logits, dim=-1)                          # [R, V]

    token_log_probs = log_probs.gather(
        dim=1, index=response_ids[0].unsqueeze(1)
    ).squeeze(1)   # [R]

    return token_log_probs.sum()   # scalar


# ── Advantage computation ──────────────────────────────────────────────────────

def compute_advantages(rewards: List[float], normalize: bool = True) -> List[float]:
    """
    GRPO baseline: group mean.
      A_i = r_i - mean(r)

    Optional normalisation by standard deviation makes the effective step
    size independent of reward scale, which stabilises early training.

    Effect of group size N:
      Larger N → lower-variance baseline → tighter advantage estimates →
      more stable updates, but N× more generation cost per step.
      N=4 is a practical default; N=8 helps on hard tasks.
    """
    mean_r = statistics.mean(rewards)
    advantages = [r - mean_r for r in rewards]

    if normalize and len(rewards) > 1:
        std_r = statistics.stdev(rewards)
        if std_r > 1e-6:
            advantages = [a / (std_r + 1e-8) for a in advantages]

    return advantages


# ── Single training step ───────────────────────────────────────────────────────

def grpo_train_step(
    model,
    tokenizer,
    optimizer: torch.optim.Optimizer,
    question:  str,
    gt_answer: str,
    config:    GRPOConfig,
) -> tuple[StepStats, List[str], List[dict]]:
    """
    One GRPO step for a single question.

    Returns (StepStats, responses, eval_results) so the caller can log
    raw outputs alongside numeric metrics.
    """
    prompt_text = build_prompt(question)

    # ── 1. Group sampling (no grad) ───────────────────────────────────────────
    responses, _ = generate_group(
        model, tokenizer, prompt_text,
        N=config.group_size,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        top_k=config.top_k,
        device=config.device,
    )

    # ── 2. Reward computation (no grad) ───────────────────────────────────────
    rewards:        List[float] = []
    is_correct:     List[bool]  = []
    eval_results:   List[dict]  = []

    for resp_text in responses:
        full_response = build_full_response(resp_text)
        result = verify(full_response, gt=gt_answer, question=question)
        rewards.append(result["reward"])
        is_correct.append(result["is_correct"])
        eval_results.append(result)

    # Skip update when all rewards are identical — zero advantage, zero gradient.
    # This is common early in training when the model produces uniform outputs.
    if max(rewards) - min(rewards) < 1e-6:
        return (
            StepStats(step=0, rewards=rewards, is_correct=is_correct, loss=0.0),
            responses,
            eval_results,
        )

    # ── 3. Relative advantages ────────────────────────────────────────────────
    advantages = compute_advantages(rewards, normalize=config.normalize_advantages)

    # ── 4 & 5. Policy gradient loss (with grad) ───────────────────────────────
    optimizer.zero_grad()

    per_response_losses: List[torch.Tensor] = []
    for resp_text, adv in zip(responses, advantages):
        log_prob = get_response_log_probs(
            model, tokenizer, prompt_text, resp_text, config.device
        )
        # Loss = -A_i * log π(response | prompt)
        # adv is a Python float (no grad); only log_prob carries the graph.
        per_response_losses.append(-adv * log_prob)

    loss = torch.stack(per_response_losses).mean()
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
    optimizer.step()

    stats = StepStats(
        step=0,   # filled by caller
        rewards=rewards,
        is_correct=is_correct,
        loss=loss.item(),
    )
    return stats, responses, eval_results


# ── Training loop ──────────────────────────────────────────────────────────────

def train(
    config:  GRPOConfig,
    dataset: Optional[List[dict]] = None,
):
    """
    Full training loop.

    Exploration vs exploitation:
      Temperature drives exploration.  High T (≥1.0) explores diverse
      formats; low T (<0.7) exploits the model's current mode.  Start
      high to find correct formats, then optionally anneal.

    Instability risks:
      - Reward hacking: model learns surface patterns (e.g. always output
        <answer>42</answer>) without genuine reasoning.  The dense reward
        from Phase 1 mitigates this.
      - Gradient explosion: clip_grad_norm is essential.
      - Policy collapse: if advantage variance collapses (all rewards equal)
        the model stops learning.  Monitor reward_variance; if it stays near
        0 for many steps, lower temperature or increase group size.
    """
    if dataset is None:
        dataset = DEMO_QUESTIONS

    model, tokenizer = load_model(config.model_name, config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    global_step = 0

    for epoch in range(1, config.num_epochs + 1):
        print(f"\n{'─' * 60}")
        print(f"  Epoch {epoch}/{config.num_epochs}")
        print(f"{'─' * 60}")

        epoch_stats: List[StepStats] = []

        for item in dataset:
            global_step += 1
            stats, responses, eval_results = grpo_train_step(
                model, tokenizer, optimizer,
                question=item["question"],
                gt_answer=item["answer"],
                config=config,
            )
            stats.step = global_step
            epoch_stats.append(stats)

            if global_step % config.log_every == 0:
                sample_outputs = [
                    (eval_results[i]["reward"],
                     eval_results[i]["is_correct"],
                     responses[i])
                    for i in range(len(responses))
                ]
                log_step(stats, sample_outputs=sample_outputs, verbose=True)

        log_epoch_summary(epoch, epoch_stats)

    print("\nTraining complete.")
    return model, tokenizer


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = GRPOConfig(
        model_name    = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        group_size    = 4,
        max_new_tokens= 256,
        temperature   = 0.9,
        learning_rate = 5e-6,
        num_epochs    = 3,
        log_every     = 1,
    )

    print("Phase 2: GRPO Training Loop")
    print(f"  Model      : {cfg.model_name}")
    print(f"  Group size : {cfg.group_size}")
    print(f"  Device     : {cfg.device}")
    print(f"  Dataset    : {len(DEMO_QUESTIONS)} questions")
    print()

    train(cfg)
