"""
Phase 2 (v2): GRPO Training Loop
=================================
Improvements over v1
--------------------
  LoRA / PEFT      -- fine-tune only adapter weights (~5% of params); base
                       weights stay frozen and act as the reference model
                       for KL computation at zero extra memory cost.
  KL penalty       -- loss += kl_beta * KL(pi_current || pi_ref) prevents
                       the policy from drifting too far from the base model.
  Clipped ratio    -- PPO-style clip(ratio, 1-e, 1+e) caps individual update
                       magnitudes; avoids destructive large steps.
  Grad accumulation -- accumulate over N questions before stepping; reduces
                       gradient noise without increasing memory.
  LR warmup+cosine -- linear warmup then cosine decay; RL fine-tuning is
                       sensitive to learning rate spikes at step 1.
  Running reward   -- cross-step reward normalisation keeps advantage scale
  normalisation       stable as accuracy improves during training.
  Checkpointing    -- saves model every save_every steps and at end of run.
  JSONL logging    -- every step appended to results/training_log.jsonl for
                       post-hoc curve analysis.
  config.yaml      -- all hyperparameters in one place; load with load_config().

Run
---
  python train_grpo.py              # reads config.yaml
  python train_grpo.py --demo       # overrides dataset to built-in 10 Qs
"""

import re
import sys
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
    _SCHEDULER_OK = True
except ImportError:
    _SCHEDULER_OK = False

try:
    from peft import get_peft_model, LoraConfig, TaskType
    _PEFT_OK = True
except ImportError:
    _PEFT_OK = False

try:
    import yaml
    _YAML_OK = True
except ImportError:
    _YAML_OK = False

from verifier import verify
from sampling_utils import build_prompt, build_full_response, generate_group
from logging_utils import StepStats, log_step, log_epoch_summary, RewardLogger


# ── Config ─────────────────────────────────────────────────────────────────────

@dataclass
class GRPOConfig:
    # Model
    model_name:           str   = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    # LoRA
    use_lora:             bool  = True
    lora_r:               int   = 16
    lora_alpha:           int   = 32
    lora_dropout:         float = 0.05
    # Sampling
    group_size:           int   = 4
    max_new_tokens:       int   = 256
    temperature:          float = 0.9
    top_k:                int   = 50
    # Optimisation
    learning_rate:        float = 5e-6
    grad_clip:            float = 1.0
    num_epochs:           int   = 3
    accumulation_steps:   int   = 4
    warmup_steps:         int   = 50
    # GRPO
    kl_beta:              float = 0.04
    clip_epsilon:         float = 0.2
    normalize_advantages: bool  = True
    # Logging & checkpoints
    log_every:            int   = 1
    save_every:           int   = 100
    checkpoint_dir:       str   = "checkpoints"
    log_path:             str   = "results/training_log.jsonl"
    # Dataset
    dataset:              str   = "gsm8k"
    max_train_samples:    int   = 500
    # Device (resolved at load time)
    device:               str   = "cpu"


def load_config(path: str = "config.yaml") -> GRPOConfig:
    """
    Load GRPOConfig from a YAML file.
    Unknown keys are silently ignored; missing keys keep dataclass defaults.
    Falls back to default GRPOConfig if PyYAML is absent or file not found.
    """
    cfg = GRPOConfig()
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"

    if not _YAML_OK:
        print("PyYAML not installed — using defaults.  pip install pyyaml")
        return cfg

    try:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
    except FileNotFoundError:
        return cfg

    valid_keys = GRPOConfig.__dataclass_fields__.keys()
    for k, v in data.items():
        if k == "device":
            cfg.device = ("cuda" if torch.cuda.is_available() else "cpu") if v == "auto" else v
        elif k in valid_keys and v is not None:
            setattr(cfg, k, v)
    return cfg


# ── Demo / fallback dataset ────────────────────────────────────────────────────

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


def load_gsm8k(max_samples: int = 500) -> List[dict]:
    """
    Load the GSM8K training split.
    Answers are extracted from the '#### <number>' suffix in each solution.
    Falls back to DEMO_QUESTIONS when 'datasets' is not installed.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("  'datasets' not installed — using DEMO_QUESTIONS.  pip install datasets")
        return DEMO_QUESTIONS

    print("Loading GSM8K ...")
    ds = load_dataset("gsm8k", "main")["train"]
    items: List[dict] = []
    for row in ds:
        m = re.search(r'####\s*(-?[\d,]+)', row["answer"])
        if m:
            items.append({
                "question": row["question"],
                "answer":   m.group(1).replace(",", ""),
            })
        if len(items) >= max_samples:
            break
    print(f"  Loaded {len(items)} GSM8K training examples.")
    return items


def get_dataset(config: GRPOConfig) -> List[dict]:
    if config.dataset == "gsm8k":
        return load_gsm8k(config.max_train_samples)
    return DEMO_QUESTIONS


# ── Model loading ──────────────────────────────────────────────────────────────

def load_model(config: GRPOConfig):
    """
    Load model + tokenizer, optionally wrapping with LoRA adapters.

    LoRA keeps base weights frozen — those frozen weights serve as the
    reference distribution for KL penalty computation at zero extra cost.
    Without LoRA, a second model copy would be needed for KL; in that case
    set kl_beta=0 or accept the ~2x memory overhead by loading twice.
    """
    print(f"Loading {config.model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if config.device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name, torch_dtype=dtype, device_map=config.device
    )

    if config.use_lora:
        if not _PEFT_OK:
            print("  peft not installed — LoRA disabled.  pip install peft")
        else:
            lora_cfg = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=["q_proj", "v_proj"],
                bias="none",
            )
            model = get_peft_model(model, lora_cfg)
            model.print_trainable_parameters()

    model.train()
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {n_train:,}   Device: {config.device}")
    return model, tokenizer


# ── Log-probability computation ────────────────────────────────────────────────

def get_response_log_probs(
    model,
    tokenizer,
    prompt_text:   str,
    response_text: str,
    device:        str,
) -> torch.Tensor:
    """
    Sum of log pi(token | context) over the response tokens.

    Prompt and response are tokenised separately so that the subword
    boundary between them is always clean (no merged tokens).
    Returns a scalar tensor; gradient graph is attached when called outside
    torch.no_grad().
    """
    prompt_ids   = tokenizer(prompt_text,   return_tensors="pt",
                             add_special_tokens=True).input_ids.to(device)
    response_ids = tokenizer(response_text, return_tensors="pt",
                             add_special_tokens=False).input_ids.to(device)

    if response_ids.shape[1] == 0:
        # Empty generation — return a zero that still carries a grad
        return torch.zeros(1, device=device, requires_grad=True).squeeze()

    full_ids       = torch.cat([prompt_ids, response_ids], dim=1)
    attention_mask = torch.ones_like(full_ids)
    logits         = model(full_ids, attention_mask=attention_mask).logits

    prompt_len = prompt_ids.shape[1]
    resp_len   = response_ids.shape[1]
    # logits[i] predicts token[i+1]; response tokens start at prompt_len
    pred_logits = logits[0, prompt_len - 1 : prompt_len + resp_len - 1, :]
    log_probs   = F.log_softmax(pred_logits, dim=-1)
    token_lp    = log_probs.gather(1, response_ids[0].unsqueeze(1)).squeeze(1)
    return token_lp.sum()


def _ref_log_probs(
    model,
    tokenizer,
    prompt:    str,
    response:  str,
    device:    str,
    use_lora:  bool,
) -> float:
    """
    Log probs under the reference (frozen) distribution.

    LoRA path  — disables all adapter layers so only the frozen base weights
                 run the forward pass.  Zero extra memory; the base model IS
                 the reference.
    Non-LoRA   — there is no separate reference; returns the current policy's
                 log prob (effectively setting kl_beta contribution to 0 for
                 this step).  Set kl_beta=0.0 in config when not using LoRA.
    """
    with torch.no_grad():
        if use_lora and _PEFT_OK and hasattr(model, "disable_adapter"):
            with model.disable_adapter():
                lp = get_response_log_probs(model, tokenizer, prompt, response, device)
        else:
            lp = get_response_log_probs(model, tokenizer, prompt, response, device)
    return lp.item()


# ── GRPO loss (clipped PG + KL) ───────────────────────────────────────────────

def compute_grpo_loss(
    log_prob_new: torch.Tensor,
    log_prob_old: float,
    log_prob_ref: float,
    advantage:    float,
    kl_beta:      float,
    clip_epsilon: float,
    device:       str,
) -> torch.Tensor:
    """
    Per-response GRPO loss:

      ratio   = exp(log_new - log_old)          -- how much policy changed
      pg_loss = -min(ratio * A, clip(ratio) * A) -- clipped policy gradient
      kl_loss = log_new - log_ref               -- KL divergence approximation
      loss    = pg_loss + kl_beta * kl_loss

    Clipping stops the ratio from straying too far from 1 in a single step,
    preventing catastrophic forgetting.  The KL term pulls the policy back
    toward the reference (base model) between steps.
    """
    adv_t   = torch.tensor(advantage,    dtype=log_prob_new.dtype, device=device)
    ref_t   = torch.tensor(log_prob_ref, dtype=log_prob_new.dtype, device=device)

    ratio   = torch.exp(log_prob_new - log_prob_old)
    clipped = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
    pg_loss = -torch.min(ratio * adv_t, clipped * adv_t)
    kl_loss = log_prob_new - ref_t        # gradient only through log_prob_new

    return pg_loss + kl_beta * kl_loss


# ── Running reward statistics ─────────────────────────────────────────────────

class RunningRewardStats:
    """
    Rolling window of reward values across training steps.

    Why this matters: early in training, rewards are mostly 0 (format errors).
    As the model learns the format, mean reward rises and the within-group std
    alone under-estimates the true scale.  Normalising by the running std keeps
    advantages comparable throughout training.
    """

    def __init__(self, window: int = 200) -> None:
        self._buf:    List[float] = []
        self._window: int        = window

    def update(self, rewards: List[float]) -> None:
        self._buf.extend(rewards)
        self._buf = self._buf[-self._window:]

    @property
    def mean(self) -> float:
        return statistics.mean(self._buf) if self._buf else 0.0

    @property
    def std(self) -> float:
        return statistics.stdev(self._buf) if len(self._buf) > 1 else 1.0

    def normalize(self, rewards: List[float]) -> List[float]:
        """Group-relative baseline then cross-step scale normalisation."""
        group_mean = statistics.mean(rewards)
        adv        = [r - group_mean for r in rewards]
        scale      = max(self.std, 1e-6)
        return [a / scale for a in adv]


# ── Single training step ───────────────────────────────────────────────────────

def grpo_train_step(
    model,
    tokenizer,
    question:      str,
    gt_answer:     str,
    config:        GRPOConfig,
    running_stats: RunningRewardStats,
) -> tuple:
    """
    One GRPO step — computes and backwards the loss but does NOT call
    optimizer.step().  The caller manages accumulation and the step boundary.

    Returns (StepStats, responses, eval_results).
    """
    prompt_text = build_prompt(question)

    # 1 ── Group sampling (no grad) ────────────────────────────────────────────
    responses, _ = generate_group(
        model, tokenizer, prompt_text,
        N=config.group_size, max_new_tokens=config.max_new_tokens,
        temperature=config.temperature, top_k=config.top_k, device=config.device,
    )

    # 2 ── Reward computation ──────────────────────────────────────────────────
    rewards:      List[float] = []
    is_correct:   List[bool]  = []
    eval_results: List[dict]  = []

    for resp_text in responses:
        result = verify(build_full_response(resp_text), gt=gt_answer, question=question)
        rewards.append(result["reward"])
        is_correct.append(result["is_correct"])
        eval_results.append(result)

    running_stats.update(rewards)

    # Skip update when the whole group has identical rewards (zero advantage)
    if max(rewards) - min(rewards) < 1e-6:
        return (
            StepStats(step=0, rewards=rewards, is_correct=is_correct, loss=0.0),
            responses, eval_results,
        )

    # 3 ── Advantages ──────────────────────────────────────────────────────────
    advantages = running_stats.normalize(rewards)

    # 4 ── Old + ref log probs (both no grad, before any weight update) ────────
    log_probs_old: List[float] = []
    log_probs_ref: List[float] = []

    for resp_text in responses:
        with torch.no_grad():
            lp_old = get_response_log_probs(
                model, tokenizer, prompt_text, resp_text, config.device
            )
        log_probs_old.append(lp_old.item())
        log_probs_ref.append(_ref_log_probs(
            model, tokenizer, prompt_text, resp_text,
            config.device, config.use_lora,
        ))

    # 5 ── Clipped PG + KL loss (with grad) ───────────────────────────────────
    losses: List[torch.Tensor] = []
    for resp_text, adv, lp_old, lp_ref in zip(
        responses, advantages, log_probs_old, log_probs_ref
    ):
        lp_new = get_response_log_probs(
            model, tokenizer, prompt_text, resp_text, config.device
        )
        losses.append(compute_grpo_loss(
            lp_new, lp_old, lp_ref, adv,
            config.kl_beta, config.clip_epsilon, config.device,
        ))

    raw_loss = torch.stack(losses).mean()
    # Divide by accumulation_steps so the effective LR is unchanged
    (raw_loss / config.accumulation_steps).backward()

    return (
        StepStats(step=0, rewards=rewards, is_correct=is_correct, loss=raw_loss.item()),
        responses, eval_results,
    )


# ── Training loop ──────────────────────────────────────────────────────────────

def train(
    config:  GRPOConfig,
    dataset: Optional[List[dict]] = None,
):
    """
    Full GRPO training loop with all v2 improvements.

    Optimizer step is decoupled from the per-question step so that gradient
    accumulation is handled cleanly: loss.backward() runs every question,
    but optimizer.step() runs every accumulation_steps questions.
    """
    if dataset is None:
        dataset = get_dataset(config)

    model, tokenizer = load_model(config)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=config.learning_rate)

    # LR schedule: warmup then cosine decay
    total_opt_steps = max(1, len(dataset) * config.num_epochs // config.accumulation_steps)
    if _SCHEDULER_OK and config.warmup_steps > 0:
        scheduler = SequentialLR(
            optimizer,
            schedulers=[
                LinearLR(optimizer, start_factor=0.1, end_factor=1.0,
                         total_iters=config.warmup_steps),
                CosineAnnealingLR(optimizer,
                                  T_max=max(1, total_opt_steps - config.warmup_steps),
                                  eta_min=1e-7),
            ],
            milestones=[config.warmup_steps],
        )
    else:
        scheduler = None

    running_stats  = RunningRewardStats(window=200)
    reward_logger  = RewardLogger(config.log_path)
    ckpt_dir       = Path(config.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    opt_step    = 0
    optimizer.zero_grad()

    for epoch in range(1, config.num_epochs + 1):
        print(f"\n{'=' * 60}")
        print(f"  Epoch {epoch}/{config.num_epochs}  |  "
              f"dataset={len(dataset)}  |  device={config.device}")
        print(f"{'=' * 60}")
        epoch_stats: List[StepStats] = []

        for item in dataset:
            global_step += 1

            stats, responses, eval_results = grpo_train_step(
                model, tokenizer,
                question=item["question"], gt_answer=item["answer"],
                config=config, running_stats=running_stats,
            )
            stats.step = global_step
            epoch_stats.append(stats)

            # ── Optimiser step at accumulation boundary ────────────────────────
            if global_step % config.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, config.grad_clip)
                optimizer.step()
                if scheduler:
                    scheduler.step()
                optimizer.zero_grad()
                opt_step += 1

            # ── Console logging ────────────────────────────────────────────────
            if global_step % config.log_every == 0:
                sample_outputs = [
                    (eval_results[i]["reward"],
                     eval_results[i]["is_correct"],
                     responses[i])
                    for i in range(len(responses))
                ]
                log_step(stats, sample_outputs=sample_outputs, verbose=True)

            # ── JSONL logging ──────────────────────────────────────────────────
            current_lr = scheduler.get_last_lr()[0] if scheduler else config.learning_rate
            reward_logger.log(stats, question=item["question"],
                              extra={"lr": current_lr, "opt_step": opt_step})

            # ── Checkpoint ────────────────────────────────────────────────────
            if config.save_every > 0 and global_step % config.save_every == 0:
                ckpt_path = ckpt_dir / f"step_{global_step}"
                model.save_pretrained(str(ckpt_path))
                tokenizer.save_pretrained(str(ckpt_path))
                print(f"  Checkpoint -> {ckpt_path}")

        log_epoch_summary(epoch, epoch_stats)

    # ── Final checkpoint ───────────────────────────────────────────────────────
    final_path = ckpt_dir / "final"
    model.save_pretrained(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    print(f"\nTraining complete.  Final model -> {final_path}")
    return model, tokenizer


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    use_demo = "--demo" in sys.argv
    cfg = load_config("config.yaml")
    if use_demo:
        cfg.dataset = "demo"

    print("Phase 2 (v2): GRPO Training")
    print(f"  model    : {cfg.model_name}")
    print(f"  lora     : {cfg.use_lora}  (r={cfg.lora_r})")
    print(f"  kl_beta  : {cfg.kl_beta}  clip_eps={cfg.clip_epsilon}")
    print(f"  accum    : {cfg.accumulation_steps}  warmup={cfg.warmup_steps}")
    print(f"  dataset  : {cfg.dataset}  max={cfg.max_train_samples}")
    print(f"  device   : {cfg.device}")
    print()
    train(cfg)
