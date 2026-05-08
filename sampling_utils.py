"""
Group sampling utilities for GRPO training.

Prompt contract
---------------
The prompt ends with the opening <think> tag so the model continues
the reasoning chain.  After decoding, we prepend "<think>\\n" to the
generated text before calling verify(), giving verify() a complete
response it can parse.
"""

import torch
from typing import List

# Prompt ends with the opening <think> tag; model completes from there.
PROMPT_TEMPLATE  = "User: {question}\nAssistant: <think>\n"
ASSISTANT_PREFIX = "<think>\n"   # prepended to generated text for verify()


def build_prompt(question: str) -> str:
    """Format a question into the standard GRPO prompt."""
    return PROMPT_TEMPLATE.format(question=question.strip())


def build_full_response(generated_text: str) -> str:
    """
    Reconstruct the full assistant response for reward computation.
    Adds the <think> opener that was part of the prompt, not the generation.
    """
    return ASSISTANT_PREFIX + generated_text


def generate_group(
    model,
    tokenizer,
    prompt_text: str,
    N: int = 4,
    max_new_tokens: int = 256,
    temperature: float = 0.9,
    top_k: int = 50,
    device: str = "cuda",
) -> tuple[List[str], int]:
    """
    Generate N independent completions for a single prompt.

    Returns
    -------
    responses       : list[str]  — N decoded strings (generated tokens only,
                                   no prompt echo)
    prompt_token_len: int        — number of prompt tokens; callers can use
                                   this to re-slice token IDs if needed
    """
    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(device)

    prompt_len = inputs["input_ids"].shape[1]
    responses: List[str] = []

    for _ in range(N):
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_k=top_k,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        # Slice off prompt tokens so we only decode what was generated.
        generated_ids = output_ids[0, prompt_len:]
        response_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        responses.append(response_text)

    return responses, prompt_len
