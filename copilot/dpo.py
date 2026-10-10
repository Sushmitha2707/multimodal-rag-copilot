"""Direct Preference Optimization - loss and helpers written in plain PyTorch."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def dpo_loss(
    policy_chosen: torch.Tensor,
    policy_rejected: torch.Tensor,
    ref_chosen: torch.Tensor,
    ref_rejected: torch.Tensor,
    beta: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Inputs are summed log-probs of the *response tokens* for each sequence, shape (batch,).
    loss = -log sigmoid( beta * [ (pi_c - ref_c) - (pi_r - ref_r) ] )
    Returns (mean loss, implicit reward margin per example).
    """
    margin = beta * ((policy_chosen - ref_chosen) - (policy_rejected - ref_rejected))
    return -F.logsigmoid(margin).mean(), margin.detach()


def sequence_logprob(model, input_ids: torch.Tensor, attention_mask: torch.Tensor, resp_mask: torch.Tensor):
    """Sum of log p(token) over response positions. resp_mask is 1 on response tokens, 0 on prompt/pad."""
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1, :]
    labels = input_ids[:, 1:]
    logp = torch.gather(logits.log_softmax(-1), 2, labels.unsqueeze(-1)).squeeze(-1)
    return (logp * resp_mask[:, 1:]).sum(-1)
