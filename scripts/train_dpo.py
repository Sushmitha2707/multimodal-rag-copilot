"""Preference-tune a small open LLM (LoRA + DPO) on feedback collected from the copilot UI.

A usable preference pair = a thumbs-DOWN answer (rejected) + the human-corrected answer (chosen).
This is a lightweight, DPO-style preference-optimisation loop, not full RLHF (no reward model / PPO).
Needs a GPU for anything beyond a toy run (e.g. a free Colab T4) and `pip install -r requirements-dpo.txt`.

    python -m scripts.train_dpo --model Qwen/Qwen2.5-0.5B-Instruct --epochs 2
"""

import argparse
import json
import random
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from copilot.dpo import dpo_loss, sequence_logprob
from copilot.feedback import build_preference_pairs
from copilot.prompts import SYSTEM_PROMPT


def encode(tok, prompt: str, response: str, max_len: int, device):
    text = tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    p_ids = tok(text, add_special_tokens=False)["input_ids"]
    r_ids = tok(response + (tok.eos_token or ""), add_special_tokens=False)["input_ids"]
    p_ids = p_ids[-max(1, max_len - len(r_ids)) :]  # keep the end of the prompt if too long
    ids = torch.tensor([p_ids + r_ids], device=device)
    resp_mask = torch.tensor([[0] * len(p_ids) + [1] * len(r_ids)], device=device, dtype=torch.float32)
    return ids, torch.ones_like(ids), resp_mask


def pair_logps(model, tok, ex, max_len, device):
    c = sequence_logprob(model, *encode(tok, ex["prompt"], ex["chosen"], max_len, device))
    r = sequence_logprob(model, *encode(tok, ex["prompt"], ex["rejected"], max_len, device))
    return c, r


@torch.no_grad()
def preference_accuracy(model, tok, data, max_len, device) -> float:
    """Fraction of held-out pairs where the model assigns higher log-prob to the chosen answer."""
    if not data:
        return float("nan")
    wins = 0
    for ex in data:
        c, r = pair_logps(model, tok, ex, max_len, device)
        wins += int(c.item() > r.item())
    return wins / len(data)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--pairs-file", default=None, help="JSONL with prompt/chosen/rejected (default: feedback DB)")
    ap.add_argument("--out", default="models/dpo-adapter")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=1536)
    ap.add_argument("--holdout", type=float, default=0.2)
    args = ap.parse_args()

    if args.pairs_file:
        data = [json.loads(line) for line in open(args.pairs_file)]
    else:
        data = build_preference_pairs()
    if len(data) < 10:
        raise SystemExit(f"Only {len(data)} preference pairs - collect at least a few dozen before training.")
    random.Random(0).shuffle(data)
    n_hold = max(2, int(len(data) * args.holdout))
    held, train = data[:n_hold], data[n_hold:]
    print(f"{len(train)} train pairs, {len(held)} held-out pairs")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).to(device)
    model = get_peft_model(
        model, LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
    )
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    before = preference_accuracy(model, tok, held, args.max_len, device)
    model.train()
    step = 0
    for epoch in range(args.epochs):
        random.shuffle(train)
        for i, ex in enumerate(train, start=1):
            with torch.no_grad(), model.disable_adapter():  # reference = base model
                ref_c, ref_r = pair_logps(model, tok, ex, args.max_len, device)
            pol_c, pol_r = pair_logps(model, tok, ex, args.max_len, device)
            loss, margin = dpo_loss(pol_c, pol_r, ref_c, ref_r, beta=args.beta)
            (loss / args.grad_accum).backward()
            if i % args.grad_accum == 0 or i == len(train):
                opt.step()
                opt.zero_grad()
                step += 1
                print(f"epoch {epoch + 1} step {step} loss {loss.item():.4f} margin {margin.item():+.3f}")

    model.eval()
    after = preference_accuracy(model, tok, held, args.max_len, device)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    result = {"heldout_pairs": len(held), "pref_acc_before": before, "pref_acc_after": after}
    Path("results").mkdir(exist_ok=True)
    Path("results/dpo_eval.json").write_text(json.dumps(result, indent=2))
    print(result)


if __name__ == "__main__":
    main()
