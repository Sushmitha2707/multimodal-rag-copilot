import pytest

torch = pytest.importorskip("torch")

from copilot.dpo import dpo_loss, sequence_logprob  # noqa: E402


def test_loss_is_log2_when_policy_equals_reference():
    c, r = torch.tensor([-10.0]), torch.tensor([-12.0])
    loss, margin = dpo_loss(c, r, c.clone(), r.clone(), beta=0.1)
    assert abs(loss.item() - 0.6931) < 1e-3 and margin.item() == 0.0


def test_loss_drops_when_policy_prefers_chosen():
    ref_c, ref_r = torch.tensor([-10.0]), torch.tensor([-10.0])
    base, _ = dpo_loss(ref_c, ref_r, ref_c, ref_r)
    better, margin = dpo_loss(torch.tensor([-5.0]), torch.tensor([-15.0]), ref_c, ref_r)
    assert better.item() < base.item() and margin.item() > 0


def test_sequence_logprob_only_counts_response_tokens():
    class Uniform(torch.nn.Module):  # uniform distribution over a vocab of 4
        def forward(self, input_ids, attention_mask):
            class Out:
                logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], 4)

            return Out()

    ids = torch.tensor([[0, 1, 2, 3]])
    mask = torch.tensor([[0.0, 0.0, 1.0, 1.0]])  # 2 response tokens
    lp = sequence_logprob(Uniform(), ids, torch.ones_like(ids), mask)
    assert abs(lp.item() - 2 * torch.log(torch.tensor(0.25)).item()) < 1e-5
