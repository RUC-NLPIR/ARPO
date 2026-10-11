"""Exercise entropy weighting through a real, CPU-only policy update."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from verl import DataProto
from verl.workers.actor.dp_actor import DataParallelPPOActor


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.tensor([[3.0, 0.0, -1.0], [0.1, 0.0, -0.1], [-2.0, 2.0, 0.5]]))

    def forward(self, input_ids, **kwargs):
        return SimpleNamespace(logits=self.logits[input_ids])


@pytest.mark.parametrize(
    "enabled,entropy_coeff,response_mask",
    [
        (True, 0.0, [1, 1, 1]),
        (False, 0.0, [1, 1, 1]),
        (True, 0.1, [1, 1, 1]),
        (False, 0.1, [1, 1, 1]),
        (True, 0.0, [1, 0, 1]),
        (True, 0.0, [0, 1, 0]),
        (True, 0.1, [0, 1, 0]),
    ],
)
def test_entropy_weighted_policy_update(monkeypatch, enabled, entropy_coeff, response_mask):
    # Keep the production forward, loss, backward and optimizer paths on CPU.
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr("verl.workers.actor.dp_actor.get_device_name", lambda: "cpu")
    monkeypatch.setattr("verl.workers.actor.dp_actor.get_torch_device", lambda: torch.cpu)
    monkeypatch.setattr("verl.utils.debug.performance._get_current_mem_info", lambda: (0, 0, 0, 0))
    config = OmegaConf.create(
        {
            "enable_entropy_balanced_clipping": True,
            "enable_entropy_balanced_advantage": enabled,
            "entropy_coeff": entropy_coeff,
            "ulysses_sequence_parallel_size": 1,
            "use_torch_compile": False,
            "use_kl_loss": False,
            "use_dynamic_bsz": False,
            "ppo_mini_batch_size": 1,
            "ppo_micro_batch_size_per_gpu": 1,
            "ppo_epochs": 1,
            "clip_ratio": 0.2,
            "clip_ratio_low": None,
            "clip_ratio_high": None,
            "loss_agg_mode": "token-mean",
            "grad_clip": 100.0,
        }
    )
    policy = TinyPolicy()
    learning_rate = 0.05
    actor = DataParallelPPOActor(config, policy, torch.optim.SGD(policy.parameters(), lr=learning_rate))
    responses = torch.tensor([[1, 2, 0]])
    advantages = torch.tensor([[1.0, 2.0, -0.5]])
    mask = torch.tensor([response_mask], dtype=torch.float32)

    # At the initial policy the ratio is one, so clipping is inactive. Compute
    # the expected SGD step directly from the entropy-weighted PPO objective.
    expected_logits = policy.logits.detach().clone().requires_grad_()
    distribution = torch.distributions.Categorical(logits=expected_logits)
    log_probs = distribution.log_prob(responses)
    entropy = distribution.entropy().unsqueeze(0)
    old_log_probs = log_probs.detach()
    weights = torch.ones_like(advantages)
    selected_entropy = entropy.detach()[mask.bool()]
    if enabled and selected_entropy.numel() > 1:
        weights += 0.2 * (entropy.detach() - selected_entropy.mean()) / selected_entropy.std()
    expected_pg_loss = -(advantages * weights * (log_probs - old_log_probs).exp() * mask).sum() / mask.sum()
    expected_loss = expected_pg_loss - entropy_coeff * (entropy * mask).sum() / mask.sum()
    expected_loss.backward()
    expected_parameters = expected_logits.detach() - learning_rate * expected_logits.grad

    batch = TensorDict(
        {
            "input_ids": torch.tensor([[0, 1, 2, 0]]),
            "responses": responses,
            "attention_mask": torch.ones(1, 4, dtype=torch.long),
            "position_ids": torch.arange(4).unsqueeze(0),
            "loss_mask": torch.cat([torch.zeros(1, 1), mask], dim=1),
            "old_log_probs": old_log_probs,
            "advantages": advantages,
        },
        batch_size=[1],
    )
    with patch.object(actor, "_forward_micro_batch", wraps=actor._forward_micro_batch) as forward:
        metrics = actor.update_policy(DataProto(batch=batch, meta_info={"temperature": 1.0, "multi_turn": True}))

    assert metrics["actor/pg_loss"][0] == pytest.approx(expected_pg_loss.item())
    assert torch.isfinite(torch.tensor(metrics["actor/grad_norm"])).all()
    torch.testing.assert_close(policy.logits, expected_parameters)
    assert forward.call_args.kwargs["calculate_entropy"] == (enabled or entropy_coeff != 0)
