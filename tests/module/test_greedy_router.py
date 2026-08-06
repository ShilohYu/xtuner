import torch

from xtuner.v1.module.router.greedy import GreedyRouterConfig, apply_random_logits


def test_force_load_balance_is_opt_in():
    config = GreedyRouterConfig(
        scoring_func="sigmoid",
        router_scaling_factor=1.0,
        norm_topk_prob=True,
    )

    assert config.force_load_balance is False
    assert config.build(n_routed_experts=32, num_experts_per_tok=4).force_load_balance is False


def test_force_load_balance_randomizes_routing_and_preserves_gradient():
    torch.manual_seed(0)
    logits = torch.zeros(16384, 32, requires_grad=True)

    random_logits = apply_random_logits(logits)
    routing_weights = random_logits.sigmoid()
    topk_ids = torch.topk(routing_weights, k=4, dim=-1).indices
    counts = torch.bincount(topk_ids.flatten(), minlength=32).float()

    assert not torch.equal(random_logits, logits)
    assert counts.max() / counts.mean() < 1.1
    routing_weights.sum().backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.norm() > 0


def test_force_load_balance_random_logits_is_torch_compile_compatible():
    torch.manual_seed(0)
    logits = torch.zeros(8, 4)
    compiled_apply_random_logits = torch.compile(apply_random_logits, backend="eager", fullgraph=True)

    random_logits = compiled_apply_random_logits(logits)

    assert random_logits.shape == logits.shape
    assert not torch.equal(random_logits, logits)
