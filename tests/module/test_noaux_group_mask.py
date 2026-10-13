"""Group exclusion must survive negative correction scores without poisoning aux gradients."""

import pytest
import torch

from xtuner.v1.module.router.noaux_router import NoAuxRouter


@pytest.mark.parametrize("bias", [[-.6, -.7, -.8, -.9], [0., -.1, -.2, -.3]])
def test_group_exclusion_and_aux_backward(bias):
    router = NoAuxRouter(
        n_routed_experts=4,
        num_experts_per_tok=2,
        router_scaling_factor=2.5,
        scoring_func="sigmoid",
        n_group=2,
        topk_group=1,
    )
    device = router.e_score_correction_bias.device
    router.e_score_correction_bias.copy_(torch.tensor(bias, device=device))
    logits = torch.zeros(1, 4, device=device, requires_grad=True)
    result = router(logits)
    assert result["topk_ids"].sort().values.tolist() == [[0, 1]]
    torch.testing.assert_close(result["topk_weights"], torch.full((1, 2), 1.25, device=device))
    # Preserve the existing masked, bias-corrected auxiliary distribution.
    corrected = logits.sigmoid() + router.e_score_correction_bias
    finite_aux = corrected * torch.tensor([[1., 1., 0., 0.]], device=device)
    expected_aux = finite_aux / finite_aux.sum(-1, keepdim=True)
    torch.testing.assert_close(result["router_weights"], expected_aux, rtol=0, atol=0)
    coefficients = torch.tensor([[.2, .8, -.3, .4]], device=device)
    loss = (result["router_weights"] * coefficients).sum() + result["topk_weights"].square().sum()
    expected = (expected_aux * coefficients).sum() + result["topk_weights"].square().sum()
    actual_grad = torch.autograd.grad(loss, logits, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, logits)[0]
    assert torch.isfinite(actual_grad).all()
    torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
