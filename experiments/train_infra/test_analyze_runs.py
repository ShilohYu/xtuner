"""A partial, duplicated or discontinuous optimizer trace must not pass completion."""
from experiments.train_infra.analyze_runs import optimizer_steps_complete


def test_optimizer_trace_requires_every_step_on_every_rank():
    full = [{"step": i} for i in range(20)]
    assert optimizer_steps_complete([full] * 8, 20)
    assert not optimizer_steps_complete([full] * 7 + [full[:6]], 20)
    assert not optimizer_steps_complete([full] * 7 + [full[:10] + full[11:] + [full[-1]]], 20)
    assert not optimizer_steps_complete([full] * 7 + [full[:-1] + [{"step": 20}]], 20)
    assert not optimizer_steps_complete([full] * 8, None)
    assert not optimizer_steps_complete([], 20)
