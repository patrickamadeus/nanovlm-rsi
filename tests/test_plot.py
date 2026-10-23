import os

from utils.log import Cost, Trace
from utils.plot import curves, plot_runs


def _run(tmp_path, name, rates):
    tr = Trace(str(tmp_path / name))
    tr.log_run(cfg_hash="h", git_sha="g", seeds={}, update="grpo", curriculum="frontier", gate_enabled=True)
    for i, r in enumerate(rates):
        tr.log_eval(split="heldout", n=200, success_rate=r, unseen_seed_rate=r, unseen_goal_rate=0.0,
                    cost=Cost(env_steps=100 * i, policy_tokens=1000 * i, gpu_seconds=10.0 * i))
    tr.log_eval(split="mmstar", n=1500, success_rate=0.3, bench_subset=False, cost=Cost())
    return str(tmp_path / name)


def test_curves_are_heldout_success_against_cumulative_cost(tmp_path):
    xs, ys, label = curves(_run(tmp_path, "a", [0.1, 0.2, 0.4]), x="gpu_seconds")
    assert xs == [0.0, 10.0, 20.0] and ys == [0.1, 0.2, 0.4]   # mmstar records are not part of the curve
    assert "grpo" in label and "frontier" in label


def test_plot_writes_a_png_comparing_runs(tmp_path):
    out = str(tmp_path / "cmp.png")
    plot_runs([_run(tmp_path, "a", [0.1, 0.3]), _run(tmp_path, "b", [0.1, 0.2])], out, x="env_steps")
    assert os.path.getsize(out) > 1000
