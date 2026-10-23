"""Plot held-out success against cumulative cost — never against iteration.

A memory update costs almost no GPU time and a GRPO update costs a lot, so iteration-indexed
curves would make one carrier look slow and the other efficient purely through the x-axis.

    python -m harness.plot runs/a runs/b --x gpu_seconds --out compare.png
"""
import argparse
import os

from harness.log import Trace

AXES = {"gpu_seconds": "cumulative GPU seconds", "env_steps": "cumulative env steps",
        "policy_tokens": "cumulative policy tokens"}


def curves(run_dir: str, x: str = "gpu_seconds") -> tuple[list[float], list[float], str]:
    tr = Trace(run_dir)
    evals = [r for r in tr.records("eval") if r.get("split") == "heldout"]
    run = (tr.records("run") or [{}])[0]
    label = f"{os.path.basename(os.path.normpath(run_dir))} ({run.get('update')}, {run.get('curriculum')})"
    return [r["cost"][x] for r in evals], [r["success_rate"] for r in evals], label


def plot_runs(run_dirs: list[str], out: str, x: str = "gpu_seconds") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    for d in run_dirs:
        xs, ys, label = curves(d, x)
        ax.plot(xs, ys, marker="o", markersize=3, label=label)
    ax.set_xlabel(AXES[x])
    ax.set_ylabel("held-out success rate")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--x", default="gpu_seconds", choices=list(AXES))
    ap.add_argument("--out", default="success_vs_cost.png")
    args = ap.parse_args()
    plot_runs(args.runs, args.out, args.x)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
