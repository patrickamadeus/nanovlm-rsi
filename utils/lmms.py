"""The bridge to lmms-eval: its tasks, prompts and scoring; our policy, with its trained LoRA as-is.

lmms-eval asks a model to answer requests; `answer` is how our policy answers one. Decoding is
greedy with no ACT: prefill — a benchmark measures the policy, it doesn't sample from it.
"""
import numpy as np


def answer(policy, args, docs) -> str:
    context, gen_kwargs, doc_to_visual, doc_id, task, split = args
    visuals = doc_to_visual(docs[task][split][doc_id]) or []
    images = [np.asarray(v.convert("RGB")) for v in visuals]
    policy.cfg.max_new_tokens = gen_kwargs.get("max_new_tokens", 256)
    text = policy.act(context, images, prefix="", greedy=True).text
    for stop in gen_kwargs.get("until") or []:
        text = text.split(stop)[0]
    return text.strip()


def run_benchmarks(policy, tasks: list[str], limit: int | None = None) -> dict:
    """{task: {metric: value}} from lmms-eval. `limit` is for debugging; reported numbers use every item."""
    from lmms_eval import evaluator
    from lmms_eval.api.model import lmms
    from tqdm import tqdm

    class PolicyLMM(lmms):
        def generate_until(self, requests):
            return [answer(policy, r.args, self.task_dict) for r in tqdm(requests, desc="benchmark")]

        def loglikelihood(self, requests):
            raise NotImplementedError("generation-only")

        def generate_until_multi_round(self, requests):
            raise NotImplementedError("single-turn only")

    out = evaluator.simple_evaluate(model=PolicyLMM(), tasks=tasks, batch_size=1, limit=limit)
    return {t: {k: v for k, v in m.items() if isinstance(v, (int, float))} for t, m in out["results"].items()}
