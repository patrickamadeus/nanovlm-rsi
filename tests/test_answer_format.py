from utils.prompt import load_scaffold, parse_turn, render

GRID = ["up", "down", "left", "right"]


def test_answer_format_reads_several_actions_and_checks_the_format():
    text = "<think>Box is right, target further right.</think><answer>Right, Right, Up</answer>"
    assert parse_turn(text, GRID, "answer", 3, ",") == (["right", "right", "up"], True)


def test_answer_format_keeps_valid_actions_but_flags_bad_ones_and_caps_the_count():
    assert parse_turn("<think>x</think><answer>Up, Jump</answer>", GRID, "answer", 3, ",") == (["up"], False)
    assert parse_turn("<think>x</think><answer>Up,Up,Up,Up</answer>", GRID, "answer", 3, ",") == (["up"] * 3, False)


def test_answer_without_think_or_without_answer():
    assert parse_turn("<answer>Down</answer>", GRID, "answer", 1, ",") == (["down"], False)   # acts, no format credit
    assert parse_turn("<think>I will go down</think>", GRID, "answer", 1, ",") == ([], False)


def test_multiword_actions_match_their_names():
    legal = ["noop", "do", "place_table", "make_wood_pickaxe"]
    assert parse_turn("<think>t</think><answer>Place Table</answer>", legal, "answer", 1, ",") == (["place_table"], True)


def test_act_format_is_the_old_single_action_line():
    assert parse_turn("ACT: up", GRID, "act", 1, ",") == (["up"], True)
    assert parse_turn("ACT: fly", GRID, "act", 1, ",") == ([], False)


def test_env_prompts_render_every_slot():
    for env in ("frozenlake", "sokoban", "craftax"):
        out = render(load_scaffold(f"prompts/{env}.txt"), memory="", feedback="[Initial Observation]:",
                     state="Step 0 of 12.", max_actions=3, sep=",", actions=GRID, system="", task="")
        assert "{" not in out.replace("{}", "") and out.count("<image>") == 1 and "up to 3 action(s)" in out


class Plan:
    """Answers each turn with the next chunk of the solver's plan in VAGEN's format."""
    version = 0

    def __init__(self, plan, per_turn, think=True):
        self.plan, self.k, self.think, self.prompts = list(plan), per_turn, think, []

    def act_batch(self, prompts, images, **kw):
        from models.vlm import RolloutStep
        out = []
        for p in prompts:
            self.prompts.append(p)
            chunk, self.plan = self.plan[:self.k], self.plan[self.k:]
            body = ", ".join(a.capitalize() for a in chunk) or "Up"
            text = (f"<think>go</think>" if self.think else "") + f"<answer>{body}</answer>"
            out.append(RolloutStep(text, [1, 2], np.zeros(2, np.float32), 5))
        return out


def _cfg(**kw):
    from configs.config import LoopConfig, PolicyConfig
    return LoopConfig(answer_format="answer", action_sep=",",
                      policy=PolicyConfig(prompt_path="prompts/frozenlake.txt"), **kw)


def test_a_turn_executes_several_actions_and_feedback_names_them():
    import numpy as np
    from envs.frozenlake import FrozenLakeEnv
    from utils.rollout import moves, rollout_group
    env = FrozenLakeEnv()
    task = next(t for t in env.tasks("train", 50, np.random.default_rng(0)) if len(env.solve(t)) >= 4)
    plan = env.solve(task)
    policy = Plan(plan, per_turn=3)
    traj = rollout_group(policy, env, [task], _cfg(max_actions_per_turn=3, max_turns=5))[0]
    assert traj.total_reward == 1.0 and moves(traj) == plan
    assert len(traj.steps) == -(-len(plan) // 3) and all(st["format_ok"] for st in traj.steps)
    assert "[Initial Observation]:" in policy.prompts[0]
    assert f"the extracted valid action is {', '.join(plan[:3])}" in policy.prompts[1]


def test_max_turns_caps_the_episode_independently_of_the_step_budget():
    import numpy as np
    from envs.frozenlake import FrozenLakeEnv
    from utils.rollout import rollout_group
    env = FrozenLakeEnv()
    task = env.tasks("train", 1, np.random.default_rng(0))[0]
    traj = rollout_group(Plan(["left", "right"] * 50, per_turn=1), env, [task], _cfg(max_turns=2))[0]
    assert len(traj.steps) <= 2


import numpy as np  # noqa: E402
