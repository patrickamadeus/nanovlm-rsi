# Prompts

One file per environment, copied from the baseline we compare against, with slots the harness fills:
`{memory}` (skills, cards, rules), `{feedback}` (first turn: "[Initial Observation]:", later: the actions
just taken), `{state}` (the env's text: step count, inventory), `{max_actions}`, `{sep}`; `<image>` marks the frame.

| file | source |
|---|---|
| `frozenlake.txt` | VAGEN, `vagen/envs/frozenlake/utils/prompt.py` (system prompt + `free_think` format), verbatim |
| `sokoban.txt` | VAGEN, `vagen/envs/sokoban/utils/prompt.py` (system prompt + `free_think` format), verbatim |
| `craftax.txt` | BALROG, `balrog/environments/crafter/__init__.py` (action list and descriptions), with our action names, one goal per episode, and VAGEN's `free_think` format |
| `default.txt`, `agent.txt` | picoRSI's own one-line `ACT:` formats (count / Cauldron, and the first grid runs) |

VAGEN's FrozenLake prompt mentions slippery tiles even in its non-slippery configuration; we keep it verbatim.
