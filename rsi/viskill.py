"""ViSkill (Li et al., 2026, arXiv 2610.12403): successful attempts become *visual* skill cards, a retrieved
card is shown to the policy as an image, and failures that followed a card earn partial reward.

  card     one composite image per skill: the start frame with the path drawn on it, then every frame in
           order labelled with its action, and the policy's own one-line strategy rendered inside the image
  admit    a success only if it is optimal-length (cfg.mem_max_excess_steps = 0) and new enough (dedupe)
  retrieve by the env's layout descriptor: score = 0.9 * similarity + 0.1 * utility, shown only if >= 0.8
  reward   a failed attempt that had a card in context gets  r_env + lambda * r_skill,
              r_skill = max(0, (l - (L - l)) / L_ref)
           l = steps it followed the card's moves before leaving them, L = its length, L_ref = the card's length.
           The paper says "furthest progress along the reference"; matching the reference's moves is our reading.

Storage, retrieval, utility and the exact revert are the skill store's (rsi/memory.py); this file adds the
card and the reward. Formats: "visual" (card only) or "text+visual" (card and the strategy as text).
"""
import os
from dataclasses import replace

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from rsi.memory import Memory
from utils.rollout import Context, moves

THUMB, PATH, LABEL_H = 112, (250, 210, 40), 18


def skill_reward(actions: list[str], reference: list[str]) -> float:
    if not reference or not actions:
        return 0.0
    followed = 0
    for a, b in zip(actions, reference):
        if a != b:
            break
        followed += 1
    return max(0.0, (followed - (len(actions) - followed)) / len(reference))


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:                       # Pillow < 10.1 has one fixed-size default font
        return ImageFont.load_default()


def render_card(traj, strategy: str = "", cell_px: int | None = None) -> np.ndarray:
    """[overview with path | frame grid], strategy underneath. Frames are the observation before each turn
    plus the final one, labelled with the turn's moves ("1 down", "2 right,right", ..., "goal")."""
    turns = [st for st in traj.steps if _acts(st)]
    frames = [st["images"][-1] for st in turns] + [traj.final_rgb]
    labels = [f"{i + 1} {','.join(_acts(st))}" for i, st in enumerate(turns)] + ["goal"]
    if len(frames) > 8:                     # keep the first, the last, and evenly spaced ones between
        keep = np.linspace(0, len(frames) - 1, 8).round().astype(int)
        frames, labels = [frames[i] for i in keep], [labels[i] for i in keep]
    overview = Image.fromarray(frames[0]).convert("RGB")
    draw = ImageDraw.Draw(overview)
    acts = [a for st in turns for a in _acts(st)]
    cells = [p for st in turns for p in (st.get("path") or ([st["pos"]] if st.get("pos") else []))]
    if cell_px and cells and len(cells) == len(acts) and all(cells):
        start = traj.steps[0]["images"][-1]
        points = [_start_pos(acts[0], cells[0], cell_px, start)] + [
            (c * cell_px + cell_px // 2, r * cell_px + cell_px // 2) for r, c in cells]
        draw.line(points, fill=PATH, width=max(cell_px // 8, 3))
        x, y = points[-1]
        draw.ellipse([x - 6, y - 6, x + 6, y + 6], fill=PATH)
    side = 2 * THUMB + LABEL_H * 2
    overview = overview.resize((side, side))
    cols = 4
    rows = (len(frames) + cols - 1) // cols
    width, height = side + cols * THUMB, max(side, rows * (THUMB + LABEL_H))
    text_h = 40 if strategy else 0
    card = Image.new("RGB", (width, height + text_h), (255, 255, 255))
    card.paste(overview, (0, 0))
    draw, font = ImageDraw.Draw(card), _font(14)
    for i, (f, label) in enumerate(zip(frames, labels)):
        x, y = side + (i % cols) * THUMB, (i // cols) * (THUMB + LABEL_H)
        card.paste(Image.fromarray(f).convert("RGB").resize((THUMB - 4, THUMB - 4)), (x + 2, y + 2))
        draw.text((x + 4, y + THUMB), label, fill=(0, 0, 0), font=font)
    if strategy:
        draw.text((6, height + 10), f"Strategy: {strategy}"[:110], fill=(0, 0, 0), font=font)
    return np.asarray(card)


def _acts(st) -> list[str]:
    return st.get("actions") or ([st["action"]] if st.get("action") else [])


def _start_pos(first_action, first_cell, cell_px, frame):
    """Where the path starts: the cell the first move left, recovered from the first recorded position."""
    from envs.frozenlake import MOVES
    r, c = first_cell
    dr, dc = MOVES.get(first_action, (0, 0))
    pr, pc = r - dr, c - dc
    if not (0 <= pr * cell_px < frame.shape[0] and 0 <= pc * cell_px < frame.shape[1]):
        pr, pc = r, c                       # the first move was blocked by the edge
    return pc * cell_px + cell_px // 2, pr * cell_px + cell_px // 2


class ViSkill(Memory):
    name = "viskill"

    def __init__(self, store, cfg, env=None, policy=None):
        if cfg.skill_format not in ("visual", "text+visual"):
            cfg = replace(cfg, skill_format="visual")
        super().__init__(store, cfg, env, policy)
        self.card_dir = os.path.join(cfg.run_dir, "cards")
        self._images: dict[str, np.ndarray] = {}

    def strategy(self, s) -> str:
        fmt, self.fmt = self.fmt, "text"     # the card always carries the policy's strategy
        try:
            return super().strategy(s)
        finally:
            self.fmt = fmt

    def skill_record(self, s) -> dict:
        record = super().skill_record(s)
        record["card_rgb"] = render_card(s.traj, record["strategy"], getattr(self.env, "cell_px", None))
        return record

    def commit(self, update) -> None:
        os.makedirs(self.card_dir, exist_ok=True)
        added = []
        for s in update.payload["skills"]:
            sid = os.urandom(5).hex()
            path = os.path.join(self.card_dir, f"{sid}.png")
            if self.store.append(replace(self.to_skill({**s, "id": sid}), card=path)) is not None:
                Image.fromarray(s["card_rgb"]).save(path)     # a card is written only for a skill that was kept
                added.append(sid)
        self.store.expire()
        self._added[update.payload["revert_offset"]] = added

    def revert(self, update) -> None:
        super().revert(update)
        for sid in self._added.get(update.payload["revert_offset"], []):
            path = os.path.join(self.card_dir, f"{sid}.png")
            if os.path.exists(path):
                os.remove(path)

    def persisted(self, update) -> bool:
        added = self._added.get(update.payload["revert_offset"], [])
        return super().persisted(update) and all(os.path.exists(os.path.join(self.card_dir, f"{sid}.png"))
                                                 for sid in added)

    def image(self, path: str) -> np.ndarray:
        if path not in self._images:
            self._images[path] = np.asarray(Image.open(path).convert("RGB"))
        return self._images[path]

    def context(self, task, env) -> Context:
        if not len(self.store) or self.cfg.mem_top_k <= 0:
            return Context()
        skills = self.store.retrieve(self.key(task, env), self.cfg.mem_top_k, self.cfg.mem_retrieve_alpha,
                                     self.cfg.mem_retrieve_min, task.family)
        skills = [s for s in skills if s.card and os.path.exists(s.card)]
        if not skills:
            return Context()
        lines = ["[Skill Reference] How you solved a similar level before:"]
        for s in skills:
            lines.append("<image>" + (f"\nStrategy: {s.strategy}" if self.fmt == "text+visual" and s.strategy else ""))
        return Context("\n".join(lines), tuple(self.image(s.card) for s in skills), tuple(s.id for s in skills))

    def signal(self, batch: list, env) -> list:
        lam = self.cfg.viskill_lambda
        if lam <= 0:
            return batch
        by_id = {s.id: s for s in self.store.skills()}
        out = []
        for s in batch:
            ref = next((by_id[i] for i in s.traj.context_ids if i in by_id), None)
            if s.reward < 1.0 and ref is not None:
                acts = moves(s.traj)
                s = replace(s, reward=s.reward + lam * skill_reward(acts, list(ref.actions)))
            out.append(s)
        return out
