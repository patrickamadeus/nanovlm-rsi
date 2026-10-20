import numpy as np

from infra.memory import Skill, SkillStore
from tests.conftest import fake_embed


def skill(situation, reward=1.0):
    return Skill(id="", family="count_objects", situation=situation, actions=("say_3",),
                 reward=reward, embedding=fake_embed(situation), uses=0, uses_before_success=0)


def make(tmp_path, **kw):
    return SkillStore(str(tmp_path / "skills.jsonl"), embed_fn=fake_embed, **kw)


def test_append_and_retrieve_top_k(tmp_path):
    store = make(tmp_path)
    store.append(skill("count three shapes"))
    store.append(skill("gather in order"))
    got = store.retrieve("count shapes", k=1)
    assert len(got) == 1 and "count" in got[0].situation


def test_dedupe_rejects_near_duplicate(tmp_path):
    store = make(tmp_path, dedupe_sim=0.95)
    assert store.append(skill("count three shapes", reward=1.0)) is not None
    assert store.append(skill("count three shapes", reward=0.5)) is None
    assert len(store) == 1


def test_dedupe_keeps_higher_reward_on_collision(tmp_path):
    store = make(tmp_path)
    store.append(skill("count three shapes", reward=0.5))
    store.append(skill("count three shapes", reward=1.0))
    assert len(store) == 1 and store.retrieve("count", 1)[0].reward == 1.0


def test_truncate_restores_byte_identical_state(tmp_path):
    store = make(tmp_path)
    path = tmp_path / "skills.jsonl"
    store.append(skill("a"))
    off, before = store.offset(), path.read_bytes()
    sid = store.append(skill("b"))
    store.note_use([sid], succeeded=True)
    store.truncate(off)
    assert path.read_bytes() == before and len(store) == 1


def test_reader_skips_truncated_final_line(tmp_path):
    store = make(tmp_path)
    store.append(skill("a"))
    with (tmp_path / "skills.jsonl").open("a") as f:
        f.write('{"op": "add", "situ')
    assert len(make(tmp_path)) == 1


def test_state_survives_reload(tmp_path):
    sid = make(tmp_path).append(skill("count three shapes"))
    again = make(tmp_path)
    assert len(again) == 1 and again.retrieve("count", 1)[0].id == sid


def test_note_use_tracks_use_attribution(tmp_path):
    store = make(tmp_path)
    sid = store.append(skill("a"))
    store.note_use([sid], succeeded=True)
    store.note_use([sid], succeeded=False)
    s = store.retrieve("a", 1)[0]
    assert s.uses == 2 and s.uses_before_success == 1


def test_expire_caps_size_by_usefulness(tmp_path):
    store = make(tmp_path, max_size=2)
    ids = [store.append(skill(f"s{i}", reward=1.0)) for i in range(5)]
    store.note_use([ids[3]], succeeded=True)
    assert store.expire() == 3 and len(store) == 2
    assert ids[3] in {s.id for s in store.retrieve("s3", 2)}


def test_as_text_lists_situation_and_actions(tmp_path):
    store = make(tmp_path)
    store.append(skill("count three shapes"))
    text = store.as_text(store.retrieve("count", 1))
    assert "count three shapes" in text and "say_3" in text
    assert store.as_text([]) == ""
