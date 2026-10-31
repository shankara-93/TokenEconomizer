"""Learning from the agent's earlier session logs (`plugin.backfill`), then from live use."""
import json
import os
import time

import pytest

from tokeneconomizer.plugin import backfill

from test_live_hooks import call, env, repo, store_for, trap_session  # noqa: F401

OLD = time.time() - 3600


def history(env, tmp_path, repo, sid, i, mtime=OLD, cwd=None):
    """A finished session log where Claude Code keeps them, naming the directory it ran in."""
    proj = tmp_path / "claude" / "projects" / "proj"
    proj.mkdir(parents=True, exist_ok=True)
    path = str(proj / f"{sid}.jsonl")
    trap_session(path, repo, i)
    lines = [json.loads(ln) for ln in open(path)]
    for n, ln in enumerate(lines):
        ln["cwd"] = cwd or repo
        ln["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime(mtime + n))
    with open(path, "w") as fh:
        fh.write("\n".join(json.dumps(x) for x in lines) + "\n")
    os.utime(path, (mtime, mtime))
    return path


@pytest.fixture
def cenv(env, tmp_path):
    return dict(env, CLAUDE_CONFIG_DIR=str(tmp_path / "claude"))


def test_history_is_folded_once_and_learned(cenv, tmp_path, repo):
    for i in range(5):
        history(cenv, tmp_path, repo, f"old{i}", i)
    first = backfill.run(cenv, repo)
    assert (first["found"], first["folded"], first["skipped"]) == (5, 5, 0)
    st = store_for(cenv, repo)
    assert set(st.read("folded.json")) == {f"old{i}" for i in range(5)}
    assert "pytest" in (st.read("injection.json") or {}).get("rendered", "")
    again = backfill.run(cenv, repo)
    assert (again["folded"], again["skipped"]) == (0, 5)
    assert len(st.read_lines("ledger.jsonl")) == 5


def test_live_and_missing_sessions_are_left_alone(cenv, tmp_path, repo):
    history(cenv, tmp_path, repo, "live", 1, mtime=time.time())
    history(cenv, tmp_path, repo, "gone", 2, cwd=str(tmp_path / "deleted"))
    assert backfill.run(cenv, repo)["found"] == 0
    assert not os.path.exists(store_for(cenv, repo).path("folded.json"))


def test_first_session_start_backfills_once(cenv, tmp_path, repo):
    history(cenv, tmp_path, repo, "old0", 0)
    base = {"session_id": "new1", "cwd": repo, "hook_event_name": "SessionStart"}
    call(cenv, dict(base, source="startup", transcript_path=str(tmp_path / "t.jsonl")))
    assert "old0" in store_for(cenv, repo).read("folded.json")
    assert store_for(cenv, repo).read(backfill.MARKER)
    history(cenv, tmp_path, repo, "old9", 9)
    call(cenv, dict(base, session_id="new2", source="startup",
                    transcript_path=str(tmp_path / "t.jsonl")))
    assert "old9" not in store_for(cenv, repo).read("folded.json")


def test_another_repositorys_history_is_left_alone(cenv, tmp_path, repo):
    other = tmp_path / "other"
    other.mkdir()
    history(cenv, tmp_path, repo, "mine", 1)
    history(cenv, tmp_path, repo, "theirs", 2, cwd=str(other))
    assert backfill.run(cenv, repo)["folded"] == 1
    assert backfill.run(cenv, str(other))["folded"] == 1
