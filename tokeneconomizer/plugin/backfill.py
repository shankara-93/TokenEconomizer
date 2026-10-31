"""Learning from the session history an agent already has on disk.

The hooks only see sessions that run while the plugin is installed. This reads the agent's own
session logs (Claude Code: `~/.claude/projects/*/<session id>.jsonl`) and folds each finished one
that ran in one repository (the one `--cwd` names) into it, oldest first, exactly as `fold.fold_session` folds a
live one. From then on the hooks keep learning as before.

  * A session is folded once (`folded.json`), so a re-run, or a session the hooks also saw, is
    never counted twice.
  * A log still being written (touched within `fold.STALE_S`) is left for its own SessionEnd.
  * A session whose directory no longer exists has no repository to learn about; it is skipped.
  * Other repositories' sessions are left alone, so a repository learns from its history only
    where the plugin is in use.
  * The paid proposer is not run unless asked (`paid=True`): a backfill can be hundreds of
    sessions, and the free layer learns from every one of them.

`maybe_start` runs it once per repository, detached, from that repository's first SessionStart.
"""
import dataclasses
import glob
import os
import re
import time
from datetime import datetime

from . import fold, repos
from .context import LiveContext
from .paths import repo_identity, state_root
from .store import Store

HOSTS = ("claude",)
MARKER = "backfill.json"
LOCK = "backfill"
_TS_RE = re.compile(r'"timestamp"\s*:\s*"([^"]+)"')


def projects_dir(env):
    base = env.get("CLAUDE_CONFIG_DIR") or os.path.join(os.path.expanduser("~"), ".claude")
    return os.path.join(base, "projects")


def _epoch(text):
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _span(path, window=65536):
    """(first, last) timestamp of a session log, read from its head and its tail only."""
    size = os.path.getsize(path)
    with open(path, "rb") as fh:
        head = fh.read(window).decode("utf-8", errors="replace")
        fh.seek(max(0, size - window))
        tail = fh.read().decode("utf-8", errors="replace")
    first, last = _TS_RE.search(head), _TS_RE.findall(tail)
    return (_epoch(first.group(1)) if first else None,
            _epoch(last[-1]) if last else None)


def discover(env, now=None, quiet_s=fold.STALE_S):
    """Finished Claude Code sessions on disk: [(started, session id, log path, cwd)], oldest
    first. Sub-agent logs live in subdirectories and are part of their parent session."""
    now = now or time.time()
    out = []
    for path in glob.glob(os.path.join(projects_dir(env), "*", "*.jsonl")):
        try:
            if now - os.path.getmtime(path) < quiet_s:
                continue
            cwd = repos._transcript_cwd(path)
            if not cwd or not os.path.isdir(cwd):
                continue
            first, _ = _span(path)
        except OSError:
            continue
        out.append((first or os.path.getmtime(path), os.path.basename(path)[:-6], path, cwd))
    return sorted(out)


def run(env, cwd, host_name="claude", paid=False, limit=0, log=None):
    """Fold every unfolded finished session of the repository `cwd` is in. Returns a summary dict, or None when
    another backfill holds the lock."""
    from . import hosts
    host = hosts.HOSTS[host_name]()
    log = log or (lambda msg: None)
    key = repo_identity(cwd, env).key
    found = [f for f in discover(env) if repo_identity(f[3], env).key == key]
    if limit:
        found = found[-limit:]
    summary = {"found": len(found), "folded": 0, "skipped": 0, "repos": set()}
    ctxs = {}
    root_store = Store(state_root(env))
    with root_store.locked(LOCK, timeout=0) as got:
        if not got:
            return None
        for started, sid, path, cwd in found:
            ctx = ctxs.get(cwd)
            if ctx is None:
                ctx = ctxs[cwd] = _context(host, cwd, env, paid)
            sess_name = os.path.join("sessions", f"{sid}.json")
            if (ctx is None or sid in (ctx.store.read("folded.json") or {})
                    or os.path.exists(ctx.store.path(sess_name))):
                summary["skipped"] += 1
                continue
            _, ended = _span(path)
            ctx.store.write(sess_name, {
                "session_id": sid, "host": host.name, "cwd": cwd, "transcript": path,
                "started": started, "ended": ended or os.path.getmtime(path),
                "backfilled": True, "turns": [], "nudges": 0, "aborts": 0,
                "judge_tokens": 0.0, "writer_tokens": 0.0})
            if fold.fold_session(ctx, sid) is None:
                summary["skipped"] += 1
                continue
            summary["folded"] += 1
            summary["repos"].add(ctx.repo.label)
            log(f"folded {sid} into {ctx.repo.label}")
    summary["repos"] = sorted(summary["repos"])
    return summary


def _context(host, cwd, env, paid):
    ctx = LiveContext.build(host, cwd, env, settle=False)
    if not ctx.config.enabled:
        return None
    cfg = ctx.config
    if "distiller" in cfg.microskills and not paid:
        if cfg.microskills == "distiller":      # nothing free to learn with
            cfg = dataclasses.replace(cfg, microskills="off")
        else:
            cfg = dataclasses.replace(cfg, microskills="envfacts")
    repos.note_repo(ctx.root, ctx.repo)
    return dataclasses.replace(ctx, config=cfg)


def maybe_start(ctx):
    """On a repository's first SessionStart, fold the agent's earlier sessions of that repository in a detached
    process. Once per repository and agent; `te backfill` runs it again by hand."""
    if ctx.host.name not in HOSTS or not ctx.config.enabled:
        return False
    if ctx.store.read(MARKER) or ctx.env.get("TE_BACKFILL") == "0":
        return False
    ctx.store.write(MARKER, {"started": time.time()})
    if ctx.env.get("TE_FOLD_SYNC") == "1":
        return bool(run(ctx.env, ctx.repo.root, ctx.host.name))
    return bool(ctx.spawn(["backfill"], "backfill.log"))
