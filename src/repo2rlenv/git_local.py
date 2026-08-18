"""Thin subprocess wrappers around the local `git` CLI.

Used by `commit_runtime` to walk a local clone's history. Bootstrap clones
at `--depth=1`; commit_runtime needs deeper history (`--depth=200` by
default) so `git log` can return real candidates.

We deliberately don't use a Python git library (GitPython, dulwich) — the
`git` CLI is the source of truth, stable across versions, and matches what
the bootstrap container has access to.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from datetime import date
from pathlib import Path

logger = logging.getLogger(__name__)


class GitError(RuntimeError):
    """Raised when a `git` subprocess returns non-zero."""


@dataclass(slots=True)
class CommitInfo:
    """One commit's metadata, parsed from `git log --format=...`."""

    sha: str  # full 40-char SHA
    parent_sha: str  # first parent (empty string for root commits)
    parents: list[str]  # full list (>1 means merge commit)
    author_name: str
    author_email: str
    authored_at: str  # ISO8601 (e.g. "2026-03-12T08:15:22Z")
    subject: str
    body: str

    @property
    def is_merge(self) -> bool:
        return len(self.parents) > 1

    @property
    def message(self) -> str:
        """Subject + blank line + body, like `git show -s`."""
        if not self.body:
            return self.subject
        return f"{self.subject}\n\n{self.body}"


# Field delimiter unlikely to appear in commit data
_FIELD_SEP = "\x1f"  # ASCII unit separator
_RECORD_SEP = "\x1e"  # ASCII record separator

# git --format placeholders — see `git help log` § PRETTY FORMATS
_LOG_FORMAT = (
    _FIELD_SEP.join(
        [
            "%H",  # full SHA
            "%P",  # parent SHAs (space-separated; first is %p)
            "%an",  # author name
            "%ae",  # author email
            "%aI",  # author date, ISO8601 strict
            "%s",  # subject
            "%b",  # body
        ]
    )
    + _RECORD_SEP
)


def _run_git(args: list[str], cwd: Path, *, timeout: int = 60) -> str:
    """Run `git ...` and return stdout. Raises GitError on non-zero exit."""
    r = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if r.returncode != 0:
        raise GitError(
            f"git {' '.join(args)!r} failed (exit {r.returncode}): {r.stderr.strip()[:400]}"
        )
    return r.stdout


def list_commits(
    clone_dir: Path,
    *,
    since: date | None = None,
    until: date | None = None,
    limit: int = 50,
    branch: str = "HEAD",
) -> list[CommitInfo]:
    """Return commits on `branch` between `since` and `until`, newest first.

    Filters happen at git's layer (`--since`, `--until`, `-n <limit>`) so
    we don't have to walk the full history in Python.

    Empty list if the range produces no commits or the branch doesn't exist.
    """
    args = [
        "log",
        f"--max-count={limit}",
        f"--format={_LOG_FORMAT}",
        "--no-decorate",
    ]
    if since is not None:
        args.append(f"--since={since.isoformat()}")
    if until is not None:
        args.append(f"--until={until.isoformat()}")
    args.append(branch)

    raw = _run_git(args, clone_dir, timeout=120)
    return [c for c in _parse_log_output(raw)]


def _parse_log_output(raw: str) -> list[CommitInfo]:
    """Parse the `--format=%H\x1f%P\x1f...\x1e` stream into CommitInfo list."""
    out: list[CommitInfo] = []
    # Records are RS-separated; the last record may have a trailing newline
    records = [r for r in raw.split(_RECORD_SEP) if r.strip()]
    for rec in records:
        fields = rec.lstrip("\n").split(_FIELD_SEP)
        if len(fields) < 7:
            logger.debug("skipping malformed git log record: %r", rec[:100])
            continue
        sha, parents_str, author_name, author_email, authored_at, subject, body = (
            fields[0],
            fields[1],
            fields[2],
            fields[3],
            fields[4],
            fields[5],
            fields[6],
        )
        parents = parents_str.split() if parents_str else []
        out.append(
            CommitInfo(
                sha=sha,
                parent_sha=parents[0] if parents else "",
                parents=parents,
                author_name=author_name,
                author_email=author_email,
                authored_at=authored_at,
                subject=subject,
                body=body.strip(),
            )
        )
    return out


def show_diff(clone_dir: Path, commit_sha: str) -> str:
    """Return the unified diff introduced by `commit_sha` (no commit message header).

    Uses `git show --format= --patch` which suppresses the commit-info
    block at the top; we get the raw `diff --git a/X b/Y` sequence the
    same shape as `gh pr diff` output, so the existing
    `split_patch_and_test_patch()` parser works unchanged.
    """
    return _run_git(
        ["show", "--format=", "--patch", "--no-color", commit_sha],
        clone_dir,
        timeout=60,
    )


def changed_files(clone_dir: Path, commit_sha: str) -> list[str]:
    """Return the list of paths touched by `commit_sha` (relative to repo root)."""
    raw = _run_git(
        ["show", "--no-color", "--format=", "--name-only", commit_sha],
        clone_dir,
        timeout=30,
    )
    return [line for line in raw.splitlines() if line.strip()]


@dataclass(frozen=True, slots=True)
class FirstParentCommit:
    """One step along a branch's first-parent history.

    `git diff parent_sha sha` is exactly the change the branch received at this
    step, whether the project squash-merges, merge-commits, or pushes directly.
    That uniformity is what makes a replayable chain of stages possible.
    """

    sha: str
    parent_sha: str
    subject: str
    committed_at: str
    is_merge: bool


def first_parent_history(
    clone_dir: Path,
    *,
    ref: str = "HEAD",
    limit: int | None = None,
) -> list[FirstParentCommit]:
    """Return `ref`'s first-parent history, OLDEST first.

    Oldest-first is the order a chain replays in. The root commit is dropped:
    it has no parent, so no diff can be taken against it.
    """
    args = ["log", "--first-parent", "--format=%H%x1f%P%x1f%cI%x1f%s", "--no-decorate"]
    if limit is not None:
        args.append(f"--max-count={limit}")
    args.append(ref)
    raw = _run_git(args, clone_dir, timeout=300)

    out: list[FirstParentCommit] = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        fields = line.split(_FIELD_SEP)
        if len(fields) < 4:
            logger.debug("skipping malformed first-parent record: %r", line[:100])
            continue
        sha, parents_str, committed_at, subject = fields[0], fields[1], fields[2], fields[3]
        parents = parents_str.split()
        if not parents:
            continue  # root commit — nothing to diff against
        out.append(
            FirstParentCommit(
                sha=sha,
                parent_sha=parents[0],
                subject=subject,
                committed_at=committed_at,
                is_merge=len(parents) > 1,
            )
        )
    out.reverse()
    return out


def range_diff(clone_dir: Path, before: str, after: str) -> str:
    """Return the unified diff that takes the tree from `before` to `after`."""
    return _run_git(
        ["diff", "--no-color", "--no-ext-diff", f"{before}..{after}"],
        clone_dir,
        timeout=120,
    )


def range_changed_files(clone_dir: Path, before: str, after: str) -> list[str]:
    """Return the paths that differ between two commits."""
    raw = _run_git(
        ["diff", "--no-color", "--name-only", f"{before}..{after}"],
        clone_dir,
        timeout=60,
    )
    return [line for line in raw.splitlines() if line.strip()]


def file_at_commit(clone_dir: Path, commit: str, path: str) -> str | None:
    """Return a file's contents at a commit, or None when it does not exist there.

    Absence is an ordinary outcome, not an error: a stage's test file may have
    been deleted by the change being replayed, and the caller mirrors that by
    removing the file rather than failing.
    """
    proc = subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=str(clone_dir),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if proc.returncode != 0:
        return None
    return proc.stdout
