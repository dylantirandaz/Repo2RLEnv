"""Thin GitHub client built on the `gh` CLI for auth simplicity.

We deliberately shell out to `gh` rather than depend on PyGithub:
  - `gh auth token` already gives us auth resolution for free
  - `gh api graphql` is easier than maintaining REST pagination logic
  - one less Python dep

If `gh` is not installed, we fall back to plain `curl`-style requests via
`urllib`. For v0.1 we only support the `gh`-installed path.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import date
from typing import Literal

logger = logging.getLogger(__name__)


class GitHubError(RuntimeError):
    pass


@dataclass(slots=True)
class PullRequestSummary:
    number: int
    title: str
    body: str
    state: str
    merged_at: str | None
    base_ref: str
    base_sha: str
    head_sha: str
    is_draft: bool
    url: str
    changed_files: list[str]


def _run_gh(args: list[str], token: str | None = None) -> str:
    if not shutil.which("gh"):
        raise GitHubError("gh CLI not found on PATH; install it or use a different auth path")
    env = None
    if token:
        import os

        env = {**os.environ, "GH_TOKEN": token}
    proc = subprocess.run(
        ["gh", *args],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
        check=False,
    )
    if proc.returncode != 0:
        raise GitHubError(f"gh {' '.join(args)!r} failed: {proc.stderr.strip()}")
    return proc.stdout


def _fetch_base_sha(owner: str, name: str, number: int, *, token: str | None = None) -> str | None:
    """
    Return a PR's base-branch commit SHA via the REST pulls endpoint, or None on failure.
    """
    try:
        raw = _run_gh(
            ["api", f"repos/{owner}/{name}/pulls/{number}", "--jq", ".base.sha"],
            token=token,
        )
    except GitHubError as exc:
        logger.warning("PR #%d: base_sha fetch failed, dropping: %s", number, exc)
        return None
    sha = raw.strip()
    if not sha:
        logger.warning("PR #%d: base_sha fetch returned empty, dropping", number)
        return None
    return sha


def list_merged_prs(
    owner: str,
    name: str,
    *,
    limit: int = 50,
    since: date | None = None,
    until: date | None = None,
    skip_drafts: bool = True,
    token: str | None = None,
) -> list[PullRequestSummary]:
    """List recently merged PRs ordered newest-first.

    Uses `gh pr list` (REST under the hood). Filters by date client-side.
    """
    args = [
        "pr",
        "list",
        "--repo",
        f"{owner}/{name}",
        "--state",
        "merged",
        "--limit",
        str(min(limit * 3, 1000)),  # over-fetch to allow client-side filtering
        "--json",
        "number,title,body,state,mergedAt,baseRefName,headRefOid,isDraft,url,files",
    ]
    raw = _run_gh(args, token=token)
    rows = json.loads(raw)

    summaries: list[PullRequestSummary] = []
    for r in rows:
        if skip_drafts and r.get("isDraft"):
            continue
        merged_at = r.get("mergedAt")
        if since and merged_at and merged_at[:10] < since.isoformat():
            continue
        if until and merged_at and merged_at[:10] > until.isoformat():
            continue
        base_sha = _fetch_base_sha(owner, name, r["number"], token=token)
        if base_sha is None:
            continue
        files = [f["path"] for f in (r.get("files") or [])]
        summaries.append(
            PullRequestSummary(
                number=r["number"],
                title=r["title"] or "",
                body=r.get("body") or "",
                state=r["state"],
                merged_at=merged_at,
                base_ref=r.get("baseRefName") or "",
                base_sha=base_sha,
                head_sha=r.get("headRefOid") or "",
                is_draft=bool(r.get("isDraft")),
                url=r["url"],
                changed_files=files,
            )
        )
        if len(summaries) >= limit:
            break
    return summaries


def fetch_pr_diff(owner: str, name: str, number: int, *, token: str | None = None) -> str:
    """Return the unified diff for a PR via `gh pr diff`."""
    return _run_gh(
        ["pr", "diff", str(number), "--repo", f"{owner}/{name}"],
        token=token,
    )


def fetch_issue(
    owner: str, name: str, number: int, *, token: str | None = None
) -> tuple[str, str] | None:
    """Return (title, body) for an issue, or None if it can't be fetched.

    Used by `pr_runtime` to source the problem statement from the linked
    issue (the bug *report*) rather than the PR body (the *fix* description,
    which routinely leaks the solution — commit SHAs, the approach, even the
    grading test names). Mirrors SWE-bench, which builds problem statements
    from issue text.

    `gh issue view` also resolves issue numbers that are actually PRs on the
    same repo; we tolerate that and just return whatever title/body comes
    back. Returns None on any error (issue deleted, cross-repo ref, etc.) so
    the caller can fall back to the PR body.
    """
    import json as _json

    try:
        raw = _run_gh(
            ["issue", "view", str(number), "--repo", f"{owner}/{name}", "--json", "title,body"],
            token=token,
        )
        data = _json.loads(raw)
    except (GitHubError, _json.JSONDecodeError):
        return None
    title = (data.get("title") or "").strip()
    body = (data.get("body") or "").strip()
    if not title and not body:
        return None
    return title, body


def get_primary_language(owner: str, name: str, *, token: str | None = None) -> str | None:
    """Return GitHub's primary language string for a repo, or None on failure.

    Used by the pipeline-language compatibility pre-flight check so we can
    fail fast (before bootstrap) if a Python-only pipeline is pointed at a
    Go / Rust / etc. repo. The result is GitHub Linguist's classification
    (e.g. "Python", "Go", "TypeScript"); use
    `bootstrap.language.language_from_github_name` to map it to LanguageHint.
    """
    import json as _json

    try:
        raw = _run_gh(
            ["api", f"repos/{owner}/{name}", "--jq", ".language"],
            token=token,
        ).strip()
    except GitHubError:
        return None
    if not raw or raw == "null":
        return None
    # `gh api --jq` strips quotes, but unwrap if present
    try:
        return _json.loads(raw) if raw.startswith('"') else raw
    except _json.JSONDecodeError:
        return raw


def fetch_commit_diff(owner: str, name: str, sha: str, *, token: str | None = None) -> str:
    """Return the unified diff for a single commit via `gh api`.

    Hits `GET /repos/{owner}/{repo}/commits/{sha}` with the `diff` media
    type — same shape as `git show --format= <sha>` output.
    """
    return _run_gh(
        [
            "api",
            f"repos/{owner}/{name}/commits/{sha}",
            "-H",
            "Accept: application/vnd.github.v3.diff",
        ],
        token=token,
    )


def fetch_commit_parent(owner: str, name: str, sha: str, *, token: str | None = None) -> str:
    """Return the first parent SHA of `sha` via `gh api`.

    Returns "" if the commit has no parents (root commit) or on any error.
    """
    import json as _json

    try:
        raw = _run_gh(
            ["api", f"repos/{owner}/{name}/commits/{sha}"],
            token=token,
        )
        data = _json.loads(raw)
        parents = data.get("parents", []) or []
        if not parents:
            return ""
        return parents[0].get("sha", "") or ""
    except (GitHubError, _json.JSONDecodeError):
        return ""


def fetch_file_at_ref(
    owner: str, name: str, path: str, ref: str, *, token: str | None = None
) -> str | None:
    """Return a file's raw text content at a given ref, or None on failure.

    Hits `GET /repos/{owner}/{repo}/contents/{path}?ref={ref}` with the raw
    media type. Used to give an LLM the full pre-fix source when synthesizing
    a regression test (the diff alone lacks imports + surrounding code).
    """
    try:
        return _run_gh(
            [
                "api",
                f"repos/{owner}/{name}/contents/{path}?ref={ref}",
                "-H",
                "Accept: application/vnd.github.raw",
            ],
            token=token,
        )
    except GitHubError:
        return None


PullRequestState = Literal["OPEN", "CLOSED", "MERGED"]

_PR_FILE_PAGE = 100
"""Files requested per PR. `PullRequestRecord.files_truncated` flags overflow."""

_GQL_PULL_REQUEST_PAGE = """
query($owner:String!,$name:String!,$size:Int!,$files:Int!,$dir:OrderDirection!,$after:String){
  rateLimit { remaining resetAt }
  repository(owner:$owner,name:$name){
    pullRequests(first:$size, orderBy:{field:CREATED_AT,direction:$dir}, after:$after){
      pageInfo { hasNextPage endCursor }
      nodes {
        number title body state createdAt closedAt mergedAt isDraft url
        baseRefName baseRefOid headRefOid additions deletions changedFiles
        mergeCommit { oid }
        author { login }
        labels(first:20){ nodes { name } }
        closingIssuesReferences(first:10){ nodes { number } }
        files(first:$files){ totalCount nodes { path additions deletions } }
      }
    }
  }
}
"""


@dataclass(frozen=True, slots=True)
class ChangedFile:
    """One path touched by a pull request, with its line churn."""

    path: str
    additions: int
    deletions: int


@dataclass(frozen=True, slots=True)
class PullRequestRecord:
    """Complete PR metadata for chain synthesis, harvested in one bulk pass.

    Distinct from `PullRequestSummary`: that type serves the single-PR
    pipelines and costs one extra REST call per PR to resolve `base_sha`.
    This one carries every field the chain builder needs — base/head/merge
    OIDs, churn, file paths, linked issues — and arrives 100 PRs per request.

    `merge_commit_sha` is set only for `state == "MERGED"`; an unmerged PR has
    no commit in the repository's history, so only its `head_sha` is fetchable
    (via `refs/pull/<number>/head`).
    """

    number: int
    title: str
    body: str
    state: PullRequestState
    created_at: str
    closed_at: str | None
    merged_at: str | None
    is_draft: bool
    url: str
    base_ref: str
    base_sha: str
    head_sha: str
    merge_commit_sha: str | None
    author: str | None
    additions: int
    deletions: int
    changed_file_count: int
    files: tuple[ChangedFile, ...]
    files_truncated: bool
    labels: tuple[str, ...]
    closes_issues: tuple[int, ...]

    @property
    def paths(self) -> tuple[str, ...]:
        return tuple(f.path for f in self.files)


@dataclass(frozen=True, slots=True)
class HarvestPage:
    """One page of harvested PRs plus the cursor needed to resume after it."""

    records: tuple[PullRequestRecord, ...]
    end_cursor: str | None
    has_next_page: bool
    rate_limit_remaining: int


def _parse_pr_node(node: dict) -> PullRequestRecord:
    state = node["state"]
    if state not in ("OPEN", "CLOSED", "MERGED"):
        raise GitHubError(f"PR #{node['number']}: unknown state {state!r}")
    files_block = node["files"] or {"totalCount": 0, "nodes": []}
    file_nodes = files_block["nodes"] or []
    merge_commit = node.get("mergeCommit")
    author = node.get("author")
    return PullRequestRecord(
        number=node["number"],
        title=node["title"] or "",
        body=node.get("body") or "",
        state=state,
        created_at=node["createdAt"],
        closed_at=node.get("closedAt"),
        merged_at=node.get("mergedAt"),
        is_draft=bool(node.get("isDraft")),
        url=node["url"],
        base_ref=node.get("baseRefName") or "",
        base_sha=node.get("baseRefOid") or "",
        head_sha=node.get("headRefOid") or "",
        merge_commit_sha=(merge_commit or {}).get("oid"),
        author=(author or {}).get("login"),
        additions=node.get("additions") or 0,
        deletions=node.get("deletions") or 0,
        changed_file_count=node.get("changedFiles") or 0,
        files=tuple(
            ChangedFile(
                path=f["path"],
                additions=f.get("additions") or 0,
                deletions=f.get("deletions") or 0,
            )
            for f in file_nodes
        ),
        files_truncated=files_block["totalCount"] > len(file_nodes),
        labels=tuple(n["name"] for n in (node["labels"]["nodes"] or [])),
        closes_issues=tuple(n["number"] for n in (node["closingIssuesReferences"]["nodes"] or [])),
    )


_TRANSIENT_GH_ERRORS = (
    "HTTP 500",
    "HTTP 502",
    "HTTP 503",
    "HTTP 504",
    "was submitted too quickly",
    "secondary rate limit",
    "abuse detection",
    "connection reset",
    "timeout awaiting response",
    "EOF",
)


def _run_gh_with_retry(
    args: list[str],
    *,
    token: str | None = None,
    attempts: int = 6,
) -> str:
    """`_run_gh` with bounded backoff on transient GitHub failures.

    A full PR harvest is ~700 sequential requests; GitHub returns an occasional
    502 or secondary-rate-limit refusal, and without a retry one blip discards
    the whole run. Only the errors in `_TRANSIENT_GH_ERRORS` are retried — a
    bad query or a permissions failure still raises on the first attempt.
    """
    delay = 2.0
    for attempt in range(1, attempts + 1):
        try:
            return _run_gh(args, token=token)
        except GitHubError as exc:
            message = str(exc)
            transient = any(marker in message for marker in _TRANSIENT_GH_ERRORS)
            if not transient or attempt == attempts:
                raise
            logger.warning(
                "transient gh failure (attempt %d/%d), retrying in %.0fs: %s",
                attempt,
                attempts,
                delay,
                message[-200:],
            )
            time.sleep(delay)
            delay = min(delay * 2, 60.0)
    raise AssertionError("unreachable: loop either returns or raises")


def harvest_pull_request_page(
    owner: str,
    name: str,
    *,
    after: str | None = None,
    page_size: int = 100,
    oldest_first: bool = True,
    token: str | None = None,
) -> HarvestPage:
    """Fetch one page of PRs in every state via a single GraphQL request.

    Costs 3 rate-limit points per 100 PRs, versus one REST call per PR for
    `list_merged_prs`. `oldest_first=False` walks newest-first, which lets two
    callers harvest opposite ends of a large repo concurrently.
    """
    args = [
        "api",
        "graphql",
        "-f",
        f"query={_GQL_PULL_REQUEST_PAGE}",
        "-F",
        f"owner={owner}",
        "-F",
        f"name={name}",
        "-F",
        f"size={page_size}",
        "-F",
        f"files={_PR_FILE_PAGE}",
        "-f",
        f"dir={'ASC' if oldest_first else 'DESC'}",
    ]
    if after is not None:
        args += ["-f", f"after={after}"]
    payload = json.loads(_run_gh_with_retry(args, token=token))
    if "errors" in payload:
        raise GitHubError(f"graphql errors: {payload['errors']}")
    connection = payload["data"]["repository"]["pullRequests"]
    page_info = connection["pageInfo"]
    return HarvestPage(
        records=tuple(_parse_pr_node(n) for n in connection["nodes"]),
        end_cursor=page_info["endCursor"],
        has_next_page=bool(page_info["hasNextPage"]),
        rate_limit_remaining=payload["data"]["rateLimit"]["remaining"],
    )
