"""SQLite corpus of every pull request in a repository.

`pr_chain` needs the *whole* PR history at once, not a recent slice: chains are
built by finding PRs that touch overlapping files in merge order, which is a
global query over tens of thousands of PRs. Holding that in memory per run and
re-fetching it per invocation is wasteful, so the corpus is persisted once and
queried thereafter.

Two properties matter:

* **Resumable.** A harvest of 65k PRs spans ~700 GraphQL requests. The cursor
  for each direction is checkpointed after every page, so an interrupted run
  resumes instead of restarting.
* **Queryable by path.** `pr_files` is indexed on `path`, which is what turns
  "which PRs touched `cron/scheduler.py`" from a full scan into a lookup.

The store is keyed by `(owner, name)` in its filename, one database per repo.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from repo2rlenv.github import (
    ChangedFile,
    PullRequestRecord,
    PullRequestState,
    harvest_pull_request_page,
)

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS pull_requests (
    number              INTEGER PRIMARY KEY,
    title               TEXT    NOT NULL,
    body                TEXT    NOT NULL,
    state               TEXT    NOT NULL CHECK (state IN ('OPEN','CLOSED','MERGED')),
    created_at          TEXT    NOT NULL,
    closed_at           TEXT,
    merged_at           TEXT,
    is_draft            INTEGER NOT NULL,
    url                 TEXT    NOT NULL,
    base_ref            TEXT    NOT NULL,
    base_sha            TEXT    NOT NULL,
    head_sha            TEXT    NOT NULL,
    merge_commit_sha    TEXT,
    author              TEXT,
    additions           INTEGER NOT NULL,
    deletions           INTEGER NOT NULL,
    changed_file_count  INTEGER NOT NULL,
    files_truncated     INTEGER NOT NULL,
    labels              TEXT    NOT NULL,
    closes_issues       TEXT    NOT NULL
);
CREATE TABLE IF NOT EXISTS pr_files (
    number      INTEGER NOT NULL REFERENCES pull_requests(number) ON DELETE CASCADE,
    path        TEXT    NOT NULL,
    additions   INTEGER NOT NULL,
    deletions   INTEGER NOT NULL,
    PRIMARY KEY (number, path)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS idx_pr_files_path ON pr_files(path);
CREATE INDEX IF NOT EXISTS idx_pr_state_merged ON pull_requests(state, merged_at);
CREATE TABLE IF NOT EXISTS harvest_cursor (
    stream      TEXT PRIMARY KEY CHECK (stream IN ('asc','desc')),
    cursor      TEXT,
    exhausted   INTEGER NOT NULL,
    pages       INTEGER NOT NULL
);
"""

_INSERT_PR = """
INSERT INTO pull_requests VALUES
    (:number,:title,:body,:state,:created_at,:closed_at,:merged_at,:is_draft,:url,
     :base_ref,:base_sha,:head_sha,:merge_commit_sha,:author,:additions,:deletions,
     :changed_file_count,:files_truncated,:labels,:closes_issues)
ON CONFLICT(number) DO UPDATE SET
    state=excluded.state, closed_at=excluded.closed_at, merged_at=excluded.merged_at,
    merge_commit_sha=excluded.merge_commit_sha, is_draft=excluded.is_draft,
    additions=excluded.additions, deletions=excluded.deletions,
    changed_file_count=excluded.changed_file_count, files_truncated=excluded.files_truncated,
    labels=excluded.labels, closes_issues=excluded.closes_issues
"""

HarvestStream = str
"""Either `'asc'` (oldest PR first) or `'desc'`. Checked by the schema."""


@dataclass(frozen=True, slots=True)
class HarvestProgress:
    """Emitted after each checkpointed page so a caller can render progress."""

    stream: HarvestStream
    pages: int
    stored: int
    corpus_size: int
    rate_limit_remaining: int
    exhausted: bool


def corpus_path(cache_dir: Path, owner: str, name: str) -> Path:
    return cache_dir / "pr_corpus" / f"{owner}__{name}.sqlite"


class PRCorpus:
    """Persistent store of harvested pull requests for one repository."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._db = sqlite3.connect(path)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        # Two harvest streams run as separate processes against one file.
        self._db.execute("PRAGMA busy_timeout=30000")
        self._db.executescript(_SCHEMA)
        self._db.commit()

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> PRCorpus:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ---- writes ----

    def store(self, records: tuple[PullRequestRecord, ...]) -> int:
        """Upsert records and their file rows in one transaction."""
        with self._db:
            self._db.executemany(
                _INSERT_PR,
                [
                    {
                        "number": r.number,
                        "title": r.title,
                        "body": r.body,
                        "state": r.state,
                        "created_at": r.created_at,
                        "closed_at": r.closed_at,
                        "merged_at": r.merged_at,
                        "is_draft": int(r.is_draft),
                        "url": r.url,
                        "base_ref": r.base_ref,
                        "base_sha": r.base_sha,
                        "head_sha": r.head_sha,
                        "merge_commit_sha": r.merge_commit_sha,
                        "author": r.author,
                        "additions": r.additions,
                        "deletions": r.deletions,
                        "changed_file_count": r.changed_file_count,
                        "files_truncated": int(r.files_truncated),
                        "labels": json.dumps(list(r.labels)),
                        "closes_issues": json.dumps(list(r.closes_issues)),
                    }
                    for r in records
                ],
            )
            self._db.executemany(
                "INSERT OR REPLACE INTO pr_files VALUES (?,?,?,?)",
                [(r.number, f.path, f.additions, f.deletions) for r in records for f in r.files],
            )
        return len(records)

    def checkpoint(self, stream: HarvestStream, cursor: str | None, *, exhausted: bool) -> None:
        with self._db:
            self._db.execute(
                """INSERT INTO harvest_cursor VALUES (?,?,?,1)
                   ON CONFLICT(stream) DO UPDATE SET
                     cursor=excluded.cursor, exhausted=excluded.exhausted,
                     pages=harvest_cursor.pages+1""",
                (stream, cursor, int(exhausted)),
            )

    # ---- reads ----

    def cursor_state(self, stream: HarvestStream) -> tuple[str | None, bool, int]:
        """Return `(cursor, exhausted, pages)` for a stream; fresh streams are `(None, False, 0)`."""
        row = self._db.execute(
            "SELECT cursor, exhausted, pages FROM harvest_cursor WHERE stream=?", (stream,)
        ).fetchone()
        if row is None:
            return None, False, 0
        return row["cursor"], bool(row["exhausted"]), row["pages"]

    def size(self) -> int:
        return int(self._db.execute("SELECT count(*) FROM pull_requests").fetchone()[0])

    def state_counts(self) -> dict[str, int]:
        return {
            row["state"]: row["n"]
            for row in self._db.execute(
                "SELECT state, count(*) AS n FROM pull_requests GROUP BY state"
            )
        }

    def numbers(self) -> set[int]:
        return {int(r[0]) for r in self._db.execute("SELECT number FROM pull_requests")}

    def merge_commit_index(self) -> dict[str, int]:
        """Map every merged PR's merge commit to its PR number.

        Kept separate from `iter_records` because attributing history needs only
        two columns, while `iter_records` also materializes every PR's file list —
        millions of rows on a repo this size.
        """
        return {
            row["merge_commit_sha"]: int(row["number"])
            for row in self._db.execute(
                "SELECT number, merge_commit_sha FROM pull_requests "
                "WHERE state='MERGED' AND merge_commit_sha IS NOT NULL"
            )
        }

    def records_by_number(self, numbers: Iterable[int]) -> dict[int, PullRequestRecord]:
        """Fetch specific PRs with their files.

        Chain emission needs the handful of PRs in one chain, not the whole
        corpus. Scanning everything per chain turns emission into an O(chains ×
        corpus) operation.
        """
        wanted = sorted(set(numbers))
        if not wanted:
            return {}
        placeholders = ",".join("?" * len(wanted))
        params = tuple(wanted)
        files: dict[int, list[ChangedFile]] = {n: [] for n in wanted}
        for row in self._db.execute(
            f"SELECT number, path, additions, deletions FROM pr_files "
            f"WHERE number IN ({placeholders}) ORDER BY number, path",
            params,
        ):
            files[row["number"]].append(
                ChangedFile(
                    path=row["path"], additions=row["additions"], deletions=row["deletions"]
                )
            )
        return {
            row["number"]: _row_to_record(row, tuple(files[row["number"]]))
            for row in self._db.execute(
                f"SELECT * FROM pull_requests WHERE number IN ({placeholders})", params
            )
        }

    def iter_records(
        self,
        *,
        state: PullRequestState | None = None,
        order: str = "merged_at",
    ) -> Iterator[PullRequestRecord]:
        """Stream records, optionally filtered by state, joined with their files."""
        if order not in ("merged_at", "created_at", "number"):
            raise ValueError(f"unsupported order {order!r}")
        where = "WHERE state=?" if state else ""
        params: tuple[object, ...] = (state,) if state else ()
        rows = self._db.execute(
            f"SELECT * FROM pull_requests {where} ORDER BY {order}, number", params
        ).fetchall()
        by_number = {r["number"]: r for r in rows}
        files: dict[int, list[ChangedFile]] = {n: [] for n in by_number}
        for fr in self._db.execute(
            "SELECT number, path, additions, deletions FROM pr_files ORDER BY number, path"
        ):
            bucket = files.get(fr["number"])
            if bucket is not None:
                bucket.append(
                    ChangedFile(
                        path=fr["path"], additions=fr["additions"], deletions=fr["deletions"]
                    )
                )
        for row in rows:
            yield _row_to_record(row, tuple(files[row["number"]]))

    def prs_touching(self, path: str) -> list[int]:
        return [
            int(r[0]) for r in self._db.execute("SELECT number FROM pr_files WHERE path=?", (path,))
        ]

    def path_frequencies(self, *, state: PullRequestState | None = None) -> dict[str, int]:
        """Count how many PRs touch each path — the raw signal for file coupling."""
        join = "JOIN pull_requests p USING(number) WHERE p.state=?" if state else ""
        params: tuple[object, ...] = (state,) if state else ()
        return {
            r["path"]: r["n"]
            for r in self._db.execute(
                f"SELECT f.path AS path, count(*) AS n FROM pr_files f {join} "
                "GROUP BY f.path ORDER BY n DESC",
                params,
            )
        }


def _row_to_record(row: sqlite3.Row, files: tuple[ChangedFile, ...]) -> PullRequestRecord:
    return PullRequestRecord(
        number=row["number"],
        title=row["title"],
        body=row["body"],
        state=row["state"],
        created_at=row["created_at"],
        closed_at=row["closed_at"],
        merged_at=row["merged_at"],
        is_draft=bool(row["is_draft"]),
        url=row["url"],
        base_ref=row["base_ref"],
        base_sha=row["base_sha"],
        head_sha=row["head_sha"],
        merge_commit_sha=row["merge_commit_sha"],
        author=row["author"],
        additions=row["additions"],
        deletions=row["deletions"],
        changed_file_count=row["changed_file_count"],
        files=files,
        files_truncated=bool(row["files_truncated"]),
        labels=tuple(json.loads(row["labels"])),
        closes_issues=tuple(json.loads(row["closes_issues"])),
    )


def harvest_stream(
    owner: str,
    name: str,
    corpus: PRCorpus,
    *,
    stream: HarvestStream,
    token: str | None = None,
    page_size: int = 100,
    max_pages: int | None = None,
    stop_when: Callable[[PRCorpus], bool] | None = None,
    on_progress: Callable[[HarvestProgress], None] | None = None,
) -> HarvestProgress:
    """Harvest one direction of a repo's PR list until exhausted or stopped.

    `stream='asc'` walks oldest-first, `'desc'` newest-first. Running both
    concurrently halves wall time on a large repo; they meet in the middle and
    `stop_when` (typically "the two streams have covered every number") ends
    them. Progress is checkpointed after each page, so this is safe to kill.
    """
    if stream not in ("asc", "desc"):
        raise ValueError(f"stream must be 'asc' or 'desc', got {stream!r}")
    cursor, exhausted, pages = corpus.cursor_state(stream)
    stored_total = 0
    progress = HarvestProgress(
        stream=stream,
        pages=pages,
        stored=0,
        corpus_size=corpus.size(),
        rate_limit_remaining=-1,
        exhausted=exhausted,
    )
    while not exhausted:
        if max_pages is not None and pages - progress.pages >= max_pages:
            break
        page = harvest_pull_request_page(
            owner,
            name,
            after=cursor,
            page_size=page_size,
            oldest_first=stream == "asc",
            token=token,
        )
        stored_total += corpus.store(page.records)
        cursor = page.end_cursor
        exhausted = not page.has_next_page
        corpus.checkpoint(stream, cursor, exhausted=exhausted)
        pages += 1
        progress = HarvestProgress(
            stream=stream,
            pages=pages,
            stored=stored_total,
            corpus_size=corpus.size(),
            rate_limit_remaining=page.rate_limit_remaining,
            exhausted=exhausted,
        )
        if on_progress is not None:
            on_progress(progress)
        if stop_when is not None and stop_when(corpus):
            break
    return progress
