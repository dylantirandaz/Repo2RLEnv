"""Build replayable long-horizon PR chains from a repository's merge history.

Chaining pull requests naively does not work. Pick PRs #100, #400 and #900
because they touch the same subsystem and #400's patch will not apply on top of
#100's: the merges in between moved the code underneath it. Any chain built that
way is unverifiable, so it is not an RL environment.

This module uses the one ordering that is replayable by construction: the
**first-parent history** of the default branch. `git diff <c>^1 <c>` is exactly
the change the branch received at step `c`, whether the project squash-merges,
merge-commits or pushes directly. Replaying consecutive first-parent steps
reproduces history, so every intermediate tree is a real commit that really
passed CI.

A chain is a contiguous run of history split into gated stages:

    base ─[carry]─▶ c1^ ─[goal 1]─▶ c1 ─[carry]─▶ c2^ ─[goal 2]─▶ c2 ─▶ ...

Each stage ends on an *anchor* — a step whose diff touches both source and test
files, so a fail-to-pass oracle can exist for it. The stage's **goal** is the
anchor's own change: that is what the agent must implement. Steps that cannot
anchor a stage (formatting bots, dependency bumps, docs) become the stage's
**carry**, which the environment applies for free when the stage opens.

Carry is what makes the partition both gapless and fair. Gapless, because no
history is skipped, so stage k+1 starts from a real commit. Fair, because the
agent is never asked to reproduce a formatting sweep, yet a later stage whose
tests depend on that sweep still sees it.

Coherence comes from *selecting* windows, never from reordering them: a window
whose anchors concentrate on one subsystem reads as one sustained piece of work.
"""

from __future__ import annotations

import logging
import re
import subprocess
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from repo2rlenv.git_local import FirstParentCommit, first_parent_history
from repo2rlenv.pipelines._pr_corpus import PRCorpus
from repo2rlenv.pipelines.pr_runtime import _is_non_bug_pr, _path_is_test

logger = logging.getLogger(__name__)

# `feat(cron): thing (#12345)` — GitHub's squash-merge subject suffix.
_SQUASH_PR_RE = re.compile(r"\(#(\d+)\)\s*$")
# `Merge pull request #12345 from fork/branch` — GitHub's merge-commit subject.
_MERGE_PR_RE = re.compile(r"^Merge pull request #(\d+)\b")

# Formatting and lint sweeps make unusable milestones: reproducing a
# formatter's exact output is not a solvable objective, and the change flips no
# test from failing to passing, so no fail-to-pass oracle exists. They are
# carried for free instead of gated.
_FORMATTING_SUBJECT_RE = re.compile(
    r"^(?:fmt|style|lint|format)\b|^chore\((?:fmt|style|lint|format)\)|"
    r"\b(?:npm run fix|ruff format|prettier --write|gofmt|cargo fmt)\b",
    re.IGNORECASE,
)

# Directory roots holding many independent components; the component name sits
# one level deeper, so `apps/desktop` is the subsystem rather than `apps`.
_MONOREPO_ROOTS = frozenset(
    {"apps", "packages", "libs", "services", "plugins", "crates", "cmd", "src", "modules"}
)
_TEST_ROOTS = frozenset({"tests", "test", "__tests__", "spec"})


def subsystem_of(path: str) -> str:
    """Return the subsystem a path belongs to.

    A test path maps onto the subsystem it exercises (`tests/gateway/x.py` →
    `gateway`) so a stage's source and test changes agree on one subsystem
    instead of splitting between `gateway` and `tests`.
    """
    parts = [p for p in path.split("/") if p]
    if not parts:
        return "(root)"
    if parts[0] in _TEST_ROOTS and len(parts) >= 2:
        parts = parts[1:]
    if parts[0] in _MONOREPO_ROOTS and len(parts) >= 3:
        return f"{parts[0]}/{parts[1]}"
    if len(parts) == 1:
        return "(root)"
    return parts[0]


def _dominant_subsystem(paths: Iterable[str]) -> str:
    counts = Counter(subsystem_of(p) for p in paths)
    if not counts:
        return "(root)"
    return counts.most_common(1)[0][0]


@dataclass(frozen=True, slots=True)
class HistoryStep:
    """One first-parent step with its diff shape resolved.

    `source_paths` and `test_paths` partition the step's changed files by the
    same rule `pr_runtime` uses, because that split decides what the agent must
    write versus what grades it.
    """

    commit: FirstParentCommit
    source_paths: tuple[str, ...]
    test_paths: tuple[str, ...]
    lines_changed: int
    pr_number: int | None

    @property
    def sha(self) -> str:
        return self.commit.sha


@dataclass(frozen=True, slots=True)
class AnchorLimits:
    """Bounds a history step must satisfy to anchor a gated stage.

    Every bound keeps a *verifiable* oracle reachable rather than making the
    task easier: a step with no test change has no fail-to-pass set, and a
    2000-line sweep is a mechanical refactor whose tests pass before and after.
    """

    min_lines_changed: int = 10
    max_lines_changed: int = 1500
    max_source_files: int = 20
    require_pr_link: bool = True


@dataclass(frozen=True, slots=True)
class CarryLimits:
    """Bounds the free, environment-applied history preceding a stage.

    Carry costs the agent nothing, so its only real cost is artifact size: one
    dependency bump can carry 300k lines, and that diff has to ship inside the
    task. A carry over budget becomes a barrier — chains are never built across
    one, which keeps both the replay guarantee and the task size sane.

    The bounds are deliberately loose. A tight `max_steps` shatters history into
    short segments and collapses the yield: on hermes-agent, dropping from 25 to
    6 cut usable chains from 417 to 75, because every run of unremarkable
    commits became a barrier instead of free setup.
    """

    max_steps: int = 25
    max_lines_changed: int = 60_000


def _is_anchor(step: HistoryStep, limits: AnchorLimits) -> bool:
    if not step.source_paths or not step.test_paths:
        return False
    if _is_non_bug_pr(step.commit.subject):
        return False
    if _FORMATTING_SUBJECT_RE.search(step.commit.subject):
        return False
    if not (limits.min_lines_changed <= step.lines_changed <= limits.max_lines_changed):
        return False
    if len(step.source_paths) > limits.max_source_files:
        return False
    return not (limits.require_pr_link and step.pr_number is None)


def read_history_steps(
    clone_dir: Path,
    corpus: PRCorpus | None,
    *,
    ref: str = "HEAD",
) -> list[HistoryStep]:
    """Read every first-parent step with its file split and PR attribution.

    One `git log --numstat` pass supplies the diff shape for the whole history;
    per-step `git diff` calls would be thousands of subprocesses.

    PR attribution prefers the corpus (`merge_commit_sha` is authoritative and
    covers squash, merge and rebase alike) and falls back to parsing the commit
    subject, which only catches GitHub's default subject formats.
    """
    order = first_parent_history(clone_dir, ref=ref)
    shapes = _read_diff_shapes(clone_dir, ref=ref)
    pr_by_commit = _pr_by_commit(corpus)

    steps: list[HistoryStep] = []
    for commit in order:
        shape = shapes.get(commit.sha)
        if shape is None:
            continue
        source_paths, test_paths, lines_changed = shape
        steps.append(
            HistoryStep(
                commit=commit,
                source_paths=source_paths,
                test_paths=test_paths,
                lines_changed=lines_changed,
                pr_number=pr_by_commit.get(commit.sha) or _pr_from_subject(commit.subject),
            )
        )
    return steps


def _pr_from_subject(subject: str) -> int | None:
    for pattern in (_SQUASH_PR_RE, _MERGE_PR_RE):
        match = pattern.search(subject)
        if match:
            return int(match.group(1))
    return None


def _pr_by_commit(corpus: PRCorpus | None) -> dict[str, int]:
    if corpus is None:
        return {}
    return corpus.merge_commit_index()


def _read_diff_shapes(
    clone_dir: Path, *, ref: str
) -> dict[str, tuple[tuple[str, ...], tuple[str, ...], int]]:
    """Return `{sha: (source_paths, test_paths, lines_changed)}` in one git pass."""
    proc = subprocess.run(
        ["git", "log", "--first-parent", "--numstat", "--no-renames", "--format=%x01%H", ref],
        cwd=str(clone_dir),
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"git log --numstat failed: {proc.stderr.strip()[:400]}")

    shapes: dict[str, tuple[tuple[str, ...], tuple[str, ...], int]] = {}
    for block in proc.stdout.split("\x01"):
        if not block.strip():
            continue
        header, _, body = block.partition("\n")
        sha = header.strip()
        if not sha:
            continue
        source: list[str] = []
        tests: list[str] = []
        lines_changed = 0
        for line in body.splitlines():
            fields = line.split("\t")
            if len(fields) != 3:
                continue
            added, deleted, path = fields
            # Binary files report "-" for both counts.
            lines_changed += (int(added) if added.isdigit() else 0) + (
                int(deleted) if deleted.isdigit() else 0
            )
            (tests if _path_is_test(path) else source).append(path)
        shapes[sha] = (tuple(source), tuple(tests), lines_changed)
    return shapes


@dataclass(frozen=True, slots=True)
class ChainStage:
    """One gated milestone.

    Three commits describe the stage, and the distinction between them is the
    whole point:

    * `before_commit` — the tree when the stage opens.
    * `carry_commit`  — after the environment applies the free, non-gated
      history. Equal to `before_commit` when there is nothing to carry.
    * `after_commit`  — the gold tree. `carry_commit..after_commit` is the
      anchor's own change: exactly what the agent must reproduce, and the diff
      that must never enter the container.
    """

    index: int
    pr_number: int | None
    title: str
    before_commit: str
    carry_commit: str
    after_commit: str
    carry_shas: tuple[str, ...]
    carry_lines_changed: int
    source_paths: tuple[str, ...]
    test_paths: tuple[str, ...]
    lines_changed: int
    subsystem: str

    @property
    def has_carry(self) -> bool:
        return self.before_commit != self.carry_commit

    @property
    def action_floor(self) -> int:
        """Fewest agent actions this stage admits: read + edit each file + submit."""
        return len(self.source_paths) + 2


@dataclass(frozen=True, slots=True)
class Chain:
    """A contiguous, replayable run of history partitioned into gated stages."""

    base_commit: str
    head_commit: str
    stages: tuple[ChainStage, ...]
    subsystem: str
    coherence: float

    @property
    def action_floor(self) -> int:
        return sum(stage.action_floor for stage in self.stages)

    @property
    def pr_numbers(self) -> tuple[int, ...]:
        return tuple(s.pr_number for s in self.stages if s.pr_number is not None)

    @property
    def lines_changed(self) -> int:
        return sum(s.lines_changed for s in self.stages)


def partition_into_segments(
    steps: list[HistoryStep],
    anchors: AnchorLimits,
    carry: CarryLimits,
) -> list[tuple[ChainStage, ...]]:
    """Fold history into gated stages, grouped into replayable runs.

    Returns *segments*: maximal runs of consecutive stages with no barrier
    between them. Only an over-budget carry creates a barrier, so segments are
    long — a non-anchor step is carried, never a reason to cut the history.

    Within a segment the partition is gapless: stage k's `before_commit` is
    stage k-1's `after_commit`, so replaying a segment reproduces history.
    """
    segments: list[tuple[ChainStage, ...]] = []
    current: list[ChainStage] = []
    pending: list[HistoryStep] = []

    for step in steps:
        if not _is_anchor(step, anchors):
            pending.append(step)
            continue
        carried_lines = sum(s.lines_changed for s in pending)
        if len(pending) > carry.max_steps or carried_lines > carry.max_lines_changed:
            # Too much free history to ship. Close the segment and restart
            # after it, so no chain is built across the gap.
            if current:
                segments.append(tuple(current))
                current = []
            pending = []
            continue
        current.append(_build_stage(step, pending, index=len(current) + 1))
        pending = []

    if current:
        segments.append(tuple(current))
    return segments


def _build_stage(
    anchor: HistoryStep,
    carried: list[HistoryStep],
    *,
    index: int,
) -> ChainStage:
    """Assemble the stage that `anchor` gates, carrying `carried` for free."""
    return ChainStage(
        index=index,
        pr_number=anchor.pr_number,
        title=anchor.commit.subject,
        before_commit=carried[0].commit.parent_sha if carried else anchor.commit.parent_sha,
        carry_commit=anchor.commit.parent_sha,
        after_commit=anchor.sha,
        carry_shas=tuple(s.sha for s in carried),
        carry_lines_changed=sum(s.lines_changed for s in carried),
        source_paths=anchor.source_paths,
        test_paths=anchor.test_paths,
        lines_changed=anchor.lines_changed,
        subsystem=_dominant_subsystem((*anchor.source_paths, *anchor.test_paths)),
    )


def _window_coherence(stages: tuple[ChainStage, ...]) -> tuple[str, float]:
    """Return the window's dominant subsystem and the share of stages in it."""
    counts = Counter(stage.subsystem for stage in stages)
    subsystem, hits = counts.most_common(1)[0]
    return subsystem, hits / len(stages)


@dataclass(frozen=True, slots=True)
class ChainShape:
    """How long and how coherent an emitted chain must be.

    `min_stages` is what makes the environment long-*horizon* rather than merely
    long: without it a single 500-line commit clears `min_action_floor` on its
    own and the result is one big task behind one gate.
    """

    min_action_floor: int = 100
    min_stages: int = 8
    max_stages: int = 40
    min_coherence: float = 0.0


@dataclass(frozen=True, slots=True)
class ChainSelection:
    """Chains chosen for emission plus the counters explaining the yield."""

    chains: tuple[Chain, ...]
    windows_considered: int
    rejected_short_horizon: int
    rejected_overlap: int
    overlap_fraction_used: float
    max_stage_reuse: int


def _grow_window(
    segment: tuple[ChainStage, ...],
    start: int,
    shape: ChainShape,
) -> tuple[ChainStage, ...] | None:
    """Grow a window from `start` until it satisfies `shape`, or fail.

    Growth stops at the first window meeting BOTH the action floor and the stage
    minimum, so windows stay as short as the requirements allow and more of the
    segment stays available for other chains.
    """
    floor = 0
    for end in range(start, min(start + shape.max_stages, len(segment))):
        floor += segment[end].action_floor
        length = end - start + 1
        if floor >= shape.min_action_floor and length >= shape.min_stages:
            window = segment[start : end + 1]
            _, coherence = _window_coherence(window)
            return window if coherence >= shape.min_coherence else None
    return None


def build_chains(
    segments: list[tuple[ChainStage, ...]],
    *,
    shape: ChainShape | None = None,
    target_count: int = 500,
    overlap_ladder: tuple[float, ...] = (0.0, 0.25, 0.5),
) -> ChainSelection:
    """Select up to `target_count` chains from replayable segments.

    Windows are taken in history order, which is what maximizes the count: a
    quality-ranked greedy pass strands runs shorter than `min_stages` between
    its picks and measurably loses chains (348 disjoint chains become 281 on
    hermes-agent) while barely moving median coherence. Coherence is therefore
    enforced as a floor in `ChainShape.min_coherence`, not used as a ranking.

    Overlap between chains is admitted in explicit rungs: the first pass takes
    only disjoint chains, and a later rung is entered *only* while the target is
    unmet. Disjoint yield is capped near `total_stages / stages_per_chain`, so a
    request the repo cannot satisfy disjointly is met with a stated, bounded
    amount of shared history rather than silently returning fewer chains.
    """
    shape = shape or ChainShape()
    for fraction in overlap_ladder:
        if not 0.0 <= fraction < 1.0:
            raise ValueError(f"overlap fractions must be in [0,1), got {fraction}")

    candidates: list[tuple[float, int, int, tuple[ChainStage, ...]]] = []
    rejected_short = 0
    for segment_index, segment in enumerate(segments):
        for start in range(len(segment)):
            window = _grow_window(segment, start, shape)
            if window is None:
                rejected_short += 1
                continue
            _, coherence = _window_coherence(window)
            candidates.append((coherence, segment_index, start, window))
    candidates.sort(key=lambda c: (c[1], c[2]))

    chosen: list[Chain] = []
    taken: set[tuple[int, int]] = set()
    claimed: Counter[tuple[int, int]] = Counter()
    rejected_overlap = 0
    fraction_used = overlap_ladder[0] if overlap_ladder else 0.0

    for fraction in overlap_ladder:
        if len(chosen) >= target_count:
            break
        fraction_used = fraction
        for coherence, segment_index, start, window in candidates:
            if len(chosen) >= target_count:
                break
            key = (segment_index, start)
            if key in taken:
                continue
            span = [(segment_index, start + offset) for offset in range(len(window))]
            overlap = sum(1 for cell in span if claimed[cell])
            if overlap > fraction * len(window):
                rejected_overlap += 1
                continue
            taken.add(key)
            chosen.append(_finalize_chain(window, coherence))
            for cell in span:
                claimed[cell] += 1

    return ChainSelection(
        chains=tuple(chosen),
        windows_considered=len(candidates),
        rejected_short_horizon=rejected_short,
        rejected_overlap=rejected_overlap,
        overlap_fraction_used=fraction_used,
        max_stage_reuse=max(claimed.values()) if claimed else 0,
    )


def _finalize_chain(window: tuple[ChainStage, ...], coherence: float) -> Chain:
    """Renumber a window's stages to 1..N and wrap it as a Chain."""
    stages = tuple(
        ChainStage(
            index=offset + 1,
            pr_number=stage.pr_number,
            title=stage.title,
            before_commit=stage.before_commit,
            carry_commit=stage.carry_commit,
            after_commit=stage.after_commit,
            carry_shas=stage.carry_shas,
            carry_lines_changed=stage.carry_lines_changed,
            source_paths=stage.source_paths,
            test_paths=stage.test_paths,
            lines_changed=stage.lines_changed,
            subsystem=stage.subsystem,
        )
        for offset, stage in enumerate(window)
    )
    subsystem, _ = _window_coherence(stages)
    return Chain(
        base_commit=stages[0].before_commit,
        head_commit=stages[-1].after_commit,
        stages=stages,
        subsystem=subsystem,
        coherence=coherence,
    )
