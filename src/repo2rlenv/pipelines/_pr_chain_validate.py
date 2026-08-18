"""Validate a chain's stages inside the bootstrap container.

Computes each stage's FAIL_TO_PASS and PASS_TO_PASS sets, which is what makes
the milestone gradeable at all. The protocol is simpler than `pr_runtime`'s
because a chain replays real history: there is no patch to apply and therefore
no patch that can fail to apply.

Per stage, inside one container:

  1. `git reset --hard <carry_commit>`  → the tree the agent starts from
     → run the stage's targeted tests → pre-status
  2. `git reset --hard <after_commit>`  → the gold tree
     → run the same tests           → post-status
  3. FAIL_TO_PASS = failing or absent in (1) and passing in (2)
     PASS_TO_PASS = passing in both

A stage with an empty FAIL_TO_PASS set has no oracle — its change did not move
any test — so it cannot be graded and the chain is rejected or trimmed. That
check is what stops a chain from silently containing unreachable milestones.

Because grading happens against the *final* tree, a stage's tests must still
pass there. Step 4 therefore re-runs every stage's tests at the chain head and
keeps only the tests that survive, which preserves the oracle invariant: real
history scores 1.0.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from repo2rlenv.bootstrap.docker import DockerSandbox
from repo2rlenv.log_parsers import parse_logs
from repo2rlenv.pipelines._pr_chain_graph import Chain, ChainStage
from repo2rlenv.pipelines.pr_runtime import (
    normalize_test_cmds_for_runtime,
    targeted_test_cmds_for_pr,
)
from repo2rlenv.pipelines.pr_runtime_validate import (
    _ensure_git,
    _slice_test_output,
)

logger = logging.getLogger(__name__)

PASSED = "PASSED"

_GIT_CLEAN = (
    "git clean -fdx -e .venv -e venv -e __pycache__ -e .tox "
    "-e node_modules -e target -e vendor -e .gradle -e .next -e .pytest_cache || true"
)


@dataclass(slots=True)
class StageValidation:
    """Per-stage oracle, or the reason the stage has none."""

    index: int
    status: str  # verified | no_oracle | unparseable
    test_cmds: list[str] = field(default_factory=list)
    fail_to_pass: list[str] = field(default_factory=list)
    pass_to_pass: list[str] = field(default_factory=list)
    reason: str = ""

    @property
    def verified(self) -> bool:
        return self.status == "verified"


@dataclass(slots=True)
class ChainValidation:
    """Outcome of validating a whole chain."""

    status: str  # verified | too_few_stages | git_unavailable | fetch_failed
    stages: list[StageValidation] = field(default_factory=list)
    reason: str = ""

    @property
    def verified_stages(self) -> list[StageValidation]:
        return [s for s in self.stages if s.verified]


def _stage_script(
    commit: str,
    test_cmds: list[str],
    *,
    tests_from: str,
    test_paths: tuple[str, ...],
) -> str:
    """Reset to a real commit, install a chosen version of the tests, and run them.

    `tests_from` decides which revision of the test files grades this run, and the
    choice is not cosmetic — the environment itself uses two different versions.
    Mid-run, `chain submit` gates a stage with the tests as they stood at that
    stage. At the end, the verifier scores with the tests as they stand at the
    chain head. A tree must therefore be checked with whichever version will
    actually be applied to it, or the oracle describes a situation that never
    occurs: grading every tree with the head's tests rejected stages whose final
    test file exercises behaviour a later stage introduces, and grading every tree
    with stage-era tests let a do-nothing agent collect 0.11.

    A test file absent at `tests_from` was deleted by then, so it is removed here
    too rather than left behind to be graded.
    """
    restore = [
        f"git checkout {tests_from} -- {path!r} 2>/dev/null || rm -f {path!r}"
        for path in test_paths
    ]
    return "\n".join(
        [
            "set -uxo pipefail",
            "cd /workspace",
            "git config --global --add safe.directory /workspace",
            f"git reset --hard {commit}",
            _GIT_CLEAN,
            *restore,
            # Markers go to STDOUT via echo: with `set -x`, a `:` no-op is traced
            # to STDERR while the runner writes STDOUT, and slicing between
            # stderr-only markers captures zero test lines.
            "echo R2E_START_TEST_OUTPUT",
            " && ".join(test_cmds) if test_cmds else "echo 'no test_cmds'",
            "echo R2E_END_TEST_OUTPUT",
        ]
    )


def fetch_chain_range(
    sandbox: DockerSandbox,
    chain: Chain,
    *,
    timeout: int = 900,
) -> tuple[bool, str]:
    """Make every commit the chain replays reachable in the container.

    The bootstrap image ships a depth-1 clone, so none of the chain's history is
    present. Fetching the head commit with a depth that covers the chain's span
    pulls the whole range in one request; a full unshallow would move hundreds of
    megabytes for history the chain never touches.

    One sandbox validates many chains in a batch, so this has to stay correct on
    a clone that earlier chains already deepened. Whether a fetch command
    *succeeded* is not the question — `git fetch --unshallow` exits non-zero on an
    already-complete repository — so success is decided only by whether the
    commits are present at the end.
    """
    if not _ensure_git(sandbox):
        return False, "git unavailable in sandbox"

    def missing_commits() -> list[str]:
        return [
            commit
            for commit in (chain.base_commit, chain.head_commit)
            if not sandbox.exec(
                f"git -C /workspace cat-file -e {commit} 2>/dev/null && echo OK", timeout=15
            ).stdout.count("OK")
        ]

    if not missing_commits():
        return True, ""  # an earlier chain in this sandbox already brought them in

    # One commit per stage plus its carry, doubled as headroom: `--depth` counts
    # along every parent, so merge commits in the range cost more than one.
    span = sum(1 + len(stage.carry_shas) for stage in chain.stages)
    depth = max(64, span * 2 + 32)
    sandbox.exec(
        f"cd /workspace && git fetch --depth {depth} origin {chain.head_commit}",
        timeout=timeout,
    )
    missing = missing_commits()
    if missing:
        logger.warning(
            "chain %s still missing %s after a depth-%d fetch; deepening",
            chain.head_commit[:12],
            [c[:12] for c in missing],
            depth,
        )
        sandbox.exec(
            "cd /workspace && "
            "{ git rev-parse --is-shallow-repository | grep -qx true "
            "&& git fetch --unshallow origin; } "
            f"|| git fetch origin {chain.head_commit} "
            "|| git fetch --depth 5000 origin",
            timeout=timeout * 2,
        )
        missing = missing_commits()
    if missing:
        return False, f"commits absent after fetch: {[c[:12] for c in missing]}"
    return True, ""


def stage_test_cmds(stage: ChainStage, base_test_cmds: list[str]) -> list[str]:
    """Narrow the repo's test command to the stage's own test files.

    A monorepo suite takes tens of minutes; the stage's own test files take
    seconds, and they are the only tests its oracle can reference.
    """
    normalized = normalize_test_cmds_for_runtime(base_test_cmds)
    return targeted_test_cmds_for_pr(normalized, list(stage.test_paths))


def _statuses(
    sandbox: DockerSandbox,
    commit: str,
    test_cmds: list[str],
    *,
    tests_from: str,
    test_paths: tuple[str, ...],
    language: str | None,
    timeout: int,
) -> dict[str, str]:
    result = sandbox.exec(
        _stage_script(commit, test_cmds, tests_from=tests_from, test_paths=test_paths),
        timeout=timeout,
    )
    log = result.truncated(max_chars=5_000_000)
    return parse_logs(test_cmds, _slice_test_output(log), language=language)


@dataclass(frozen=True, slots=True)
class _StageTrees:
    """A stage's per-test results on each tree the reward depends on.

    Naming them makes the oracle conditions readable: a fail-to-pass test is one
    that is broken where the agent starts and fixed where the agent finishes.
    """

    base: dict[str, str]
    start: dict[str, str]
    gold: dict[str, str]
    head: dict[str, str]

    def _passed(self, tree: dict[str, str], name: str) -> bool:
        return tree.get(name) == PASSED

    def is_fail_to_pass(self, name: str) -> bool:
        return (
            self._passed(self.gold, name)
            and self._passed(self.head, name)
            and not self._passed(self.start, name)
            and not self._passed(self.base, name)
        )

    def is_pass_to_pass(self, name: str) -> bool:
        return all(
            self._passed(tree, name) for tree in (self.base, self.start, self.gold, self.head)
        )


def _stage_tree_statuses(
    sandbox: DockerSandbox,
    chain: Chain,
    stage: ChainStage,
    cmds: list[str],
    *,
    language: str | None,
    timeout: int,
) -> _StageTrees | None:
    """Run a stage's tests on all four trees. None when the gold tree is unparseable.

    Each tree is graded with the test version that will really be applied to it:

    * `start` and `gold` use the STAGE-ERA tests, because `chain submit` gates the
      milestone with those files while the agent is working on it.
    * `base` and `head` use the HEAD tests, because the terminal verifier restores
      those before computing the reward.

    The gold tree runs first: if its output cannot be parsed there is no oracle to
    derive and the remaining runs would be wasted.
    """

    def at(commit: str, *, tests_from: str) -> dict[str, str]:
        return _statuses(
            sandbox,
            commit,
            cmds,
            tests_from=tests_from,
            test_paths=stage.test_paths,
            language=language,
            timeout=timeout,
        )

    gold = at(stage.after_commit, tests_from=stage.after_commit)
    if not gold:
        return None
    return _StageTrees(
        base=at(chain.base_commit, tests_from=chain.head_commit),
        start=at(stage.carry_commit, tests_from=stage.after_commit),
        gold=gold,
        head=at(chain.head_commit, tests_from=chain.head_commit),
    )


def validate_chain(
    *,
    sandbox: DockerSandbox,
    chain: Chain,
    base_test_cmds: list[str],
    language: str | None = None,
    min_stages: int = 8,
    max_pass_to_pass: int = 50,
    min_pass_to_pass: int = 1,
    timeout: int = 900,
) -> ChainValidation:
    """Derive every stage's oracle against all four trees the reward depends on.

    A test earns a place in a stage's FAIL_TO_PASS set only if it:

    1. **fails at the chain base** — otherwise an agent that does nothing collects
       credit for it, which was measured at 0.11 mean reward before this check
       existed;
    2. **fails at the stage's own start** — otherwise the milestone is already
       satisfied when it opens and asks for no work;
    3. **passes at the stage's gold tree** — otherwise the stage's own change is
       not what makes it pass, and `chain submit` could never accept the stage;
    4. **passes at the chain head** — otherwise it cannot pass at the final tree,
       where the reward is actually computed, and the gold patch would score
       below 1.0.

    PASS_TO_PASS must pass on all four. Every run installs the *head's* test files
    first, because those are the files the reward is computed with.

    Returns `status="verified"` when at least `min_stages` stages have a usable
    oracle. Stages without one are reported individually so the caller can trim
    the chain rather than discard it.
    """
    ok, reason = fetch_chain_range(sandbox, chain, timeout=timeout)
    if not ok:
        return ChainValidation(status="fetch_failed", reason=reason)

    stages: list[StageValidation] = []
    for stage in chain.stages:
        if not stage.test_paths:
            # Without a path to target, the repo's bare test command would run the
            # whole suite — minutes of work that can produce no per-stage oracle.
            stages.append(
                StageValidation(
                    index=stage.index,
                    status="no_oracle",
                    reason="no test file of this stage exists at both its gold tree and the head",
                )
            )
            continue
        cmds = stage_test_cmds(stage, base_test_cmds)
        trees = _stage_tree_statuses(
            sandbox,
            chain,
            stage,
            cmds,
            language=language,
            timeout=timeout,
        )
        if trees is None:
            stages.append(
                StageValidation(
                    index=stage.index,
                    status="unparseable",
                    test_cmds=cmds,
                    reason="no per-test results parsed from the gold tree",
                )
            )
            continue

        fail_to_pass = sorted(name for name in trees.gold if trees.is_fail_to_pass(name))
        pass_to_pass = sorted(name for name in trees.gold if trees.is_pass_to_pass(name))
        if not fail_to_pass:
            stages.append(
                StageValidation(
                    index=stage.index,
                    status="no_oracle",
                    test_cmds=cmds,
                    pass_to_pass=pass_to_pass[:max_pass_to_pass],
                    reason="no test fails at the start and passes at the head",
                )
            )
            continue
        if len(pass_to_pass) < min_pass_to_pass:
            stages.append(
                StageValidation(
                    index=stage.index,
                    status="no_regression_guard",
                    test_cmds=cmds,
                    fail_to_pass=fail_to_pass,
                    reason=(
                        f"{len(pass_to_pass)} pass-to-pass test(s), need {min_pass_to_pass}: "
                        "nothing would catch this stage breaking existing behaviour"
                    ),
                )
            )
            continue

        stages.append(
            StageValidation(
                index=stage.index,
                status="verified",
                test_cmds=cmds,
                fail_to_pass=fail_to_pass,
                pass_to_pass=pass_to_pass[:max_pass_to_pass],
            )
        )

    usable = [s for s in stages if s.verified]
    if len(usable) < min_stages:
        return ChainValidation(
            status="too_few_stages",
            stages=stages,
            reason=f"{len(usable)} stage(s) have an oracle, need {min_stages}",
        )
    return ChainValidation(status="verified", stages=stages)
