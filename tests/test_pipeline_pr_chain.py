"""Unit tests for the pr_chain pipeline.

Docker is not driven here — that is covered by end-to-end runs against real
repos. These tests pin the invariants a chain environment depends on:

  * subsystem attribution, including test-path mirroring
  * the stage partition is GAPLESS — the property that makes a chain replayable
  * carry absorption, and barriers when carry exceeds budget
  * anchors reject formatting sweeps and backports
  * chain selection honours the action floor, stage minimum and coherence floor
  * disjoint chains are preferred, and overlap is admitted only when needed
  * the emitted Dockerfile stays under the kernel's argv limit
  * the reward is the mean of the per-stage scores
"""

from __future__ import annotations

import io
import itertools
import json
import tarfile

import pytest

from repo2rlenv.git_local import FirstParentCommit
from repo2rlenv.pipelines._pr_chain_graph import (
    AnchorLimits,
    CarryLimits,
    Chain,
    ChainShape,
    ChainStage,
    HistoryStep,
    build_chains,
    partition_into_segments,
    subsystem_of,
)
from repo2rlenv.pipelines._pr_chain_steps import build_chain_steps, step_name
from repo2rlenv.pipelines._pr_chain_validate import (
    ChainValidation,
    StageValidation,
    _StageTrees,
)
from repo2rlenv.pipelines.pr_chain import (
    PRChainPipeline,
    build_chain_plan,
    build_stage_instruction,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def make_step(
    sha: str,
    *,
    parent: str,
    subject: str = "fix(cron): thing",
    source: tuple[str, ...] = ("cron/scheduler.py",),
    tests: tuple[str, ...] = ("tests/cron/test_scheduler.py",),
    lines: int = 100,
    pr: int | None = 1,
) -> HistoryStep:
    return HistoryStep(
        commit=FirstParentCommit(
            sha=sha,
            parent_sha=parent,
            subject=subject,
            committed_at="2026-01-01T00:00:00Z",
            is_merge=False,
        ),
        source_paths=source,
        test_paths=tests,
        lines_changed=lines,
        pr_number=pr,
    )


def linear_steps(count: int, **kwargs) -> list[HistoryStep]:
    """`count` consecutive anchor steps: c1 parented on c0, c2 on c1, ..."""
    return [make_step(f"c{i + 1}", parent=f"c{i}", pr=100 + i, **kwargs) for i in range(count)]


def make_stage(index: int, *, source_count: int = 2, subsystem: str = "cron") -> ChainStage:
    return ChainStage(
        index=index,
        pr_number=1000 + index,
        title=f"fix: thing {index}",
        before_commit=f"b{index}",
        carry_commit=f"b{index}",
        after_commit=f"a{index}",
        carry_shas=(),
        carry_lines_changed=0,
        source_paths=tuple(f"src/f{index}_{n}.py" for n in range(source_count)),
        test_paths=(f"tests/test_{index}.py",),
        lines_changed=50,
        subsystem=subsystem,
    )


# ---------------------------------------------------------------------------
# subsystem attribution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("cron/scheduler.py", "cron"),
        # A test path maps onto the subsystem it exercises, not onto "tests",
        # so a stage's source and test changes agree on one subsystem.
        ("tests/cron/test_scheduler.py", "cron"),
        ("tests/gateway/relay/test_x.py", "gateway"),
        # Monorepo roots hold many components; the component is one level deeper.
        ("apps/desktop/src/main.ts", "apps/desktop"),
        ("packages/core/index.ts", "packages/core"),
        # A bare root file has no subsystem.
        ("README.md", "(root)"),
        ("", "(root)"),
    ],
)
def test_subsystem_of(path: str, expected: str) -> None:
    assert subsystem_of(path) == expected


# ---------------------------------------------------------------------------
# stage partition
# ---------------------------------------------------------------------------


def test_partition_is_gapless() -> None:
    """Stage k must start exactly where stage k-1 ended.

    This is the property that makes a chain replayable: any gap means stage k
    begins from a tree that never existed in history.
    """
    segments = partition_into_segments(linear_steps(5), AnchorLimits(), CarryLimits())
    assert len(segments) == 1
    stages = segments[0]
    assert len(stages) == 5
    for previous, current in itertools.pairwise(stages):
        assert current.before_commit == previous.after_commit


def test_non_anchor_steps_are_carried_not_dropped() -> None:
    """A docs-only commit becomes free carry, and the partition stays gapless."""
    steps = [
        make_step("c1", parent="c0", pr=1),
        # No test change -> cannot anchor a stage.
        make_step("c2", parent="c1", source=("docs/x.md",), tests=(), pr=2),
        make_step("c3", parent="c2", pr=3),
    ]
    stages = partition_into_segments(steps, AnchorLimits(), CarryLimits())[0]
    assert len(stages) == 2
    carried = stages[1]
    assert carried.carry_shas == ("c2",)
    assert carried.has_carry
    # before -> carry covers the free churn; carry -> after is the graded goal.
    assert carried.before_commit == "c1"
    assert carried.carry_commit == "c2"
    assert carried.after_commit == "c3"


def test_carry_over_budget_creates_a_barrier() -> None:
    """An unshippable carry splits the history instead of bloating a stage.

    The anchor that would have carried the oversized churn (c3) is dropped with
    it: shipping its stage would mean shipping a 500k-line diff into the task.
    The next stage still starts from a real commit, so nothing is left dangling.
    """
    steps = [
        make_step("c1", parent="c0", pr=1),
        make_step("c2", parent="c1", source=("uv.lock",), tests=(), lines=500_000, pr=2),
        make_step("c3", parent="c2", pr=3),
        make_step("c4", parent="c3", pr=4),
    ]
    segments = partition_into_segments(steps, AnchorLimits(), CarryLimits())
    assert [len(s) for s in segments] == [1, 1]
    # No chain spans the barrier, and the stage after it starts at a real commit.
    assert segments[0][-1].after_commit == "c1"
    assert segments[1][0].before_commit == "c3"
    assert segments[1][0].after_commit == "c4"
    assert segments[1][0].carry_shas == ()


@pytest.mark.parametrize(
    "subject",
    [
        "fmt(js): `npm run fix` on merge",
        "style: reformat everything",
        "chore(lint): ruff format",
        "backport: fix from stable",
        'Revert "feat: thing"',
    ],
)
def test_unusable_subjects_do_not_anchor_a_stage(subject: str) -> None:
    """Formatting sweeps and backports cannot be milestones.

    Reproducing a formatter's exact output is not a solvable objective, and such
    a change moves no test, so no fail-to-pass oracle can exist for it.
    """
    steps = [make_step("c1", parent="c0", subject=subject, pr=1)]
    assert partition_into_segments(steps, AnchorLimits(), CarryLimits()) == []


def test_anchor_requires_both_source_and_test_changes() -> None:
    source_only = [make_step("c1", parent="c0", tests=(), pr=1)]
    test_only = [make_step("c1", parent="c0", source=(), pr=1)]
    assert partition_into_segments(source_only, AnchorLimits(), CarryLimits()) == []
    assert partition_into_segments(test_only, AnchorLimits(), CarryLimits()) == []


def test_require_pr_link_rejects_unattributed_steps() -> None:
    steps = [make_step("c1", parent="c0", pr=None)]
    assert partition_into_segments(steps, AnchorLimits(require_pr_link=True), CarryLimits()) == []
    assert partition_into_segments(steps, AnchorLimits(require_pr_link=False), CarryLimits())


# ---------------------------------------------------------------------------
# chain selection
# ---------------------------------------------------------------------------


def test_chain_needs_both_floor_and_stage_minimum() -> None:
    """One large commit clearing the floor is not a long-horizon environment."""
    segment = (make_stage(1, source_count=200),)
    selection = build_chains(
        [segment], shape=ChainShape(min_action_floor=100, min_stages=8), target_count=5
    )
    assert selection.chains == ()
    assert selection.rejected_short_horizon == 1


def test_selected_chain_meets_the_declared_floor() -> None:
    segment = tuple(make_stage(i + 1, source_count=3) for i in range(40))
    selection = build_chains(
        [segment],
        shape=ChainShape(min_action_floor=100, min_stages=8),
        target_count=1,
        overlap_ladder=(0.0,),
    )
    (chain,) = selection.chains
    assert chain.action_floor >= 100
    assert len(chain.stages) >= 8
    # Stages are renumbered per chain so a task's stages read 1..N.
    assert [s.index for s in chain.stages] == list(range(1, len(chain.stages) + 1))


def test_first_rung_yields_only_disjoint_chains() -> None:
    """Rung 0.0 must never let two chains share a stage."""
    segment = tuple(make_stage(i + 1, source_count=3) for i in range(80))
    disjoint = build_chains(
        [segment],
        shape=ChainShape(min_action_floor=100, min_stages=8),
        target_count=100,
        overlap_ladder=(0.0,),
    )
    assert disjoint.max_stage_reuse == 1
    assert disjoint.overlap_fraction_used == 0.0
    # 80 stages at floor 5 each fit four 20-stage chains and no more.
    assert len(disjoint.chains) == 4


def test_overlap_rung_is_entered_only_when_the_target_is_unmet() -> None:
    """The ladder must not spend overlap it does not need.

    A request the segment satisfies disjointly stays on rung 0.0; a request it
    cannot climbs, and the reported rung says which was used.
    """
    segment = tuple(make_stage(i + 1, source_count=3) for i in range(80))
    satisfied = build_chains(
        [segment],
        shape=ChainShape(min_action_floor=100, min_stages=8),
        target_count=2,
        overlap_ladder=(0.0, 0.5),
    )
    assert len(satisfied.chains) == 2
    assert satisfied.overlap_fraction_used == 0.0

    unsatisfied = build_chains(
        [segment],
        shape=ChainShape(min_action_floor=100, min_stages=8),
        target_count=100,
        overlap_ladder=(0.0, 0.5),
    )
    assert unsatisfied.overlap_fraction_used == 0.5
    # Even the top rung refuses a window whose history is already fully claimed,
    # so the yield is capped rather than filled with duplicate chains.
    assert len(unsatisfied.chains) >= len(satisfied.chains)
    assert unsatisfied.max_stage_reuse == 1


def test_coherence_floor_filters_scattered_windows() -> None:
    mixed = tuple(make_stage(i + 1, source_count=3, subsystem=f"sub{i % 8}") for i in range(40))
    assert (
        build_chains(
            [mixed], shape=ChainShape(min_action_floor=100, min_stages=8, min_coherence=0.9)
        ).chains
        == ()
    )
    assert build_chains(
        [mixed], shape=ChainShape(min_action_floor=100, min_stages=8, min_coherence=0.0)
    ).chains


def test_overlap_fraction_must_be_a_fraction() -> None:
    with pytest.raises(ValueError, match="overlap fractions"):
        build_chains([(make_stage(1),)], overlap_ladder=(1.0,))


# ---------------------------------------------------------------------------
# plan + payload
# ---------------------------------------------------------------------------


def _chain_of(stage_count: int) -> Chain:
    stages = tuple(make_stage(i + 1) for i in range(stage_count))
    return Chain(
        base_commit=stages[0].before_commit,
        head_commit=stages[-1].after_commit,
        stages=stages,
        subsystem="cron",
        coherence=1.0,
    )


def _validation_of(chain: Chain, *, unverified: set[int] = frozenset()) -> ChainValidation:
    return ChainValidation(
        status="verified",
        stages=[
            StageValidation(
                index=stage.index,
                status="no_oracle" if stage.index in unverified else "verified",
                test_cmds=["pytest -v"],
                fail_to_pass=[] if stage.index in unverified else [f"tests/t.py::s{stage.index}"],
                pass_to_pass=["tests/t.py::keep"],
            )
            for stage in chain.stages
        ],
    )


def test_plan_drops_unverified_stages_and_renumbers() -> None:
    """A stage whose change moved no test cannot be graded, so it is not shipped."""
    chain = _chain_of(5)
    plan = build_chain_plan(
        chain,
        _validation_of(chain, unverified={2, 4}),
        repo="o/n",
        stage_instructions={i: f"do {i}" for i in range(1, 6)},
        base_test_cmds=["pytest -v"],
    )
    assert [s["index"] for s in plan["stages"]] == [1, 2, 3]
    assert [s["after_commit"] for s in plan["stages"]] == ["a1", "a3", "a5"]
    assert all(s["fail_to_pass"] for s in plan["stages"])


def _tar_of(files: dict[str, str]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        for name, content in files.items():
            data = content.encode("utf-8")
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# instructions
# ---------------------------------------------------------------------------


def test_stage_instruction_strips_solution_leaks() -> None:
    stage = make_stage(1)
    text = build_stage_instruction(
        stage,
        title="fix: drop duplicate replies",
        body="Cherry-pick 0123456789abcdef from #4321 and rerun tests/test_x.py",
    )
    assert "fix: drop duplicate replies" in text
    assert "0123456789abcdef" not in text


# ---------------------------------------------------------------------------
# reward
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# contract
# ---------------------------------------------------------------------------


def test_pipeline_declares_bootstrap_and_pull_requests() -> None:
    from repo2rlenv.sources import Capability

    assert PRChainPipeline.requires_bootstrap is True
    assert Capability.PULL_REQUESTS in PRChainPipeline.required_capabilities
    assert PRChainPipeline.name.value == "pr_chain"


def test_pipeline_refuses_to_run_without_a_bootstrap_image() -> None:
    """Stages are graded by running the repo's tests, which needs the image."""
    with pytest.raises(RuntimeError, match="requires a BootstrapResult"):
        PRChainPipeline(input=None, options=None, bootstrap=None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# strict oracle conditions
# ---------------------------------------------------------------------------


def _trees(base: str, start: str, gold: str, head: str) -> _StageTrees:
    """One test's status on each of the four trees the reward depends on."""
    return _StageTrees(
        base={"t": base},
        start={"t": start},
        gold={"t": gold},
        head={"t": head},
    )


def test_fail_to_pass_requires_failing_at_the_chain_base() -> None:
    """A test already passing at the base pays a do-nothing agent.

    This is the defect that scored an untouched tree 0.111 on hermes-agent: the
    oracle was derived from stage-era test files but graded with the head's, so
    tests that were broken in their old form already passed in their final one.
    """
    passing_at_base = _trees("PASSED", "FAILED", "PASSED", "PASSED")
    assert not passing_at_base.is_fail_to_pass("t")


def test_fail_to_pass_requires_failing_where_the_stage_opens() -> None:
    """A test already passing when the stage opens asks for no work."""
    assert not _trees("FAILED", "PASSED", "PASSED", "PASSED").is_fail_to_pass("t")


def test_fail_to_pass_requires_passing_at_the_stage_gold_tree() -> None:
    """`chain submit` gates on the stage's own change, so that change must fix it."""
    assert not _trees("FAILED", "FAILED", "FAILED", "PASSED").is_fail_to_pass("t")


def test_fail_to_pass_requires_passing_at_the_chain_head() -> None:
    """The reward is computed at the final tree, so an unreachable test is no oracle.

    Shipping one docked the gold patch to 0.991 instead of 1.0.
    """
    assert not _trees("FAILED", "FAILED", "PASSED", "FAILED").is_fail_to_pass("t")


def test_fail_to_pass_accepts_a_test_broken_at_the_start_and_fixed_at_the_end() -> None:
    assert _trees("FAILED", "FAILED", "PASSED", "PASSED").is_fail_to_pass("t")


def test_a_test_absent_at_the_start_counts_as_failing() -> None:
    """A test the chain adds is the commonest fail-to-pass shape."""
    added = _StageTrees(base={}, start={}, gold={"t": "PASSED"}, head={"t": "PASSED"})
    assert added.is_fail_to_pass("t")


def test_pass_to_pass_requires_passing_everywhere() -> None:
    assert _trees("PASSED", "PASSED", "PASSED", "PASSED").is_pass_to_pass("t")
    for broken in range(4):
        statuses = ["PASSED"] * 4
        statuses[broken] = "FAILED"
        assert not _trees(*statuses).is_pass_to_pass("t")


def test_fail_to_pass_and_pass_to_pass_are_disjoint() -> None:
    """No test may be both an objective and a regression guard."""
    for combo in itertools.product(["PASSED", "FAILED"], repeat=4):
        trees = _trees(*combo)
        assert not (trees.is_fail_to_pass("t") and trees.is_pass_to_pass("t"))


# ---------------------------------------------------------------------------
# instruction quality gate
# ---------------------------------------------------------------------------


def test_plan_drops_stages_without_a_real_problem_statement() -> None:
    """A stage whose PR carried only a title gives the agent nothing to work from."""
    chain = _chain_of(4)
    instructions = {
        1: "**fix: thing** " + " ".join(["word"] * 30),
        2: "**fix: bare title only**",
        3: "**fix: another** " + " ".join(["word"] * 30),
        4: "**short**",
    }
    plan = build_chain_plan(
        chain,
        _validation_of(chain),
        repo="o/n",
        stage_instructions=instructions,
        base_test_cmds=["pytest -v"],
        min_instruction_words=12,
    )
    assert [s["after_commit"] for s in plan["stages"]] == ["a1", "a3"]
    assert [s["index"] for s in plan["stages"]] == [1, 2]


def test_plan_keeps_every_stage_when_the_gate_is_off() -> None:
    chain = _chain_of(3)
    plan = build_chain_plan(
        chain,
        _validation_of(chain),
        repo="o/n",
        stage_instructions=dict.fromkeys((1, 2, 3), "tiny"),
        base_test_cmds=["pytest -v"],
    )
    assert len(plan["stages"]) == 3


# ---------------------------------------------------------------------------
# native Harbor steps — one stage becomes one environment transition
# ---------------------------------------------------------------------------


def _plan_of(stage_count: int, monkeypatch) -> tuple[dict, Chain]:
    chain = _chain_of(stage_count)
    monkeypatch.setattr(
        "repo2rlenv.pipelines._pr_chain_steps.file_at_commit",
        lambda clone_dir, commit, path: f"# {path} @ {commit}\n",
    )
    monkeypatch.setattr(
        "repo2rlenv.pipelines._pr_chain_steps.range_diff",
        lambda clone_dir, before, after: f"diff {before}..{after}\n",
    )
    plan = build_chain_plan(
        chain,
        _validation_of(chain),
        repo="o/n",
        stage_instructions={i: f"objective {i} " + "word " * 20 for i in range(1, stage_count + 1)},
        base_test_cmds=["pytest -v"],
    )
    return plan, chain


def test_every_stage_becomes_one_harbor_step(monkeypatch, tmp_path) -> None:
    """Step count IS stage count — that is what makes the horizon countable.

    Harbor runs one agent phase and one verifier per entry in `steps`, so a
    100-stage chain is a 100-transition environment by construction rather than
    by an estimate of how many actions an agent might take.
    """
    plan, _ = _plan_of(5, monkeypatch)
    steps = build_chain_steps(
        plan,
        clone_dir=tmp_path,
        verifier_source="# verifier\n",
        language="python",
        agent_timeout_sec=3600.0,
        verifier_timeout_sec=900.0,
    )
    assert len(steps) == 5
    assert [s.name for s in steps] == [step_name(i) for i in range(1, 6)]


def test_each_step_ships_its_own_oracle_and_graded_tests(monkeypatch, tmp_path) -> None:
    """A step carries everything its transition needs, and nothing from others."""
    plan, _ = _plan_of(3, monkeypatch)
    steps = build_chain_steps(
        plan,
        clone_dir=tmp_path,
        verifier_source="# verifier\n",
        language="python",
        agent_timeout_sec=3600.0,
        verifier_timeout_sec=900.0,
    )
    first = steps[0]
    assert json.loads(first.aux_files["tests/f2p.json"]) == ["tests/t.py::s1"]
    assert "tests/verifier.py" in first.aux_files
    # The graded tests are copied over the tree, so weakening them cannot help.
    assert 'cp "$SCRIPT_DIR/files/$rel"' in first.test_script
    # Gold lives under solution/, which Harbor uploads for the oracle agent only.
    assert first.solve_script is not None
    assert "solution/patch.diff" in first.aux_files
    # No other stage's material leaks into this step.
    assert not any("stage-002" in key or "stage-003" in key for key in first.aux_files)


def test_carry_is_applied_by_step_setup_not_asked_of_the_agent(monkeypatch, tmp_path) -> None:
    """Unrelated churn is replayed in `workdir/setup.sh` before the agent runs."""
    plan, _ = _plan_of(3, monkeypatch)
    steps = build_chain_steps(
        plan,
        clone_dir=tmp_path,
        verifier_source="",
        language="python",
        agent_timeout_sec=3600.0,
        verifier_timeout_sec=900.0,
    )
    # Stage 1 opens on the chain base, so it has nothing carried.
    assert "No carried history" in steps[0].aux_files["workdir/setup.sh"]
    # Later stages resume from the previous stage's gold commit.
    later = steps[1]
    assert "workdir/carry.diff" in later.aux_files
    assert "git apply" in later.aux_files["workdir/setup.sh"]


def test_checkpoint_steps_abort_a_hopeless_run(monkeypatch, tmp_path) -> None:
    """`min_reward` stops paying for agent invocations that earn nothing.

    It is periodic on purpose: failing one milestone is normal and partial credit
    is the point of averaging, so gating every step would throw away good runs.
    """
    plan, _ = _plan_of(6, monkeypatch)
    steps = build_chain_steps(
        plan,
        clone_dir=tmp_path,
        verifier_source="",
        language="python",
        agent_timeout_sec=3600.0,
        verifier_timeout_sec=900.0,
        checkpoint_every=3,
    )
    gated = [s.name for s in steps if s.min_reward is not None]
    assert gated == [step_name(3), step_name(6)]
    assert all(s.min_reward == 0.01 for s in steps if s.min_reward is not None)

    ungated = build_chain_steps(
        plan,
        clone_dir=tmp_path,
        verifier_source="",
        language="python",
        agent_timeout_sec=3600.0,
        verifier_timeout_sec=900.0,
        checkpoint_every=0,
    )
    assert all(s.min_reward is None for s in ungated)


def test_writer_emits_the_layout_harbor_discovers(tmp_path, monkeypatch) -> None:
    """Harbor finds steps by path convention, so the paths are the contract."""
    from repo2rlenv.emitter.harbor import HarborTask, write_harbor_task

    plan, _chain = _plan_of(2, monkeypatch)
    steps = build_chain_steps(
        plan,
        clone_dir=tmp_path,
        verifier_source="# v\n",
        language="python",
        agent_timeout_sec=3600.0,
        verifier_timeout_sec=900.0,
    )
    task = HarborTask(
        name="t",
        org="o",
        description="d",
        instruction="top-level briefing",
        oracle_diff="diff",
        repo2env={},
        environment_dockerfile="FROM x\n",
        steps=steps,
        multi_step_reward_strategy="mean",
    )
    path = write_harbor_task(task, tmp_path / "out")
    for index in (1, 2):
        step_dir = path / "steps" / step_name(index)
        assert (step_dir / "instruction.md").is_file()
        assert (step_dir / "tests" / "test.sh").is_file()
        assert (step_dir / "solution" / "solve.sh").is_file()
        assert (step_dir / "workdir" / "setup.sh").is_file()
    import tomllib

    config = tomllib.loads((path / "task.toml").read_text())
    assert config["multi_step_reward_strategy"] == "mean"
    assert [s["name"] for s in config["steps"]] == [step_name(1), step_name(2)]
