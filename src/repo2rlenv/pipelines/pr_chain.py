"""pr_chain — many pull requests replayed as gated stages in one environment.

Every other pipeline here emits a *single* objective: fix one bug, implement one
function, patch one CVE. A competent agent finishes such a task in a handful of
actions, which makes the resulting environment useless for studying what happens
over a long horizon — planning, carrying context, recovering from a milestone
that went wrong twenty actions ago.

`pr_chain` emits one environment per contiguous run of a repository's history,
split into stages that each end on a real pull request. The agent advances
through them with an in-container `chain` command, and the reward is the mean
per-stage score, recomputed from scratch at the end.

How the pieces divide up:

* `_pr_chain_graph`   — which runs of history are replayable, and where the
                        stage boundaries fall.
* `_pr_chain_validate`— each stage's fail-to-pass oracle, derived by running the
                        real tests at the real commits.
* `_pr_chain_steps`    — renders each stage as a native Harbor step, so the
                        runtime supplies the observation/action/reward loop.

The gold source diff for a stage never enters the container. Test payloads and
instructions do, which is the same disclosure `pr_runtime` already makes (its
`test.sh` carries the test patch inline).

----------------------------------------------------------------------------
Acknowledgment
----------------------------------------------------------------------------
The fail-to-pass / pass-to-pass oracle shape and the "reset, run tests, compare
status maps" validation protocol come from:

  SWE-bench (Princeton NLP) — https://github.com/princeton-nlp/SWE-bench (MIT)
  Jimenez et al., "SWE-bench: Can Language Models Resolve Real-World GitHub
  Issues?", ICLR 2024.

The long-horizon, milestone-gated framing is our own. Implementation is
independent; no code is copied. Released under Apache-2.0.
----------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import ClassVar

from repo2rlenv.auth import auth_clone_url, resolve_repo_token
from repo2rlenv.bootstrap.docker import DockerSandbox
from repo2rlenv.bootstrap.spec import BootstrapResult
from repo2rlenv.emitter.harbor import HarborTask, write_harbor_task
from repo2rlenv.git_local import file_at_commit, range_diff
from repo2rlenv.pipelines._env_guard import egress_guard_compose, git_history_scrub
from repo2rlenv.pipelines._pr_chain_graph import (
    AnchorLimits,
    CarryLimits,
    Chain,
    ChainSelection,
    ChainShape,
    ChainStage,
    build_chains,
    partition_into_segments,
    read_history_steps,
)
from repo2rlenv.pipelines._pr_chain_steps import build_chain_steps
from repo2rlenv.pipelines._pr_chain_validate import (
    ChainValidation,
    StageValidation,
    stage_test_cmds,
    validate_chain,
)
from repo2rlenv.pipelines._pr_corpus import PRCorpus, corpus_path, harvest_stream
from repo2rlenv.pipelines.base import PipelineResult
from repo2rlenv.pipelines.pr_runtime import (
    _reflow_pr_body,
    _strip_info_leak,
    _word_count,
    normalize_test_cmds_for_runtime,
)
from repo2rlenv.sources import Capability
from repo2rlenv.spec.input import GenerationInput, PipelineName
from repo2rlenv.spec.options import PRChainOptions

logger = logging.getLogger(__name__)

PIPELINE_VERSION = "0.1.0"
CHAIN_ROOT = "/opt/rl_chain"

_GIT_CLEAN_EXCLUDES = (
    "-e .venv -e venv -e __pycache__ -e .tox -e node_modules "
    "-e target -e vendor -e .gradle -e .next -e .pytest_cache"
)


# ---------------------------------------------------------------------------
# Instructions
# ---------------------------------------------------------------------------


def build_stage_instruction(
    stage: ChainStage,
    *,
    title: str,
    body: str,
) -> str:
    """Describe one milestone without disclosing how it was implemented.

    A PR body is written by the person who already solved the problem, so it
    routinely names the fix, the commit to cherry-pick, or the grading test. The
    same leak strip `pr_runtime` applies is applied here, and what remains is the
    statement of intent.
    """
    clean_title = _strip_info_leak(title).strip() or stage.title
    clean_body = _reflow_pr_body(_strip_info_leak(body)).strip()
    if not clean_body:
        clean_body = "(no description was recorded for this change)"
    return f"**{clean_title}**\n\n{clean_body}"


def build_chain_instruction(
    chain: Chain,
    *,
    repo: str,
    stage_count: int,
) -> str:
    """The task-level briefing: the protocol, not the work.

    Stage objectives are deliberately withheld here and revealed by
    `chain status` one at a time. Publishing all of them up front turns a
    long-horizon task into a single large specification, and an agent that can
    read stage 20 before starting stage 1 plans against the answer key rather
    than against the repository.
    """
    return "\n".join(
        [
            f"# Sustained development on `{repo}`",
            "",
            f"You are working in `/workspace`, a checkout of `{repo}` at commit "
            f"`{chain.base_commit[:12]}`.",
            "",
            f"Ahead of you are **{stage_count} stages** of real development history for "
            f"the `{chain.subsystem}` area of this codebase. Each stage is one change that "
            "was actually made to this project, and each is graded by that change's own "
            "tests.",
            "",
            "## How to work",
            "",
            "```bash",
            "chain status     # show the current stage and what it must achieve",
            "chain submit     # grade the current stage and open the next one",
            "chain log        # review what you have completed so far",
            "```",
            "",
            "Start with `chain status`. Implement the stage in the repository, then run "
            "`chain submit`. If the stage's tests do not pass, `chain submit` tells you "
            "which ones failed so you can iterate. When a stage will not converge, "
            "`chain submit --force` moves on and keeps whatever partial credit you earned.",
            "",
            "## How you are scored",
            "",
            "Your reward is the mean of the per-stage scores, so **every stage you land "
            "adds to it** — a partial run is worth more than an abandoned one.",
            "",
            "Scoring happens against the repository as you finally leave it, and every "
            "stage's tests are re-run then. Work must therefore *accumulate*: do not undo "
            "an earlier stage to make a later one pass. Test files are restored from the "
            "project's own history before grading, so editing or deleting a test cannot "
            "raise your score.",
            "",
            "Some stages arrive with unrelated project churn already applied for you "
            "(formatting sweeps, dependency bumps). That is deliberate and needs no "
            "action from you.",
        ]
    )


def _with_runnable_test_paths(chain: Chain, clone_dir: Path) -> Chain:
    """Drop each stage's test paths that do not exist at both its gold tree and the head.

    A stage's test paths come from its diff, so they include files the PR deleted
    or renamed. Handing pytest a path that is not there makes it abort collection
    and report the entire invocation as a single error, which reads as "every test
    failed" and silently destroys the stage's oracle — 8 of 24 stages on a
    hermes-agent chain were lost to exactly this before the filter existed.

    Both trees must have the file: the stage's gold tree is where its oracle is
    derived, and the head is where the reward is computed.
    """
    resolved = tuple(
        replace(
            stage,
            test_paths=tuple(
                path
                for path in stage.test_paths
                if file_at_commit(clone_dir, stage.after_commit, path) is not None
                and file_at_commit(clone_dir, chain.head_commit, path) is not None
            ),
        )
        for stage in chain.stages
    )
    dropped = sum(
        len(before.test_paths) - len(after.test_paths)
        for before, after in zip(chain.stages, resolved, strict=True)
    )
    if dropped:
        logger.info("dropped %d unrunnable test path(s) across the chain", dropped)
    return replace(chain, stages=resolved)


# ---------------------------------------------------------------------------
# Chain payload (baked into the image)
# ---------------------------------------------------------------------------


def build_chain_plan(
    chain: Chain,
    validation: ChainValidation,
    *,
    repo: str,
    stage_instructions: dict[int, str],
    base_test_cmds: list[str],
    min_instruction_words: int = 0,
) -> dict[str, object]:
    """Assemble `plan.json`: everything the controller and verifier need.

    A stage is included only when it has both a working oracle and a real problem
    statement. A stage whose change moved no test cannot be graded; a stage whose
    PR carried only a title gives the agent nothing to work from. Either way,
    shipping it would ask for work that cannot be earned or cannot be understood.

    Dropping a stage must not leave a hole in the replay. Each kept stage's
    `before_commit` is therefore set to the previous KEPT stage's `after_commit`,
    so its carry covers everything that happened in between — including the work
    of the stages that were dropped. Without this the agent's tree diverges from
    history at the first dropped stage and every later stage's tests can fail
    through no fault of the agent, while the gold patch still scores 1.0 because
    it contains the whole chain's diff.
    """
    by_index = {v.index: v for v in validation.stages}
    stages: list[dict[str, object]] = []
    resume_from = chain.base_commit
    for stage in chain.stages:
        found = by_index.get(stage.index)
        instruction = stage_instructions[stage.index]
        if found is None or not found.verified or _word_count(instruction) < min_instruction_words:
            continue
        stages.append(
            {
                "index": len(stages) + 1,
                "pr_number": stage.pr_number,
                "title": stage.title,
                "instruction": instruction,
                "before_commit": resume_from,
                "carry_commit": stage.carry_commit,
                "after_commit": stage.after_commit,
                "source_paths": list(stage.source_paths),
                "test_paths": list(stage.test_paths),
                "test_cmds": found.test_cmds or normalize_test_cmds_for_runtime(base_test_cmds),
                "fail_to_pass": found.fail_to_pass,
                "pass_to_pass": found.pass_to_pass,
            }
        )
        resume_from = stage.after_commit
    return {
        "repo": repo,
        "base_commit": chain.base_commit,
        "head_commit": chain.head_commit,
        "subsystem": chain.subsystem,
        "coherence": round(chain.coherence, 4),
        "action_floor": sum(len(s["source_paths"]) + 2 for s in stages),
        "pr_numbers": [s["pr_number"] for s in stages if s["pr_number"] is not None],
        "stages": stages,
    }


def _module_source(module_file: str) -> str:
    return (Path(__file__).parent / module_file).read_text(encoding="utf-8")


def build_chain_dockerfile(
    bootstrap_image: str,
    chain: Chain,
    *,
    language: str | None,
) -> str:
    """Build `environment/Dockerfile` for a chain task.

    Nothing chain-specific is baked in any more. Harbor uploads each step's
    instruction, setup script, tests and solution itself, so the image only has
    to be the repository sitting at the chain's base commit with git and python
    available. An earlier revision embedded the whole chain as base64 across
    several `RUN` layers to dodge the 128 KiB `MAX_ARG_STRLEN` limit; native
    steps make all of that unnecessary.

    The working tree is positioned at the base commit and the git history is then
    scrubbed, so the future — including every stage's fix — is not readable from
    `.git`.
    """
    return "".join(
        [
            "# Auto-generated by Repo2RLEnv pr_chain\n",
            f"FROM {bootstrap_image}\n",
            "WORKDIR /workspace\n",
            "# Defensive: each step's verifier is Python, and git is needed to\n",
            "# position the tree. Bootstrap images for non-Python repos may ship neither.\n",
            "RUN (command -v git >/dev/null 2>&1 && "
            "[ -e /etc/ssl/certs/ca-certificates.crt ]) || \\\n"
            "    (apt-get update && apt-get install -y --no-install-recommends "
            "git ca-certificates \\\n"
            "     && rm -rf /var/lib/apt/lists/*) || \\\n"
            "    (apk add --no-cache git ca-certificates && update-ca-certificates) || true\n",
            "RUN command -v python3 >/dev/null 2>&1 || \\\n"
            "    (apt-get update && apt-get install -y --no-install-recommends python3 \\\n"
            "     && rm -rf /var/lib/apt/lists/*) || apk add --no-cache python3 || true\n",
            "# Bring the chain's history into the clone, then sit at its base commit.\n",
            "RUN git config --global --add safe.directory /workspace \\\n"
            f"    && git fetch --depth {_fetch_depth(chain)} origin {chain.head_commit} "
            "2>/dev/null \\\n"
            "       || git fetch --unshallow origin 2>/dev/null || true\n",
            f"RUN git reset --hard {chain.base_commit} && git clean -fdx {_GIT_CLEAN_EXCLUDES}\n",
            git_history_scrub(chain.base_commit),
            _path_env_line(language),
        ]
    )


def _fetch_depth(chain: Chain) -> int:
    span = sum(1 + len(stage.carry_shas) for stage in chain.stages)
    return max(64, span * 2 + 32)


def _path_env_line(language: str | None) -> str:
    """Persist toolchain paths for non-Python repos.

    A bootstrap agent often installs Go, Rust or Node outside `/usr/bin` without
    exporting `PATH` anywhere the controller's non-interactive shell will read.
    """
    extras = {
        "go": "/usr/local/go/bin:/root/go/bin",
        "rust": "/root/.cargo/bin",
        "node": "/usr/local/lib/node_modules/.bin",
        "java": "/usr/lib/jvm/default-java/bin",
    }
    prefix = extras.get((language or "").lower())
    return f'ENV PATH="{prefix}:${{PATH}}"\n' if prefix else ""


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class PRChainPipeline:
    """Long-horizon PR chains. Implements the `Pipeline` Protocol."""

    name: ClassVar[PipelineName] = PipelineName.PR_CHAIN
    required_capabilities: ClassVar[frozenset[Capability]] = frozenset({Capability.PULL_REQUESTS})
    requires_bootstrap: ClassVar[bool] = True
    experimental: ClassVar[bool] = True

    def __init__(
        self,
        input: GenerationInput,
        options: PRChainOptions,
        bootstrap: BootstrapResult | None = None,
    ):
        if bootstrap is None:
            raise RuntimeError(
                "pr_chain requires a BootstrapResult: its stages are graded by running "
                "the repository's own tests, which needs the bootstrap image"
            )
        self.input = input
        self.options = options
        self.bootstrap = bootstrap
        self._progress_cb = None

    def set_progress_callback(self, cb) -> None:
        self._progress_cb = cb

    def _emit_progress(self, name: str, outcome: str, reason: str = "") -> None:
        if self._progress_cb is not None:
            try:
                self._progress_cb(name=name, outcome=outcome, reason=reason)
            except Exception as exc:  # a reporting failure must not end generation
                logger.debug("progress callback failed: %s", exc)

    # ----- corpus + chain construction ---------------------------------------

    def ensure_corpus(self, owner: str, name: str, *, token: str | None) -> PRCorpus:
        """Open the repo's PR corpus, harvesting whatever is missing.

        Chain building is a global query over the whole PR list, so the corpus is
        persisted once per repo and reused. Harvest is resumable: each page is
        checkpointed, so an interrupted run continues instead of restarting.
        """
        corpus = PRCorpus(corpus_path(self.input.bootstrap.cache_dir, owner, name))
        if not self.options.harvest:
            return corpus
        for stream in ("asc", "desc"):
            _, exhausted, _ = corpus.cursor_state(stream)
            if exhausted:
                continue
            logger.info("harvesting %s PRs (%s stream), corpus at %d", name, stream, corpus.size())
            harvest_stream(owner, name, corpus, stream=stream, token=token)
        return corpus

    def select_chains(self, clone_dir: Path, corpus: PRCorpus) -> ChainSelection:
        opts = self.options
        steps = read_history_steps(clone_dir, corpus, ref=opts.ref)
        segments = partition_into_segments(
            steps,
            AnchorLimits(
                min_lines_changed=opts.min_lines_changed,
                max_lines_changed=opts.max_lines_changed,
                max_source_files=opts.max_source_files_per_stage,
                require_pr_link=opts.require_pr_link,
            ),
            CarryLimits(
                max_steps=opts.max_carry_steps,
                max_lines_changed=opts.max_carry_lines,
            ),
        )
        logger.info(
            "history: %d steps -> %d segments, %d gated stages",
            len(steps),
            len(segments),
            sum(len(s) for s in segments),
        )
        # Select against a padded floor: validation strips the stages whose change
        # moved no test, and the chain must still clear the real floor afterwards.
        # With no validation there is nothing to strip, so no padding is needed.
        target_steps = (
            opts.min_steps if opts.skip_validation else int(opts.min_steps * opts.step_margin)
        )
        if not 0 <= opts.shard_index < opts.shard_count:
            raise ValueError(
                f"shard_index {opts.shard_index} out of range for shard_count {opts.shard_count}"
            )
        # Select `limit` chains per shard so every worker emits a full batch, then
        # keep only this shard's stride. The selection is deterministic, so the
        # shards partition it with no overlap and no coordination.
        selection = build_chains(
            segments,
            shape=ChainShape(
                # One stage becomes one Harbor step, so the step target IS
                # the stage minimum. The action floor is left at 0: it was a
                # proxy for horizon that measurement showed to be 14x off, and
                # steps are now counted directly.
                min_action_floor=0,
                min_stages=target_steps,
                max_stages=opts.max_steps,
                min_coherence=opts.min_coherence,
            ),
            target_count=opts.limit * opts.shard_count,
            overlap_ladder=tuple(opts.overlap_ladder),
        )
        if opts.shard_count == 1:
            return selection
        mine = selection.chains[opts.shard_index :: opts.shard_count]
        logger.info(
            "shard %d/%d takes %d of %d selected chain(s)",
            opts.shard_index,
            opts.shard_count,
            len(mine),
            len(selection.chains),
        )
        return replace(selection, chains=mine)

    # ----- run loop -----------------------------------------------------------

    def run(self, out_dir: Path) -> PipelineResult:
        out_dir.mkdir(parents=True, exist_ok=True)
        owner, name = self.input.repo.owner_name
        token = resolve_repo_token(self.input.repo, self.input.auth)

        clone_dir = self._full_clone(owner, name, token=token)
        corpus = self.ensure_corpus(owner, name, token=token)
        selection = self.select_chains(clone_dir, corpus)
        logger.info(
            "selected %d chain(s) from %d candidate window(s); overlap rung %.2f",
            len(selection.chains),
            selection.windows_considered,
            selection.overlap_fraction_used,
        )

        emitted = 0
        skip_reasons: dict[str, int] = {}
        sandbox = None if self.options.skip_validation else self._start_validation_sandbox()
        try:
            for chain in selection.chains:
                task_id = self._task_id(owner, name, chain)
                if (out_dir / task_id / "task.toml").exists():
                    # Validating one chain costs four test runs per stage, so a
                    # batch that restarts must not redo the chains it already
                    # finished. Delete the task directory to force a rebuild.
                    skip_reasons["already_emitted"] = skip_reasons.get("already_emitted", 0) + 1
                    self._emit_progress(task_id, "skipped", "already_emitted")
                    continue
                outcome = self._emit_chain(
                    chain,
                    corpus,
                    clone_dir,
                    out_dir,
                    task_id=task_id,
                    sandbox=sandbox,
                )
                if outcome is None:
                    emitted += 1
                    self._emit_progress(task_id, "emitted")
                else:
                    skip_reasons[outcome] = skip_reasons.get(outcome, 0) + 1
                    self._emit_progress(task_id, "skipped", outcome)
        finally:
            if sandbox is not None:
                sandbox.cleanup()
            corpus.close()

        return PipelineResult(
            candidates=len(selection.chains),
            emitted=emitted,
            skipped=len(selection.chains) - emitted,
            out_dir=out_dir,
            skip_reasons=skip_reasons,
        )

    def _task_id(self, owner: str, name: str, chain: Chain) -> str:
        return f"{owner}__{name}-chain-{chain.base_commit[:8]}-{len(chain.stages)}st"

    def _full_clone(self, owner: str, name: str, *, token: str | None) -> Path:
        """Maintain a bare mirror with full history.

        The bootstrap clone is depth-1 and cannot answer any question about
        history, so chain building keeps its own mirror per repo. A bare clone is
        enough: every read here is `git log`, `git diff` or `git show`, none of
        which needs a working tree.
        """
        mirror = self.input.bootstrap.cache_dir / "chain_mirror" / f"{owner}__{name}.git"
        if (mirror / "HEAD").exists():
            logger.info("reusing chain mirror at %s", mirror)
            return mirror
        mirror.parent.mkdir(parents=True, exist_ok=True)
        url = auth_clone_url(self.input.repo.url, token)
        logger.info("cloning %s/%s mirror (full history)", owner, name)
        proc = subprocess.run(
            ["git", "clone", "--bare", url, str(mirror)],
            capture_output=True,
            text=True,
            timeout=3600,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"chain mirror clone failed: {proc.stderr.strip()[:400]}")
        return mirror

    def _start_validation_sandbox(self):
        """Start one container from the bootstrap image, shared by every chain.

        Validation only ever runs `git reset` and the test suite, so one
        long-lived container serves all chains and avoids paying image start-up
        per chain.
        """
        marker = Path(tempfile.mkdtemp(prefix="r2e-pr-chain-"))
        (marker / ".keep").write_text("")
        return DockerSandbox.start(
            base_image=self.bootstrap.image_tag,
            repo_dir=marker,
            platform=self.input.bootstrap.platform,
        )

    def _stage_instructions(self, chain: Chain, corpus: PRCorpus) -> dict[int, str]:
        """Build every stage's objective text, sourced from its PR when known."""
        records = corpus.records_by_number(chain.pr_numbers)
        instructions: dict[int, str] = {}
        for stage in chain.stages:
            record = records.get(stage.pr_number) if stage.pr_number else None
            instructions[stage.index] = build_stage_instruction(
                stage,
                title=record.title if record else stage.title,
                body=record.body if record else "",
            )
        return instructions

    def _emit_chain(
        self,
        chain: Chain,
        corpus: PRCorpus,
        clone_dir: Path,
        out_dir: Path,
        *,
        task_id: str,
        sandbox: DockerSandbox | None,
    ) -> str | None:
        """Validate and write one chain. Returns a skip reason, or None on success."""
        opts = self.options
        chain = _with_runnable_test_paths(chain, clone_dir)
        if sandbox is None:
            validation = self._unvalidated(chain)
        else:
            validation = validate_chain(
                sandbox=sandbox,
                chain=chain,
                base_test_cmds=self.bootstrap.test_cmds,
                language=self.bootstrap.language.value,
                min_stages=opts.min_steps,
                max_pass_to_pass=opts.max_pass_to_pass_per_stage,
                min_pass_to_pass=opts.min_pass_to_pass_per_stage,
                timeout=opts.validation_timeout_sec,
            )
        if validation.status != "verified":
            logger.info("chain %s skipped: %s (%s)", task_id, validation.status, validation.reason)
            return validation.status

        plan = build_chain_plan(
            chain,
            validation,
            repo="/".join(self.input.repo.owner_name),
            stage_instructions=self._stage_instructions(chain, corpus),
            base_test_cmds=self.bootstrap.test_cmds,
            min_instruction_words=opts.min_instruction_words,
        )
        stages = plan["stages"]
        assert isinstance(stages, list)
        floor = sum(len(s["source_paths"]) + 2 for s in stages)
        if len(stages) < opts.min_steps:
            # Dropping unverifiable stages can push a chain below the horizon it
            # was selected for; emitting it anyway would break the guarantee the
            # dataset makes about every task.
            return "below_min_steps_after_validation"

        write_harbor_task(
            self._build_task(chain, plan, clone_dir, task_id=task_id, floor=floor),
            out_dir,
        )
        return None

    def _unvalidated(self, chain: Chain) -> ChainValidation:
        """Derive stage shells with no oracle, for debugging the emission path.

        Emitted tasks carry empty fail-to-pass sets, so they score 0 and are not
        training data. `skip_validation` exists to exercise chain construction
        without paying for thousands of test runs.
        """
        return ChainValidation(
            status="verified",
            stages=[
                StageValidation(
                    index=stage.index,
                    status="verified",
                    test_cmds=stage_test_cmds(stage, self.bootstrap.test_cmds),
                )
                # A stage with no targetable test file would ship the repo's bare
                # test command, i.e. the whole suite. Match what validation does
                # with those stages so the debug path emits the same shape.
                for stage in chain.stages
                if stage.test_paths
            ],
        )

    def _build_task(
        self,
        chain: Chain,
        plan: dict[str, object],
        clone_dir: Path,
        *,
        task_id: str,
        floor: int,
    ) -> HarborTask:
        owner, name = self.input.repo.owner_name
        stages = plan["stages"]
        assert isinstance(stages, list)

        image_ref = (
            self.bootstrap.image_digest
            if self.bootstrap.pushed_to_registry
            else self.bootstrap.image_tag
        )
        dockerfile = build_chain_dockerfile(
            image_ref,
            chain,
            language=self.bootstrap.language.value,
        )
        steps = build_chain_steps(
            plan,
            clone_dir=clone_dir,
            verifier_source=_module_source("_pr_runtime_verifier.py"),
            language=self.bootstrap.language.value,
            agent_timeout_sec=self.options.step_agent_timeout_sec,
            verifier_timeout_sec=self.options.step_verifier_timeout_sec,
            checkpoint_every=self.options.hopeless_checkpoint_every,
        )
        # The oracle is the whole chain's diff: applying it lands the gold tree,
        # where every surviving stage test passes, so the oracle agent scores 1.0.
        oracle_diff = range_diff(clone_dir, chain.base_commit, chain.head_commit)

        f2p_total = sum(len(s["fail_to_pass"]) for s in stages)
        repo2env = {
            "pipeline": "pr_chain",
            "pipeline_version": PIPELINE_VERSION,
            "repo": f"{owner}/{name}",
            "ref": chain.base_commit,
            "reference": f"https://github.com/{owner}/{name}/compare/"
            f"{chain.base_commit[:12]}...{chain.head_commit[:12]}",
            "source_access": self.input.repo.access,
            "built_at": datetime.now(UTC).isoformat(),
            "reward_kinds": ["test_execution"],
            "pr_chain": {
                "base_commit": chain.base_commit,
                "head_commit": chain.head_commit,
                "subsystem": chain.subsystem,
                "coherence": round(chain.coherence, 4),
                "step_count": len(stages),
                # TOML has no null: only stages with a resolved PR are listed.
                # `step_count` above is the authoritative total.
                "pr_numbers": [s["pr_number"] for s in stages if s["pr_number"] is not None],
                "reward_mode": "harbor_multi_step_mean",
                "bootstrap_image": self.bootstrap.image_digest,
            },
            "reward_calibration": {
                # One Harbor step per stage, so this is the environment's step
                # count: the number of observation/action/reward cycles it takes.
                "step_count": len(stages),
                "f2p_count": f2p_total,
                "p2p_count": sum(len(s["pass_to_pass"]) for s in stages),
                "min_agent_actions": floor,
                "difficulty": "hard",
            },
        }

        return HarborTask(
            name=task_id,
            org=self.input.output.org,
            description=(
                f"{len(stages)} sequential {chain.subsystem} changes from {owner}/{name} history"
            ),
            instruction=build_chain_instruction(
                chain, repo=f"{owner}/{name}", stage_count=len(stages)
            ),
            oracle_diff=oracle_diff,
            repo2env=repo2env,
            difficulty="hard",
            category="feature",
            keywords=[name, "pr_chain", "long_horizon", chain.subsystem],
            environment_dockerfile=dockerfile,
            steps=steps,
            multi_step_reward_strategy="mean",
            aux_files={
                "environment/docker-compose.yaml": egress_guard_compose(),
                "chain/plan.json": json.dumps(plan, indent=2),
            },
        )
