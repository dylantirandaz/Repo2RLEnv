"""Write Harbor-compliant task directories.

The minimal (text-only) path emits:
  task.toml + instruction.md + solution/patch.diff
No environment/, no tests/. Reward kind = "diff_similarity".

When a pipeline supplies `task.environment_dockerfile` + `task.test_script`
(pr_diff with emit_harbor_env=True, and all _runtime pipelines), the writer
also emits environment/Dockerfile + tests/test.sh and seeds the
[metadata.repo2env.reproducibility] subtable.

----------------------------------------------------------------------------
Acknowledgment
----------------------------------------------------------------------------
The output FORMAT (task.toml schema, directory layout, /logs/verifier/reward.txt
contract, [metadata] tables) is defined by:

  Harbor Framework (Laude Institute / Terminal-Bench creators)
  https://github.com/harbor-framework/harbor    (Apache-2.0)
  https://www.harborframework.com/docs/tasks

We emit Harbor's format directly so any Harbor-compatible runtime, agent
harness, or downstream framework (OpenReward, SkyRL via Harbor, etc.) can
consume our datasets unchanged. We do NOT depend on the `harbor` Python
package — we generate the file format from scratch. The format itself is a
spec (data layout); using it does not require a license grant. Repo2RLEnv-
specific provenance lives inside Harbor's free-form `[metadata]` table under
the namespaced subtable `[metadata.repo2env]`.

Released under Apache-2.0.
----------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tomli_w


@dataclass(slots=True)
class HarborStep:
    """One step of a native Harbor multi-step task.

    Harbor drives the loop itself: it shows `instruction`, runs the agent, then
    runs `test_script` and records that step's reward before moving on. So a step
    is a real environment transition — observation, action, reward — rather than
    something a task-supplied controller simulates inside the container.

    `min_reward` aborts the remaining steps when this one scores below the
    threshold, which is Harbor's `terminated`. Leave it unset to let an agent that
    fails one milestone keep earning credit on later ones.
    """

    name: str
    instruction: str
    test_script: str
    solve_script: str | None = None
    aux_files: dict[str, str] = field(default_factory=dict)
    agent_timeout_sec: float | None = None
    verifier_timeout_sec: float = 900.0
    min_reward: float | None = None


@dataclass(slots=True)
class HarborTask:
    name: str
    org: str
    description: str
    instruction: str
    oracle_diff: str
    repo2env: dict[str, Any]
    difficulty: str = "medium"
    category: str = "bugfix"
    keywords: list[str] = field(default_factory=list)
    # Optional — only set for sandbox-required pipelines (e.g. pr_runtime).
    # Lite tasks (pr_diff) leave these as None; Harbor falls back to its own
    # default env / test runner for those.
    environment_dockerfile: str | None = None
    test_script: str | None = None
    # Harbor's per-task budgets. The defaults suit a single-fix task; a
    # long-horizon chain needs far more, because one session must cover every
    # stage and the verifier re-runs every stage's tests.
    agent_timeout_sec: float = 1800.0
    verifier_timeout_sec: float = 300.0
    # Extra files written under the task dir (relative path -> content), e.g.
    # {"tests/verifier.py": ..., "tests/f2p.json": ...}. Harbor exposes tests/
    # at /tests in the container so test.sh can read them.
    aux_files: dict[str, str] = field(default_factory=dict)
    # Native Harbor multi-step tasks. When non-empty the writer emits
    # `steps/<name>/{instruction.md,tests/test.sh,solution/solve.sh}` and a
    # `steps` array in task.toml, and Harbor runs one agent+verifier cycle per
    # step. `multi_step_reward_strategy` defaults to Harbor's own "mean".
    steps: list[HarborStep] = field(default_factory=list)
    multi_step_reward_strategy: str | None = None


def _content_hash(task: HarborTask) -> str:
    h = hashlib.sha256()
    h.update(task.instruction.encode("utf-8"))
    h.update(b"\0")
    h.update(task.oracle_diff.encode("utf-8"))
    return f"sha256:{h.hexdigest()}"


def _step_table(step: HarborStep) -> dict[str, Any]:
    """Render one `[[steps]]` entry for task.toml.

    Only non-default keys are emitted so the file stays readable at 100 steps.
    """
    table: dict[str, Any] = {
        "name": step.name,
        "verifier": {"timeout_sec": step.verifier_timeout_sec},
    }
    if step.agent_timeout_sec is not None:
        table["agent"] = {"timeout_sec": step.agent_timeout_sec}
    if step.min_reward is not None:
        table["min_reward"] = step.min_reward
    return table


def write_harbor_task(task: HarborTask, dest_dir: Path) -> Path:
    """Materialize the task directory at dest_dir/<task.name>. Returns the path."""
    task_path = dest_dir / task.name
    task_path.mkdir(parents=True, exist_ok=True)

    # task.toml
    repo2env = dict(task.repo2env)
    # v0.2.0: introduces [metadata.repo2env.reproducibility] subtable; the bump
    # is additive — old readers ignore the new subtable, new readers see it.
    repo2env.setdefault("spec_version", "0.2.0")
    repo2env.setdefault("content_hash", _content_hash(task))
    # Default reward kinds — sandbox-required tasks override with
    # test_execution as the primary signal
    if task.test_script is not None:
        repo2env.setdefault("reward_kinds", ["test_execution", "diff_similarity"])
    else:
        repo2env.setdefault("reward_kinds", ["diff_similarity"])

    # For sandbox-required tasks (those that emit environment/Dockerfile),
    # seed [metadata.repo2env.reproducibility] with mode=local_only and the
    # un-pullable local image ref. `repo2rlenv push` rewrites this in-place
    # to mode=registry / inline_dockerfile after the push step.
    if task.environment_dockerfile is not None and "reproducibility" not in repo2env:
        bootstrap_image = ""
        # Derive from the Dockerfile's first FROM line so we don't need the
        # caller to plumb it separately. Matches the same anchor used by
        # `registry.integration._FROM_LINE_RE`.
        m = re.search(
            r"^(\s*FROM\s+)(\S+)", task.environment_dockerfile, re.IGNORECASE | re.MULTILINE
        )
        if m:
            bootstrap_image = m.group(2).strip()
        repo2env["reproducibility"] = {
            "mode": "local_only",
            "image_ref": bootstrap_image or "local/r2e-bootstrap:unknown",
            "image_tag": bootstrap_image or "local/r2e-bootstrap:unknown",
            "image_visibility": "private",
        }

    # Harbor's task.toml requires `task.name` in `<org>/<name>` format —
    # validated at load-time by harbor.models.task.config.PackageInfo. We
    # keep the filesystem-safe slug (with `__` for path safety) as the
    # directory name, but emit the schema-required `org/slug` form in
    # task.toml so harbor accepts the task.
    qualified_name = f"{task.org}/{task.name}"
    payload: dict[str, Any] = {
        "version": "1.0",
        "task": {
            "name": qualified_name,
            "description": task.description,
        },
        "metadata": {
            "difficulty": task.difficulty,
            "category": task.category,
            "keywords": task.keywords,
            "repo2env": repo2env,
        },
        "agent": {"timeout_sec": task.agent_timeout_sec},
        "verifier": {"timeout_sec": task.verifier_timeout_sec},
    }
    if task.steps:
        payload["multi_step_reward_strategy"] = task.multi_step_reward_strategy or "mean"
        payload["steps"] = [_step_table(step) for step in task.steps]
    (task_path / "task.toml").write_bytes(tomli_w.dumps(payload).encode("utf-8"))

    # instruction.md
    (task_path / "instruction.md").write_text(task.instruction, encoding="utf-8")

    # solution/patch.diff — canonical SWE-bench-style oracle (what trainers consume)
    sol_dir = task_path / "solution"
    sol_dir.mkdir(exist_ok=True)
    (sol_dir / "patch.diff").write_text(task.oracle_diff, encoding="utf-8")

    # solution/solve.sh — Harbor's oracle agent runs this script inside the
    # container; it should leave the working tree in the "fixed" state. We
    # `git apply` the canonical patch.diff so we keep one oracle artifact
    # (patch.diff) and just provide the execution shim Harbor needs.
    (sol_dir / "solve.sh").write_text(
        "#!/bin/bash\n"
        "set -euxo pipefail\n"
        "cd /workspace\n"
        "git config --global --add safe.directory /workspace\n"
        # Harbor uploads the whole solution/ dir into the container under
        # /solution; the patch.diff sits next to this script.
        'PATCH="$(dirname "$0")/patch.diff"\n'
        'git apply --verbose --reject "$PATCH"\n',
        encoding="utf-8",
    )
    (sol_dir / "solve.sh").chmod(0o755)

    # Optional environment/Dockerfile + tests/test.sh — written only for
    # sandbox-required tasks (pr_runtime, future commit_runtime, etc.).
    if task.environment_dockerfile is not None:
        env_dir = task_path / "environment"
        env_dir.mkdir(exist_ok=True)
        (env_dir / "Dockerfile").write_text(task.environment_dockerfile, encoding="utf-8")
    if task.test_script is not None:
        tests_dir = task_path / "tests"
        tests_dir.mkdir(exist_ok=True)
        (tests_dir / "test.sh").write_text(task.test_script, encoding="utf-8")
        # mark executable; harbor expects test.sh to be runnable
        (tests_dir / "test.sh").chmod(0o755)

    # steps/<name>/{instruction.md, tests/test.sh, solution/solve.sh} — the
    # layout Harbor discovers for a native multi-step task. Each step's verifier
    # produces that step's reward, so one step is one environment transition.
    for step in task.steps:
        step_dir = task_path / "steps" / step.name
        (step_dir / "tests").mkdir(parents=True, exist_ok=True)
        (step_dir / "instruction.md").write_text(step.instruction, encoding="utf-8")
        test_path = step_dir / "tests" / "test.sh"
        test_path.write_text(step.test_script, encoding="utf-8")
        test_path.chmod(0o755)
        if step.solve_script is not None:
            solution_dir = step_dir / "solution"
            solution_dir.mkdir(parents=True, exist_ok=True)
            solve_path = solution_dir / "solve.sh"
            solve_path.write_text(step.solve_script, encoding="utf-8")
            solve_path.chmod(0o755)
        for rel_path, content in step.aux_files.items():
            target = (step_dir / rel_path).resolve()
            if not str(target).startswith(str(step_dir.resolve())):
                raise ValueError(f"step aux_file escapes step dir: {rel_path!r}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")

    # Auxiliary task files (relative paths under the task dir). Harbor mounts
    # the task's tests/ dir into the container at /tests, so a pipeline can
    # ship e.g. tests/verifier.py + tests/f2p.json + tests/p2p.json as plain,
    # inspectable artifacts and have test.sh read them — instead of baking
    # everything as base64 blobs inside test.sh.
    for rel_path, content in (task.aux_files or {}).items():
        # Defensive: keep aux files inside the task dir.
        target = (task_path / rel_path).resolve()
        if not str(target).startswith(str(task_path.resolve())):
            raise ValueError(f"aux_file path escapes task dir: {rel_path!r}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    return task_path
