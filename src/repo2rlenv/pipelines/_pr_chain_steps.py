"""Render a validated chain as native Harbor steps.

Harbor already implements the loop a long-horizon environment needs. For each
entry in a task's `steps` array it runs, in order:

    steps/<name>/workdir/setup.sh   the state the agent starts this step from
    steps/<name>/instruction.md     the observation
    <the agent phase>               the action
    steps/<name>/tests/test.sh      the transition, which returns the reward

`min_reward` aborts the remaining steps, and `multi_step_reward_strategy="mean"`
averages the per-step rewards into the trial reward. So one chain stage becomes
one real environment step, counted and scored by the runtime rather than
simulated by anything this project ships into the container.

Two consequences of grading in place, both of which simplify the task:

* A stage's tests are restored from files shipped with *that step*, so an agent
  only ever sees the tests for the milestone it is working on — not the whole
  chain's.
* The gold patch for a stage lives in `steps/<name>/solution/`, which Harbor
  uploads for the oracle agent only. Nothing reveals it to a normal run.
"""

from __future__ import annotations

import json
from pathlib import Path

from repo2rlenv.emitter.harbor import HarborStep
from repo2rlenv.git_local import file_at_commit, range_diff

# Reward below which a checkpoint step aborts the rest of the chain. A stage that
# earns literally nothing means the agent is not making contact with the task
# (cannot run the suite, cannot edit the tree), and the remaining steps would
# each cost another agent invocation for the same zero.
HOPELESS_STEP_REWARD = 0.01

# How often to place such a checkpoint. Every step would be far too strict —
# failing a single milestone is normal and partial credit is the whole point of
# averaging — so the gate is periodic.
CHECKPOINT_EVERY = 25


def step_name(index: int) -> str:
    """Zero-padded so `steps/` sorts correctly at three digits."""
    return f"stage-{index:03d}"


def build_step_setup_script(*, has_carry: bool) -> str:
    """`workdir/setup.sh` — put the tree into the state this step starts from.

    Carry is the project churn between the previous graded milestone and this one:
    formatting sweeps, dependency bumps, and any stage that validation could not
    grade. Applying it here rather than asking the agent for it keeps the replay
    gapless without inventing work.
    """
    if not has_carry:
        return "#!/bin/bash\n# No carried history for this step.\nexit 0\n"
    return (
        "#!/bin/bash\n"
        "set -uo pipefail\n"
        "cd /workspace\n"
        "git config --global --add safe.directory /workspace\n"
        'CARRY="$(dirname "$0")/carry.diff"\n'
        '[ -s "$CARRY" ] || exit 0\n'
        "# Tolerate hunks already present: an agent may have written equivalent\n"
        "# code, and a partly-applied carry is better than a failed step setup.\n"
        'git apply --verbose "$CARRY" \\\n'
        '  || git apply --verbose --3way "$CARRY" \\\n'
        '  || git apply --verbose --reject "$CARRY" \\\n'
        "  || true\n"
        "exit 0\n"
    )


def build_step_test_script(*, test_cmds: list[str], language: str | None) -> str:
    """`tests/test.sh` — restore this stage's tests, run them, emit the reward.

    Test files are *copied* from `tests/files/` rather than patched in. The agent
    may have rewritten or deleted them, and a copy is both deterministic and the
    anti-tamper guard: a stage cannot be passed by weakening its tests.
    """
    path_prelude = _path_prelude(language)
    joined = " && ".join(test_cmds) if test_cmds else "echo 'no test_cmds'"
    escaped = joined.replace("'", "'\\''")
    return (
        "#!/bin/bash\n"
        "set -uxo pipefail\n"
        f"{path_prelude}"
        'SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"\n'
        "cd /workspace\n"
        "git config --global --add safe.directory /workspace\n"
        "mkdir -p /logs/verifier\n"
        "# Restore the graded tests over whatever the agent left behind.\n"
        'if [ -d "$SCRIPT_DIR/files" ]; then\n'
        '  (cd "$SCRIPT_DIR/files" && find . -type f -print0) | \\\n'
        '    while IFS= read -r -d "" rel; do\n'
        '      mkdir -p "/workspace/$(dirname "$rel")"\n'
        '      cp "$SCRIPT_DIR/files/$rel" "/workspace/$rel"\n'
        "    done\n"
        "fi\n"
        f"( {joined} ) > /logs/verifier/test_output.log 2>&1\n"
        "TEST_EXIT_CODE=$?\n"
        "cat /logs/verifier/test_output.log\n"
        'python3 "$SCRIPT_DIR/verifier.py" \\\n'
        "  --log /logs/verifier/test_output.log \\\n"
        '  --f2p "$SCRIPT_DIR/f2p.json" --p2p "$SCRIPT_DIR/p2p.json" \\\n'
        f"  --test-cmds '{escaped}' --exit-code \"$TEST_EXIT_CODE\" \\\n"
        "  --out-dir /logs/verifier || \\\n"
        '  { [ "$TEST_EXIT_CODE" -eq 0 ] && echo "1.0" > /logs/verifier/reward.txt \\\n'
        '    || echo "0.0" > /logs/verifier/reward.txt; }\n'
        "# reward.txt is the verdict, not this script's exit code.\n"
        "exit 0\n"
    )


def build_step_solve_script() -> str:
    """`solution/solve.sh` — apply this stage's gold patch.

    Harbor uploads a step's `solution/` for the oracle agent only, so this is the
    one artifact that must never be readable during a normal run.
    """
    return (
        "#!/bin/bash\n"
        "set -euxo pipefail\n"
        "cd /workspace\n"
        "git config --global --add safe.directory /workspace\n"
        'PATCH="$(dirname "$0")/patch.diff"\n'
        'git apply --verbose --reject "$PATCH"\n'
    )


def _path_prelude(language: str | None) -> str:
    """Prepend known toolchain dirs for languages installed outside /usr/bin."""
    extras = {
        "go": ["/usr/local/go/bin", "$HOME/go/bin"],
        "rust": ["$HOME/.cargo/bin"],
        "node": ["/usr/local/lib/node_modules/.bin"],
        "java": ["/usr/lib/jvm/default-java/bin"],
    }
    dirs = extras.get((language or "").lower(), [])
    return f'export PATH="{":".join(dirs)}:$PATH"\n' if dirs else ""


def build_chain_steps(
    plan: dict[str, object],
    *,
    clone_dir: Path,
    verifier_source: str,
    language: str | None,
    agent_timeout_sec: float,
    verifier_timeout_sec: float,
    checkpoint_every: int = CHECKPOINT_EVERY,
) -> list[HarborStep]:
    """Turn a validated plan into one Harbor step per gated stage."""
    stages = plan["stages"]
    assert isinstance(stages, list)

    steps: list[HarborStep] = []
    for entry in stages:
        index = int(entry["index"])
        name = step_name(index)
        aux: dict[str, str] = {
            "tests/verifier.py": verifier_source,
            "tests/f2p.json": json.dumps(entry["fail_to_pass"], indent=2),
            "tests/p2p.json": json.dumps(entry["pass_to_pass"], indent=2),
        }

        has_carry = entry["before_commit"] != entry["carry_commit"]
        if has_carry:
            aux["workdir/carry.diff"] = range_diff(
                clone_dir, str(entry["before_commit"]), str(entry["carry_commit"])
            )
        aux["workdir/setup.sh"] = build_step_setup_script(has_carry=has_carry)

        # The graded tests, taken from this stage's gold tree.
        for rel_path in entry["test_paths"]:
            content = file_at_commit(clone_dir, str(entry["after_commit"]), str(rel_path))
            if content is not None:
                aux[f"tests/files/{rel_path}"] = content

        # The gold source patch for this stage only — oracle agent territory.
        aux["solution/patch.diff"] = range_diff(
            clone_dir, str(entry["carry_commit"]), str(entry["after_commit"])
        )

        steps.append(
            HarborStep(
                name=name,
                instruction=str(entry["instruction"]),
                test_script=build_step_test_script(
                    test_cmds=list(entry["test_cmds"]), language=language
                ),
                solve_script=build_step_solve_script(),
                aux_files=aux,
                agent_timeout_sec=agent_timeout_sec,
                verifier_timeout_sec=verifier_timeout_sec,
                min_reward=(
                    HOPELESS_STEP_REWARD
                    if checkpoint_every > 0 and index % checkpoint_every == 0
                    else None
                ),
            )
        )
    return steps
