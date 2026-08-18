# `pr_chain`

Long-horizon environments: **one task per contiguous run of a repository's
history**, split into milestones that each end on a real pull request. The agent
works stage by stage through an in-container `chain` command; the reward is the
mean of the per-stage `f2p_rate × p2p_rate` scores, recomputed from scratch
against the tree the agent leaves behind.

Every other pipeline here emits a single objective, which a competent agent
finishes in well under twenty actions. `pr_chain` exists for the behaviours that
only appear over a long horizon — planning across milestones, carrying context,
recovering from a mistake made twenty actions ago. A chain is emitted only when
its *minimum* solvable action count clears a declared floor (default 100).

| | |
|---|---|
| Status | **experimental (v0.9)** |
| Sandbox required at gen | Yes — Docker via the [bootstrap phase](../reference/BOOTSTRAP.md) |
| LLM required at gen | No |
| Reward kinds emitted | `test_execution` |
| Source | GitHub · GitLab (needs pull requests) |
| Design doc | [RFC 0011](../rfcs/0011-pr-chain.md) |

## Why chains are built from history, not from similar PRs

The obvious way to build a longer task is to pick several PRs that touch the same
subsystem and concatenate them. It does not work: **a PR's diff only applies to
the tree it was written against.** Pick PRs #100, #400 and #900 and #400's patch
will not apply on top of #100's, because the merges in between moved the code
underneath it.

`pr_chain` therefore uses the one ordering that is replayable by construction: the
**first-parent history** of the default branch. `git diff <c>^1 <c>` is exactly
the change the branch received at step `c`, whether the project squash-merges,
merge-commits or pushes directly. Replaying consecutive steps reproduces history,
so every intermediate tree is a real commit that really passed CI.

```
base ─[carry]─▶ c1^ ─[goal 1]─▶ c1 ─[carry]─▶ c2^ ─[goal 2]─▶ c2 ─▶ ...
```

- **Anchor** — a step whose diff touches both source *and* test files, so a
  fail-to-pass oracle can exist. Its change is the stage's **goal**: what the
  agent implements.
- **Carry** — steps that cannot anchor a stage (formatting sweeps, dependency
  bumps, docs). The environment applies these **for free** when the stage opens.

Carry keeps the partition gapless *and* fair: no history is skipped, so stage k+1
starts from a real commit, yet the agent is never asked to reproduce
`npm run fix` output. A carry over budget becomes a barrier that no chain crosses.

Coherence — the share of a chain's stages in one subsystem — is applied as a
**filter** (`min_coherence`), not as a ranking. Ranking by coherence strands short
runs between picks and loses chains without meaningfully raising coherence.

## What the agent sees

`instruction.md` publishes the protocol and the stage count, not the objectives.
Stages are revealed one at a time, so an agent cannot plan against the answer key:

```bash
chain status     # the current stage and what it must achieve
chain submit     # grade the current stage and open the next one
chain log        # what has been completed so far
```

`chain submit` is the environment's step function: it materializes the stage's
tests, runs the targeted command, reports which tests failed, and advances.
`chain submit --force` moves on without a full pass and keeps partial credit.

## Task shape

```
<owner>__<repo>-chain-<base8>-<N>st/
├── task.toml                       # Harbor metadata + [metadata.repo2env.pr_chain]
├── instruction.md                  # the protocol; stage objectives are withheld
├── environment/Dockerfile          # FROM <bootstrap_image>, positioned at base_commit
├── environment/docker-compose.yaml # egress guard
├── tests/test.sh                   # hands scoring to the in-image verifier
├── chain/plan.json                 # inspectable copy of the stage plan
└── solution/patch.diff             # the chain's whole diff (the oracle)
```

The chain payload — `plan.json`, the controller, the verifier and every stage's
test files — is baked into the image as base64. It is written across several `RUN`
steps because Linux caps a single argv string at 128 KiB, and a 27-stage chain
encodes to over 400 KiB; one `RUN` fails with `argument list too long`.

## Reward

```
stage_reward = f2p_rate × p2p_rate          (as in pr_runtime)
reward       = mean(stage_reward)
```

Averaging is deliberate: an agent that lands 14 of 25 milestones scores
meaningfully above one that lands 2, which is what makes the signal usable over
100+ steps. `/logs/verifier/reward-details.json` carries the per-stage breakdown
plus a `horizon` block (submissions, stages accepted).

Two properties matter:

- **The ledger is telemetry, not reward.** The agent owns every file in its own
  container, so the controller's records cannot be trusted for scoring. The
  terminal verifier recomputes everything.
- **Scoring happens at the final tree.** A stage is credited only if its behaviour
  is still present at the end, so work must accumulate — reverting an earlier
  stage to pass a later one loses the earlier stage's credit.

Because later history legitimately edits or deletes an earlier stage's tests, the
oracle sets are filtered at generation time to tests that still pass on the gold
head. Two copies of every test file ship: the stage-era copy gates the stage
mid-run, the head copy scores the end.

## Options

```bash
repo2rlenv generate --repo NousResearch/hermes-agent --pipeline pr_chain \
  --pipeline-opt limit=500 \
  --pipeline-opt min_action_floor=100 \
  --pipeline-opt min_coherence=0.5 \
  --force-language python \
  --out ./datasets/hermes-chain
```

| Option | Default | Meaning |
|---|---|---|
| `limit` | 500 | chains to emit |
| `min_action_floor` | 100 | fewest agent actions the chain can be solved in |
| `min_stages` | 8 | stage minimum, so one big commit is not a "chain" |
| `max_stages` | 40 | upper bound on stages per chain |
| `min_coherence` | 0.0 | share of stages in the dominant subsystem |
| `require_pr_link` | false | demand a resolvable PR number for every stage |
| `max_carry_steps` | 25 | free history the environment may replay per stage |
| `max_carry_lines` | 60000 | as above, by line count |
| `validation_margin` | 1.35 | headroom for stages that lose their oracle |
| `overlap_ladder` | `[0.0, 0.25, 0.5]` | rungs of admitted stage sharing |
| `agent_timeout_sec` | 28800 | one session must cover every stage |
| `verifier_timeout_sec` | 5400 | the verifier re-runs every stage's tests |

## Yield

Disjoint yield is arithmetic: `gated_stages / stages_per_chain`. Asking for more
climbs the overlap ladder rather than silently returning fewer chains, and the
rung actually used is reported in the run summary.

Measured on `NousResearch/hermes-agent` (65,944 PRs harvested; 11,055 merged):

| Quantity | Value |
|---|---|
| PRs harvested | 65,944 (11,055 merged · 33,160 closed · 21,729 open) |
| First-parent history steps | 19,072 · 11,467 resolve to a PR |
| Gated stages, PR-anchored (default) | 5,886 in 36 segments |
| Gated stages, best-effort attribution | 9,807 in 12 segments |
| Candidate windows at the padded floor | 4,100 · 344 clear `min_coherence=0.4` |
| Stages surviving validation | 20 of 24 on the measured chain (83%) |
| Emitted chain | 20 stages · floor 142 · `hermes_cli` · 467 F2P · 315 P2P |
| **Gold patch reward** | **1.000000** (20/20 stages resolved) |
| **Untouched tree reward** | **0.000000** (0/20 stages resolved) |

Validation costs four test runs per stage at roughly 8 s each on this repo, so a
24-stage chain takes about 13 minutes. It is embarrassingly parallel across
chains: `shard_count` workers with distinct `shard_index` values split one
deterministic selection between them.

With best-effort attribution (`require_pr_link=false`) and no coherence floor the
same repository supplies 423 chains at a 100-action floor, 411 of them sharing no
stage with any other. The disjoint ceiling is arithmetic —
`gated_stages / stages_per_chain` — and works out to 426, so the strict defaults
above trade most of that count for stages that all name a real PR and concentrate
on one subsystem.

## Anti-contamination

- The working tree sits at the chain's base commit and the git history is then
  scrubbed, so no future commit — and therefore no stage's fix — is reachable
  from `.git`. Verified on a built image: the chain head and stage-2 gold commits
  are both pruned and only `refs/heads/base` survives.
- The egress guard blackholes the fix-bearing hosts (PyPI, GitHub and their CDNs).
- Test files are **copied** over the tree before grading, so editing or deleting a
  test cannot raise the score.
- Stage test payloads and instructions enter the container; no stage's gold source
  diff does. This is the same disclosure `pr_runtime` makes.

## Environment requirements

**This is where a chain dataset is most likely to go quietly wrong.** A chain
resets the tree to historical commits, so the bootstrap image must satisfy the
**union** of what those commits need, not just HEAD's. When it does not, pytest
fails *collection* — which the log parser sees as a single error and the oracle
reads as "every test in this stage failed". The stage silently loses its oracle
and the chain quietly shrinks. Three concrete instances on hermes-agent:

1. **Eagerly-imported optional providers.** At older commits
   `tools/web_tools.py` imports `firecrawl` at module scope; HEAD made it lazy.
   The project's own `all` extra deliberately *excludes* those providers by
   policy, so `--extra all` does not help — `uv sync --all-extras` does. Fixing
   this alone took stage survival on the measured chain from 7/24 to 20/24.
2. **Interpreter version.** `.python-version` pins 3.11, and the `wake` extra
   pulls `tflite-runtime`, which ships cp311 wheels only. A 3.12 image cannot
   install the full extra set.
3. **Dropped pytest plugins.** Older `pyproject.toml` revisions set
   `addopts = "-m 'not integration' -n auto"` and pinned `pytest-xdist`, which
   HEAD later removed — a HEAD-only install dies on
   `pytest: error: unrecognized arguments: -n`. Install the plugins and pass
   `-n 0` in `BootstrapSpec.test_cmds` so output stays in the single-worker
   format the log parser reads.

Note also that test commands run through `bash -lc`, a login shell that re-reads
`/etc/profile` and discards `ENV PATH`. An image whose toolchain lives in a
virtualenv must expose it via `/etc/profile.d`, not `ENV PATH` alone.

The working recipe for this repository is in
[`workspace/hermes-agent.Dockerfile`](https://github.com/huggingface/Repo2RLEnv);
`BootstrapSpec.user_dockerfile` + `BootstrapSpec.test_cmds` feed it to the
bootstrap phase with no LLM agent involved.

## Limitations

- **Only merged PRs can gate a stage.** Open and closed-unmerged PRs have no
  verified merge, so no fail-to-pass oracle can be derived from them. They are
  harvested into the corpus and remain available for future diff-graded stages.
- **Coherence has a low ceiling at this horizon.** Clearing a 100-action floor
  takes 21-40 stages, and across 4,100 such windows the maximum
  dominant-subsystem share is 0.68. Do not set `min_coherence` above ~0.5 on this
  repository without measuring first.
- **PASS_TO_PASS is per-stage.** It is computed over the stage's own targeted test
  files, so a stage that adds a brand-new test file has none. That is why
  `min_pass_to_pass_per_stage` defaults to 0 and the count is reported instead.
- **Stage instructions are not rewritten.** A stage's objective is its PR title
  plus a leak-stripped body; `min_instruction_words` drops the ones with no real
  problem statement rather than synthesizing one.
