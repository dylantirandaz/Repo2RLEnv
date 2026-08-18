# RFC 0011: `pr_chain`

**Status:** implemented
**Created:** 2026-08-17

## Summary

One environment per **contiguous run of a repository's history**, split into
milestones that each end on a real pull request. The agent advances stage by
stage through an in-container `chain` command; the terminal verifier recomputes
the reward from scratch against the tree the agent leaves, averaging the
per-stage `f2p_rate × p2p_rate` scores. Long-horizon by construction: a chain is
only emitted when its *minimum* solvable action count clears a declared floor
(default 100).

## Motivation

Every existing pipeline emits a **single** objective — fix one bug, implement one
function, patch one CVE. A competent agent finishes such a task in well under
twenty actions. That makes the resulting environments useless for studying the
behaviours that only appear over a long horizon: planning across milestones,
carrying context, and recovering from a mistake made twenty actions ago.

The obvious way to build a longer task is to pick several PRs that touch the same
subsystem and concatenate them. That does not work, and the reason is worth
recording: **a PR's diff only applies to the tree it was written against.** Pick
PRs #100, #400 and #900 from one subsystem and #400's patch will not apply on top
of #100's, because the merges in between moved the code underneath it. Any chain
built that way is unverifiable, so it is not an RL environment.

## Design

### The one replayable ordering

`pr_chain` uses the **first-parent history** of the default branch. `git diff
<c>^1 <c>` is exactly the change the branch received at step `c`, whether the
project squash-merges, merge-commits or pushes directly — hermes-agent, the
repository this was built against, does all three (4,526 squash commits, 1,629
merge commits, 19,072 first-parent steps). Replaying consecutive first-parent
steps reproduces history, so every intermediate tree is a real commit that
really passed CI.

A chain is therefore a contiguous run of history, partitioned into stages:

```
base ─[carry]─▶ c1^ ─[goal 1]─▶ c1 ─[carry]─▶ c2^ ─[goal 2]─▶ c2 ─▶ ...
```

- **Anchor** — a step whose diff touches both source and test files, so a
  fail-to-pass oracle can exist for it. The anchor's own change is the stage's
  **goal**: what the agent must implement.
- **Carry** — steps that cannot anchor a stage (formatting sweeps, dependency
  bumps, docs). The environment applies these *for free* when the stage opens.

Carry is what makes the partition both gapless and fair. Gapless, because no
history is skipped, so stage k+1 starts from a real commit. Fair, because the
agent is never asked to reproduce a formatting sweep, yet a later stage whose
tests depend on that sweep still sees it. An earlier draft simply folded carried
steps into the stage's expected work; that quietly asked the agent to reproduce
`npm run fix` output and could make a later stage unreachable.

A carry over budget becomes a **barrier**: chains are never built across one, so
a 300k-line dependency bump cannot be shipped inside a task. Barrier bounds are
deliberately loose — dropping `max_carry_steps` from 25 to 6 cut usable chains on
hermes-agent from 417 to 75, because every run of unremarkable commits became a
barrier instead of free setup.

### Coherence comes from selection, never reordering

A window whose anchors concentrate on one subsystem reads as one sustained piece
of work. `min_coherence` filters on that share. It is a *filter*, not a ranking:
ranking candidates by coherence and taking them greedily strands runs shorter
than `min_stages` between its picks and measurably loses chains (348 disjoint
chains became 281 on hermes-agent) while barely moving median coherence.
Windows are therefore taken in history order.

### Horizon guarantee

`action_floor` = Σ over stages of (`source_files` + 2): one read, one edit per
file, one `chain submit`. A chain is emitted only when the floor clears
`min_action_floor` **and** it has at least `min_stages` stages. The stage minimum
is load-bearing: without it a single 500-line commit clears a floor of 100 on its
own and the result is one big task behind one gate, not a horizon.

Selection uses a padded floor (`validation_margin`, default 1.35) because
validation strips stages whose change moved no test; the post-validation check
against the real floor is absolute.

### Interaction protocol

Harbor gives a task one agent session and one terminal verifier run. A
long-horizon environment needs more, so the image carries a stage controller:

```bash
chain status     # the current stage and what it must achieve
chain submit     # grade the current stage and open the next one
chain log        # what has been completed so far
```

`chain submit` is the environment's step function: it materializes the stage's
tests, runs the targeted command, records the outcome and advances the pointer.

Stage objectives are revealed one at a time. Publishing all of them in
`instruction.md` would turn a long-horizon task into one large specification, and
an agent that can read stage 20 before starting stage 1 plans against the answer
key rather than against the repository.

### Reward

`reward = mean(stage_reward)`, where each stage scores `f2p_rate × p2p_rate`
exactly as in `pr_runtime`.

Averaging rather than gating all-or-nothing is what makes the signal usable over
100+ steps: an agent that lands 14 of 25 milestones scores meaningfully above one
that lands 2.

Two properties are deliberate:

- **The ledger is telemetry, not reward.** The agent owns every file in its own
  container, so nothing the controller records can be trusted for scoring. The
  terminal verifier recomputes everything against the final tree; the ledger only
  reports the horizon actually travelled.
- **Scoring happens at the final tree.** A stage is credited only if its
  behaviour is still present at the end, so the agent cannot pass stage 3 by
  reverting stage 2. Work must accumulate, which is the realistic setting.

Because later history legitimately edits or deletes an earlier stage's tests, the
oracle sets are filtered at generation time to the tests that still pass on the
gold head. That preserves the invariant that real history scores 1.0.

**Two copies of every test file ship**, and the reason is a bug found in
end-to-end testing. Mid-run, stage k must be passable without having done k+1, so
its gate needs the tests as they stood at stage k. At the end, the oracle sets
were validated at the head, so terminal scoring needs the head's tests. Shipping
only the per-stage copies scored the gold patch 0.991 instead of 1.0, because one
stage's test file had been edited later in history.

### Corpus

Chain building is a global query over the whole PR list, so the PR corpus is
persisted once per repo in SQLite and reused. `list_merged_prs` cannot serve this
— it caps at 1,000 PRs and spends one REST call per PR resolving `base.sha`. The
bulk harvester fetches 100 PRs per GraphQL request at 3 rate-limit points, is
checkpointed per page so an interrupted run resumes, and runs as two streams
(oldest-first and newest-first) that meet in the middle. Harvesting all 65,944
PRs of hermes-agent cost roughly 2,000 of 5,000 hourly points.

## Verification

- **Reward kind** — `test_execution`.
- **Oracle invariant** — applying `solution/patch.diff` (the chain's whole diff)
  lands the gold tree, where every surviving stage test passes.
- **Non-tamper** — test files are *copied* over the tree from sealed payloads
  before grading, so editing or deleting a test cannot raise the score. Copying
  beats patching here: the agent may have rewritten the file, and a patch would
  conflict.
- **Anti-contamination** — the working tree sits at the chain's base commit and
  the git history is then scrubbed, so no future commit (and therefore no stage's
  fix) is reachable from `.git`. Verified on a built task image: the chain head
  and stage-2 gold commits are both pruned, and only `refs/heads/base` survives.
  The egress guard from `_env_guard` blackholes the fix-bearing hosts as usual.
- **Disclosure** — stage test payloads and instructions enter the container; no
  stage's gold source diff does. This is the same posture `pr_runtime` already
  takes, whose `test.sh` carries the test patch inline.

### Measured on `NousResearch/hermes-agent`

| Quantity | Value |
|---|---|
| PRs harvested | 65,944 (11,055 merged · 33,160 closed · 21,729 open) |
| First-parent history steps | 19,072 · 11,467 resolve to a PR |
| Gated stages, PR-anchored (shipped default) | 5,886 in 36 segments |
| Gated stages, best-effort attribution | 9,807 in 12 segments |
| Chains at floor 100, best-effort, fully disjoint | 411 of 423 |
| Median stages per chain | 23 |
| Validation cost | ~8 s per test run, 4 runs per stage |
| Stages surviving validation | 20 of 24 (83%) |
| Emitted chain | 20 stages · floor 142 · `hermes_cli` · 467 F2P · 315 P2P |
| **Gold patch reward** | **1.000000** — 20/20 stages resolved |
| **Untouched tree reward** | **0.000000** — 0/20 stages resolved |

The disjoint ceiling is arithmetic: `gated_stages / stages_per_chain`, or 426 with
best-effort attribution. Asking for more chains than that climbs an explicit
overlap ladder rather than silently returning fewer, and the rung used is reported.

Both reward endpoints were wrong before the oracle was tightened, and the two
failures had different causes worth recording:

* **Gold scored 0.991, not 1.0.** One stage's test file had been edited later in
  history, so the shipped stage-era copy no longer contained the test its oracle
  named. Fixed by shipping the head's copies as well and grading the final tree
  with those.
* **An untouched tree scored 0.111,** with one stage as high as 0.833. The oracle
  was derived from stage-era test files but graded with the head's, so tests that
  failed at a stage's start in their old form already passed there in their final
  form. Fixed by requiring a fail-to-pass test to fail on **all four** trees the
  reward depends on, each checked with the test version that will really be
  applied to it.

A third defect had nothing to do with the oracle and cost more stages than either:
the bootstrap image installed HEAD's dependencies only, so at older commits
`tools/web_tools.py`'s module-scope `import firecrawl` failed and pytest aborted
*collection*. A collection error parses as one failed test, which reads as "every
test in this stage failed". Installing every extra on the project's own Python
3.11 took survival from 7 of 24 stages to 20 of 24.

## Alternatives considered

- **Reset each stage to gold before the next one.** Gives clean per-stage credit
  and prevents cascading failure, but it needs the gold source diff inside the
  container, which hands the agent the answer. Rejected.
- **Chain by file coupling instead of history order.** Produces thematically
  tighter chains but the stages do not apply on top of each other, so nothing can
  be verified. This is the failure mode the whole design exists to avoid.
- **A sidecar container holding sealed stage data.** Would hide future stages'
  tests from the agent. Rejected as disproportionate: it doubles the runtime
  surface, and the disclosure matches what `pr_runtime` already accepts.
- **Include open and unmerged-closed PRs as runtime stages.** They have no
  verified merge, so no fail-to-pass oracle can be derived and they cannot gate a
  milestone. They are harvested into the corpus (33,160 closed, 21,729 open on
  hermes-agent) and remain available for diff-graded use; see Future work.

## Future work

- **Diff-graded stages from unmerged PRs.** An open or closed-unmerged PR has a
  real diff against a known base, which `pr_diff`'s reward can score. Such a
  stage would be non-mutating — score the agent's patch, then restore the tree —
  so it composes with the replay chain without desyncing it. This is what would
  lift the yield ceiling past the `gated_stages / stages_per_chain` limit.
- **LLM-synthesized stage instructions.** Currently a stage's objective is its PR
  title plus leak-stripped body. `commit_runtime` shows that a rewrite produces
  cleaner, less gameable problem statements.
- **Parallel validation.** Stage validation is embarrassingly parallel across
  chains; one container per worker would cut the full 417-chain run from roughly
  58 hours serial to a handful.
