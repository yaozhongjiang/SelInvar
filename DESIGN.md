# Method and experiment design

What the code computes, why each experiment exists, and which reported result it
carries. Read this before `src/`; the module docstrings assume it.

## The question

A decision rule reading natural-language signals from a strategic counterpart
has to satisfy two things at once. It must not move when the counterpart changes
only how a fixed transaction is presented, and it must still move when the
transaction genuinely changes. At inference the latent state is unavailable, so
the rule cannot separate the two by inspection; it has to be built so that one
class of change cannot reach it while the other can. We call that *selective
invariance* and ask when it is attainable.

Either half alone is trivial, which is why both are measured. A rule that
rejects everything is perfectly invariant to every manipulation; a rule that
trusts every claim tracks every genuine improvement. Any experiment reporting
only one half can be passed by a degenerate rule, and an earlier version of this
benchmark could only report the first half — every rendering it produced was
revenue-neutral, so nothing in it could catch insensitivity.

## The criterion

Write `phi(m)` for the observables a decision rule may read, and

    Delta_k(m) = { phi(T_a m) - phi(m) : a in A_k }

for the displacement set channel `k` can produce on a fixed latent state.
Comparing `Delta_k` with the direction carrying utility sorts channels into
three cases, and the case decides the treatment rather than the designer doing
so:

| case | condition | treatment | price |
|---|---|---|---|
| 1 | `Delta_k` moves no payoff-relevant observable | quotient the channel out | zero |
| 2 | the informative direction already lies in the polar cone | pessimistic completion over an enumerable set | zero, plus enumeration |
| 3 | the directions are opposed | no invariant map is informative; estimate the residual | strictly positive |

Two propositions bound this. **Closure is exact exactly when the attacked
completion set contains the clean one** — sufficiency and necessity both, so the
precondition is the boundary of the guarantee rather than a convenience; the
argument uses nothing about how quality is priced. **Black-box proofness forces
invariance**: a guarantee holding for every decision function `f` cannot use the
weaker cone condition available to whoever designs `f`, so no prompt-level
defence attains it.

## Benchmark construction

Every attack is a *paired, revenue-neutral re-rendering*: the same latent
transaction, the same true expected settlement price, only the presentation
differs. Two invariants make that testable rather than assumed, and both are
asserted in `tests/test_core.py`:

- revenue neutrality to `1e-9` on the calibrated families (`9.1e-13` on the
  boundary construction), checked by arithmetic on each rendering;
- an honest rendering must price the offer at *exactly* its true value.

The second exists because the first is not enough. `clean HAR == 0` cannot see a
systematic *under*valuation, since undervaluing only causes rejection; a
double-subtraction bug in the Deal or No Deal valuation hid behind that blind
spot and undervalued honest offers by 1.8 points of a 10-point pie.

## Experiments

Each row names the runner, what it establishes, and where the number lands.

| runner | what it settles | where reported |
|---|---|---|
| `run_main.py` | scripted arms on the full test split; model arms per family | RQ1, RQ2 |
| `run_selective_invariance.py` | the responsive half: honest renderings in which the seller genuinely concedes | RQ2 |
| `run_boundary.py` | the failing side of the closure precondition, sets made incomparable | RQ2, App. |
| `run_utility_sensitivity.py` | closure survives repricing quality as `q^gamma`, `gamma` in {0.5, 1, 2} | RQ2 |
| `run_ablations.py` | which component carries the gain | RQ2, App. |
| `run_conformal.py`, `run_adaptive_attacker.py`, `certify` | coverage, and a seller that best-responds to the published rule | RQ2 |
| `run_dnd_rounds.py` | re-encoding splits the one surviving channel | RQ3 |
| `run_crossdomain.py` | the criterion applied to indirect prompt injection, our own testbed | App. |
| `run_agentdojo.py` | the same operators inside AgentDojo (banking, travel, Slack), their verdicts | RQ4 |
| `run_prose.py` | the offer as a seller's message rather than a structured record | RQ4 |
| `analyze.py`, `analyze_extra.py`, `compare_models.py` | every processed summary the claims recompute from | all |

### Design decisions that are load-bearing

**Common random numbers across arms.** Seeds derive from `(task, attack, repeat)`
and never from the arm or the defence, so a difference between two arms is
attributable to the arm. The cross-domain runner originally put the defence in
the seed; two defences on the same document then drew different samples and the
measured cost of a defence was decoding noise. `cell_seed` takes
`(doc_id, channel)` only, and a test asserts its signature.

**Seeds must survive the process.** Python salts `str.__hash__` per process, so
`hash()` gave different seeds on every run and the recorded numbers were not
reproducible. Seeds are SHA-256.

**One decision per cell, task as the unit.** Each task yields several correlated
renderings, so the independent unit is the task. Intervals are task-level
bootstrap: 5,000 resamples in the main pipeline, 3,000 in the per-family and
side analyses.

**Failures are recorded, never coerced.** API errors, empty completions, and
unparseable output are stored with their status rather than silently scored as
rejections. A reasoning model that spends its whole budget thinking returns
empty content, and mapping that to REJECT would make the arm look
pathologically safe.

**Exactness is asserted per task.** A reported `0.0000` is an invariance over
every task, not a rounded mean. `check_artifacts.py` asserts the per-task flag,
because a mean of offsetting errors would read the same.

**Environments are not pooled.** Reservation utilities, attack sets, and scales
differ, so a pooled mean corresponds to no decision problem. Files carrying
their own attack vocabulary — the improvement, boundary, and cross-domain runs —
are excluded from the negotiation summaries. Pooling one of them once diluted
the headline harmful-acceptance rate from `0.097` to `0.068` while the gate
still reported green, because the gate compared the write-up against summaries
that were themselves stale.

## Verification

`make verify` recomputes all 90 registered values from `outputs/processed` and
fails if any disagrees with the recomputed value. `make test` runs 66 invariants.
Several guard defects that once produced plausible but wrong results, and the
gate itself is guarded: the processed summaries must be newer than the raw
records they summarise, or the recomputation is vacuous.

## AgentDojo

`run_agentdojo.py` needs AgentDojo, which pins a large dependency set the rest of
this artifact does not use, so it is installed beside the project rather than in
it:

    python -m venv adj_venv && ./adj_venv/bin/pip install agentdojo==0.1.35
    OPENAI_API_KEY=... ./adj_venv/bin/python scripts/run_agentdojo.py --suite banking

`src/adj_defenses.py` imports nothing from AgentDojo, so `make test` checks the
operators without that venv. Attack success and task utility are AgentDojo's own
`security` and `utility` verdicts; this repository supplies only the defenses.
Two details are load-bearing. Their `important_instructions` attack addresses the
agent by a model name it reads out of the pipeline name, so an unlisted model has
to be named `local` or nothing runs. And the class-two operator refuses a call by
wrapping the runtime's `run_function`, not by editing the assistant message: a
turn stripped of all its tool calls serializes with a null content field that the
API rejects, and a transcript shaped differently from the undefended arm would no
longer be comparable to it.

## Model access

`run_main.py` and `run_crossdomain.py` speak an OpenAI-compatible API and read
`NEGOTIATION_API_BASE` and `NEGOTIATION_API_KEY`; a local server works. Two
protocol notes matter. Reasoning families reject `max_tokens` and any
temperature other than the default, so `--reasoning-api` sends
`max_completion_tokens` and no sampling parameters — an explicit flag rather
than something sniffed from the model id, because an arm run that way is not
decoding-matched and must not enter the matched-pair comparison. Rate limits are
per-model and hosted endpoints return 429; requests retry with backoff on
429/408/409/5xx and fail immediately on any other client error.
