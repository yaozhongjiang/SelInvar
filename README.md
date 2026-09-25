# SelInvar: selective invariance under strategic information manipulation

A decision rule that reads natural-language signals from a strategic
counterparty has to satisfy two things at once. It must not move when the
counterparty changes only how a fixed transaction is presented, and it must
still move when the transaction genuinely changes. At inference the latent state
is unavailable, so the rule cannot tell the two apart by inspection; it has to
be built so that one class of change cannot reach it and the other can.

This repository contains the benchmark, the operators, and the verification for
that question: when is such *selective invariance* attainable, and what does it
cost when it is not.

## Quick start

```
pip install -r requirements.txt
make all
```

`make test` runs 66 invariants. `make verify` recomputes 94 reported values from
the summaries in `outputs/processed/` and fails if any disagrees. Values of
`0.0000` are asserted as invariances over every task rather than as rounded
means, so a single non-zero task fails the check.

Both commands are deterministic and need no model access.

## What the code does

Seller-controlled information is sorted into three classes by comparing the
displacements a channel can produce against the direction that carries utility:

| class | condition | treatment | price |
|---|---|---|---|
| 1 | the channel moves no payoff-relevant observable | quotient it out | zero |
| 2 | the informative direction lies in the polar cone | pessimistic completion over an enumerable set | zero, plus enumeration |
| 3 | the directions are opposed | no invariant map is informative; estimate the residual | strictly positive |

The classification is mechanical and is fixed before any model runs. Class one
and two admit channel-level invariance with no learned risk score; class three
does not, and the code measures what estimating it costs instead of assuming it
away.

`DESIGN.md` states what each experiment settles and which design decisions are
load-bearing. Read it before `src/`; the module docstrings assume it.

## Layout

```
src/                benchmark construction, decision rules, attack and improvement renderers
scripts/            runners, analysis, and the claim checker
tests/              the invariants
data/               task files, splits, and the source corpora
outputs/processed/  the summaries every reported value is recomputed from
DESIGN.md           method and experiment design
```

## What is not included

Raw per-trial records are roughly 500 MB and are omitted; `make regenerate`
prints the order in which the runners rebuild them from `data/`. Four runners
need model access: `run_main.py`, `run_prose.py`, `run_crossdomain.py` and
`run_agentdojo.py` speak a chat-completions API and read `NEGOTIATION_API_BASE`
and `NEGOTIATION_API_KEY`, so a local server works as well as a hosted endpoint.
Everything else is deterministic.

`run_agentdojo.py` additionally needs AgentDojo, which pins a large dependency
set the rest of this tree does not use, so it is installed beside the project
rather than in it:

```
python -m venv adj_venv && ./adj_venv/bin/pip install agentdojo==0.1.35
./adj_venv/bin/python scripts/run_agentdojo.py --suite banking
```

Attack success and task utility there are AgentDojo's own `security` and
`utility` verdicts; this repository supplies only the defenses.
`src/adj_defenses.py` imports nothing from AgentDojo, so `make test` checks the
operators without that environment.

## Scope

The exact zeros are properties of the operators rather than statistical
estimates, and the tests assert them per task. The residual statistical layer's
guarantee is conditional: its ambiguity set misses its own coverage precondition
on roughly a quarter of messages, and a counterparty that best-responds to the
published rule inflates realized risk several-fold above the calibrated
certificate. Closure is not free either. In the banking suite its price falls
entirely on the one task whose legitimate recipient arrives only inside
untrusted content, because a channel is closed against every writer or against
none.

One hosted model family runs under an API that rejects the recorded sampling
parameters, so it is excluded from the decoding-matched comparison and recorded
separately.

## Data

`data/` carries derivatives of public corpora: CraigslistBargain, Deal or No
Deal, and an Amazon price-history sample. They are redistributed here in the
form the runners consume, under the terms of their original releases, and the
listing text is shipped verbatim, including contact strings that appear in the
public source listings. Nothing in this repository executes a tool call, sends a
message, or contacts a counterparty: the renderings are data and the decisions
are recorded fields.

## License

MIT, see `LICENSE`. The corpora under `data/` are redistributed under the terms
of their original releases and are not covered by it.
