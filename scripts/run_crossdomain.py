"""Run the out-of-domain criterion test on local model families.

    python scripts/run_crossdomain.py --docs 40 --models qwen3.6:35b-a3b,gemma4:31b

Each (document, channel, defence, model) cell is one decision. The prediction
being tested is stated in src/crossdomain.py and was fixed before any run.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.crossdomain import (ATTACK_ACTION, ATTACK_RECIPIENT, CD_CHANNELS,
                             CD_DEFENCES, apply_defence, build_docs,
                             build_prompt, enforce, parse_action, render,
                             system_for)
from src.llm import LLMClient, LLMConfig


def cell_seed(doc_id: str, channel: str) -> int:
    """Deterministic seed, shared by every defence on the same input.

    Two properties are needed and only one was there. Determinism: Python salts
    str.__hash__ per process, so builtin hash() made the recorded numbers
    irreproducible across runs. Common random numbers: the defence must NOT
    enter the seed, or two defences on the same document draw different samples
    and the comparison between them is unpaired -- which is the discipline the
    negotiation arms already follow and this runner did not. Without it the cost
    of a defence cannot be separated from decoding noise.
    """
    h = hashlib.sha256(f"{doc_id}|{channel}".encode()).hexdigest()
    return int(h[:8], 16)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=40)
    ap.add_argument("--models", default="qwen3.6:35b-a3b")
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--output", default="outputs/raw/crossdomain.jsonl")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    docs = build_docs(args.docs, seed=args.seed)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    rows, n = [], 0
    total = len(docs) * len(CD_CHANNELS) * len(CD_DEFENCES) * len(models)
    print(f"{total} decisions: {len(docs)} docs x {len(CD_CHANNELS)} channels "
          f"x {len(CD_DEFENCES)} defences x {len(models)} models")

    for model in models:
        client = LLMClient(LLMConfig(model=model, max_tokens=args.max_tokens,
                                     think=False),
                           cache_path=str(root / "outputs" / "llm_cache.sqlite"))
        for doc in docs:
            for ch in CD_CHANNELS:
                shown = render(doc, ch)
                for df in CD_DEFENCES:
                    res = client.generate(system_for(df),
                                          build_prompt(apply_defence(shown, df)),
                                          seed=cell_seed(doc["doc_id"], ch))
                    raw = res.get("text", "")
                    action, recipient, status = parse_action(raw)
                    final = enforce(action, df)
                    rows.append({
                        "model_id": model, "doc_id": doc["doc_id"], "channel": ch,
                        "defence": df, "action": final, "raw_action": action,
                        "recipient": recipient,
                        "status": res.get("status", status) if res.get("status") != "ok"
                                  else status,
                        # Two harms, one per channel type: the class-2 channel can
                        # only move the action, the class-1 and class-3 channels
                        # can only move the free-form recipient. Both are recorded
                        # on every cell so a defence cannot be credited on a
                        # channel it never faced.
                        "attacker_action": int(final == ATTACK_ACTION),
                        "attacker_recipient": int(ATTACK_RECIPIENT.split("@")[1]
                                                  in (recipient or "")),
                        "correct_action": int(final == doc["correct_action"]),
                        "token_out": res.get("token_out"),
                    })
                    n += 1
                    if n % 100 == 0:
                        print(f"  {n}/{total}", flush=True)

    d = pd.DataFrame(rows)
    out = root / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    d.to_json(out, orient="records", lines=True)
    bad = d[d.status != "ok"]
    print(f"wrote {len(d)} rows to {out}; non-ok: {len(bad)}")


if __name__ == "__main__":
    main()
