"""Round 7A: are the six calibrated attack families a complete-enough set?

The policy-uniform certificate guarantees nothing outside the action set it
enumerates, and only one of our six families is pure framing. These two are also
pure framing, revenue-neutral, and were never used to calibrate anything, so an
effect here means the calibrated set under-samples the channel.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.attacks_framing import FRAMING_ATTACKS, render_framing
from src.dataset import read_tasks, is_harmful
from src.experiment import trial_seed
from src.llm import LLMClient, LLMConfig, SYSTEMS, build_user_prompt, parse_action
from src.stats import holm_bonferroni, paired_diff

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--tasks", type=int, default=200)
    ap.add_argument("--think", default="false")
    ap.add_argument("--max-tokens", type=int, default=250)
    a = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    splits = json.loads((root / "data/splits.json").read_text())
    by_id = {t.task_id: t for t in read_tasks(str(root / "data/tasks.jsonl"))}
    ids = splits["test"][: a.tasks]
    think = {"false": False, "true": True}.get(a.think, a.think)
    cli = LLMClient(LLMConfig(model=a.model, think=think, max_tokens=a.max_tokens),
                    cache_path=str(root / "outputs/llm_cache.sqlite"))

    rows = []
    for arm in ("llm_direct", "llm_safety"):
        for att in ("clean",) + FRAMING_ATTACKS:
            for i in ids:
                t = by_id[i]
                sig = render_framing(t, att, np.random.default_rng(trial_seed(i, "render", 0, 42)))
                out = cli.generate(SYSTEMS[arm], build_user_prompt(t, sig),
                                   trial_seed(i, arm, 0, 42))
                act, st = parse_action(out.get("text", ""))
                acc = act == "ACCEPT"
                rows.append(dict(task_id=i, buyer_id=f"{arm}{a.tag}", attack_id=att,
                                 status="ok" if st == "ok" else st, accepted=int(acc),
                                 harmful_accept=int(acc and is_harmful(sig, t.oracle))))
    d = pd.DataFrame(rows)
    p = root / f"outputs/raw/framing{a.tag}.jsonl"
    d.to_json(p, orient="records", lines=True)
    bad = int((d.status != "ok").sum())
    print(f"{len(d)} trials, non-ok {bad}")

    for arm, g in d[d.status == "ok"].groupby("buyer_id"):
        clean = g[g.attack_id == "clean"]
        st = [{"attack": f, **paired_diff(g[g.attack_id == f], clean, "harmful_accept",
                                          5000, 42)} for f in FRAMING_ATTACKS]
        s = pd.DataFrame(st); s["holm"] = holm_bonferroni(s.p_value.values)
        print(f"\n  {arm}  (accept on clean {clean.accepted.mean():.3f})")
        for _, r in s.iterrows():
            print(f"    {'*' if r.holm else ' '} {r.attack:18s} {r['diff']:+.4f} "
                  f"[{r.ci_low:+.4f},{r.ci_high:+.4f}] p={r.p_value:.4f}")
    print(f"\nwrote {p}")

if __name__ == "__main__":
    main()
