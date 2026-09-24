"""Static experiment integrity checks."""
import argparse,json
from pathlib import Path
import pandas as pd
p=argparse.ArgumentParser();p.add_argument('--results',required=True);a=p.parse_args();d=pd.read_csv(a.results)
checks={
 'required_columns': set(['task_id','model_id','buyer_id','attack_id','accepted','harmful_accept','true_utility','repeat']).issubset(d.columns),
 'no_duplicate_trial_keys': not d.duplicated([c for c in ['task_id','model_id','buyer_id','attack_id','repeat'] if c in d]).any(),
 'binary_accept': set(d.accepted.dropna().unique()).issubset({0,1}),
 'binary_harmful_accept': set(d.harmful_accept.dropna().unique()).issubset({0,1}),
 'nonnegative_latency': d.latency_s.dropna().ge(0).all() if 'latency_s' in d else True,
}
checks={k: bool(v) for k,v in checks.items()}
print(json.dumps(checks,indent=2)); raise SystemExit(0 if all(checks.values()) else 1)
