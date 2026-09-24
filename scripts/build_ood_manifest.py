"""Create leave-group-out manifests from a task JSONL file.
No task is assigned to multiple partitions; grouping is explicit to prevent leakage.
"""
import argparse,json
from pathlib import Path
from collections import defaultdict
p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--group-key',default='task_family');p.add_argument('--output',required=True);a=p.parse_args()
groups=defaultdict(list)
for line in Path(a.input).read_text().splitlines():
    if line.strip():
        d=json.loads(line); groups[d.get('metadata',{}).get(a.group_key,d['task_id'])].append(d['task_id'])
manifest={'groups':{k:v for k,v in groups.items()},'protocol':'leave-one-group-out; group labels are never mixed across train/validation/test'}
Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text(json.dumps(manifest,indent=2))
