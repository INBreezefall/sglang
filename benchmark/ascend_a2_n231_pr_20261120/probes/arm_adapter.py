"""Run the previously published operator and routing protocols against the staged arm."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

n,flag,out=int(sys.argv[1]),int(sys.argv[2]),Path(sys.argv[3])
cases={
 48541:('route_bench.py',['decode','on' if flag else 'off']),
 48539:('lpm_bench.py',['64' if flag else '0']),
 48560:('numeric_bench.py',['int8' if flag else 'bf16']),
 48562:('index_bench.py',['int8' if flag else 'bf16','21']),
 48555:('pool_bench.py',['compact' if flag else 'bf16']),
}
name,argv=cases[n];probe=Path('/bench/probes')/name
raw=out.with_suffix('.native.json')
sys.argv=[str(probe),str(raw),*argv]
runpy.run_path(str(probe),run_name='__main__')
sample=json.loads(raw.read_text())
assert sample['sha']==os.environ['ARM_SHA']
sample.update(feature=n,flag=flag,protocol_sha256=hashlib.sha256(probe.read_bytes()).hexdigest())
out.write_text(json.dumps(sample,indent=2)+'\n')
