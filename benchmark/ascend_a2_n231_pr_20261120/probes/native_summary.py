"""Extract anonymous formal per-request scores and counter checks without traffic text."""
import collections
import hashlib
import json
from pathlib import Path
import statistics
import math
import sys
from datetime import datetime

concurrency=int(sys.argv[1]);assert concurrency in (44,48)
output_cap=(int(sys.argv[2]) or None) if len(sys.argv)>2 else None

files=list(Path('.').rglob('teleagi_benchmark_results_*.json'))
assert len(files)==1
s=json.loads(files[0].read_text());cfg=s['test_config']
assert (cfg['concurrent_workers'],cfg['top_k_sessions'],cfg['total_requests'],cfg['warmup_concurrent_workers'],cfg['temperature_override'],cfg['top_p_override'],cfg['max_tokens_override'])==(concurrency,concurrency,200,concurrency,.7,1.,output_cap)
rows=[r for r in s['results'] if not r.get('is_warmup') and r.get('turn_id',0)>0]
good=[r for r in rows if r.get('success')]
tts=[r['first_token_latency'] for r in good if r.get('first_token_latency') is not None]
toks=sum(r.get('completion_tokens') or 0 for r in good)
expected_intervals=sum(max(0,(r.get('completion_tokens') or 0)-1) for r in good)
output_only=sum((r.get('output_only_token_count') or 0) for r in good) if all(isinstance(r.get('output_only_token_count'),int) for r in good) else None
client_intervals=[]
for r in good:
    n=r.get('completion_tokens') or 0;tt=r.get('first_token_latency');wall=r.get('total_time')
    if n>1 and isinstance(tt,(int,float)) and isinstance(wall,(int,float)) and wall>tt:
        client_intervals.append((n-1,wall-tt))
window=cfg.get('formal_window') or {}
def timestamp(s):return datetime.fromisoformat(s.replace('Z','+00:00')).timestamp()
official_wall=(timestamp(window['end'])-timestamp(window['start'])) if window.get('start') and window.get('end') else None
def count(f,key):
    return sum(float(l.rsplit(' ',1)[1]) for l in Path(f).read_text().splitlines() if l.startswith((key+'{',key+' ')))
def diff(key):return count('after.prom',key)-count('before.prom',key)
nc=diff('sglang:inter_token_latency_seconds_count');dt=diff('sglang:inter_token_latency_seconds_sum')
def pct(vals,p):
    z=sorted(vals);i=(len(z)-1)*p;k=math.floor(i);return z[k]+(z[math.ceil(i)]-z[k])*(i-k)
def token_lengths(key):
    values=[r[key] for r in good if isinstance(r.get(key),int)]
    return dict(count=len(values),min=min(values,default=None),p95=pct(values,.95) if values else None,max=max(values,default=None))
def sid(row):
    value=row.get('session_id',row.get('session',''))
    return hashlib.sha256(str(value).encode()).hexdigest()
per_turn={}
for turn in sorted({r['turn_id'] for r in rows}):
    turn_rows=[r for r in rows if r['turn_id']==turn]
    times=[r['first_token_latency'] for r in turn_rows if r.get('success') and r.get('first_token_latency') is not None]
    per_turn[turn]=dict(requests=len(turn_rows),success=sum(bool(r.get('success')) for r in turn_rows),
        mean_ttft_s=statistics.mean(times) if times else None,p95_ttft_s=pct(times,.95) if times else None)
print(json.dumps(dict(concurrency=concurrency,warmup_concurrency=concurrency,top_k_sessions=concurrency,max_tokens_override=output_cap,requested=200,requests=len(rows),success=len(good),sessions=len({sid(r) for r in rows}),turns=dict(collections.Counter(r['turn_id'] for r in rows)),
 mean_ttft_s=statistics.mean(tts) if tts else None,p95_ttft_s=pct(tts,.95) if tts else None,maximum_ttft_s=max(tts,default=None),
 decode_tps=nc/dt if dt>0 else None,observed_intervals=nc,observed_inter_token_seconds=dt,expected_intervals=expected_intervals,counters_match=nc==expected_intervals and dt>0,
 client_decode_coverage=len(client_intervals),client_mean_decode_tps=statistics.mean(n/t for n,t in client_intervals) if client_intervals else None,
 client_weighted_decode_tps=sum(n for n,t in client_intervals)/sum(t for n,t in client_intervals) if client_intervals else None,
 client_formal_window_s=official_wall,client_aggregate_output_tps=toks/official_wall if official_wall and official_wall>0 else None,
 ttft_above_12s=sum(t>12 for t in tts),ttft_above_30s=sum(t>30 for t in tts),
 output_tokens=toks,output_only_tokens=output_only,client_output_only_tps=(output_only/official_wall if output_only is not None and official_wall and official_wall>0 else None),token_lengths={'prompt':token_lengths('prompt_tokens'),'completion':token_lengths('completion_tokens')},turn_metrics=per_turn,payload_case_sha256=hashlib.sha256(Path('native-point.sh').read_bytes()).hexdigest(),full_client_result_sha256=hashlib.sha256(files[0].read_bytes()).hexdigest(),
 rows=[dict(session_sha256=sid(r),turn=r.get('turn_id'),success=r.get('success'),ttft_s=r.get('first_token_latency'),total_s=r.get('total_time'),input_tokens=r.get('prompt_tokens'),output_tokens=r.get('completion_tokens')) for r in rows])))
