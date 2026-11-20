"""Measure completion order under the production LPM policy with real pinned H2D work."""
import json,sys,time,os,random
from pathlib import Path
from types import SimpleNamespace as NS
import torch,torch_npu
from sglang.srt.managers.schedule_policy import SchedulePolicy
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.runtime_context import get_disagg
torch.npu.set_device(0);torch.set_num_threads(1)
target=int(sys.argv[2])
if target: os.environ["SGLANG_HICACHE_LPM_COMPUTE_WEIGHT"]=str(target)
from sglang.srt.runtime_context import get_context
with get_context().override_server_args(disaggregation_mode="prefill"):
    policy=SchedulePolicy("lpm",RadixCache.create_simulated(),True,False,False)
# Prefix matching is supplied from a fixed fixture. Sorting and all copies run normally.
policy._compute_prefix_matches=lambda *args:set()
host=torch.empty((1024,32768),dtype=torch.uint8,pin_memory=True)
host.fill_(29)
dev=torch.empty_like(host,device="npu");keep=torch.zeros(32,dtype=torch.int32,device="npu")
rows=[]
for rep in range(20):
    requests=[]
    for i in range(16):
        hit=1024-i*16 if i<8 else 64+i
        l2=hit if i<8 else 0
        requests.append(NS(rid=str(i),origin_input_ids=range(hit+4),num_matched_prefix_tokens=hit,host_hit_length=l2,host_loaded_length=0))
    random.Random(48539+rep).shuffle(requests)
    torch.npu.synchronize()
    start=time.perf_counter_ns(); ordered=policy.calc_priority(requests)
    ids=[];finish=[]
    for req in ordered:
        count=req.host_hit_length
        if count:dev[:count].copy_(host[:count],non_blocking=True)
        keep.add_(1)
        torch.npu.synchronize()
        ids.append(int(req.rid));finish.append((time.perf_counter_ns()-start)/1e6)
    assert sorted(ids)==list(range(16))
    if rep>=3:rows.append(dict(order=ids,completion_ms=finish,bytes=sum(q.host_hit_length for q in requests)*32768))
assert int(keep[0])==320 and bool((dev[0]==29).all())
Path(sys.argv[1]).write_text(json.dumps(dict(sha=os.environ["ARM_SHA"],device=torch.npu.get_device_name(),weight=target,rows=rows),indent=2)+"\n")
