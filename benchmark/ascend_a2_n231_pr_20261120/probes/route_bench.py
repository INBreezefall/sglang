"""Route three turns on eight independent physical paged radix caches."""
import json,os,random,sys,subprocess
from array import array
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

CAP=32768
def rank_replay(events):
    import torch, torch_npu
    from sglang.srt.hardware_backend.npu.allocator_npu import NPUPagedTokenToKVPoolAllocator
    from sglang.srt.mem_cache.base_prefix_cache import InsertParams
    from sglang.srt.mem_cache.kv.alloc.demand import KVDemand
    from sglang.srt.mem_cache.radix_cache import RadixKey
    from sglang.test.kits.unified_radix_cache.config import CacheConfig
    from sglang.test.kits.unified_radix_cache.fixtures import build_fixture,match_prefix
    torch.npu.set_device(0);torch.set_num_threads(1)
    records=[]
    for case in events:
        cfg=CacheConfig(page_size=128,kv_size=CAP,max_context_len=32768,head_num=1,head_dim=64,num_layers=1)
        with patch("sglang.test.kits.unified_radix_cache.fixtures.TokenToKVPoolAllocator",partial(NPUPagedTokenToKVPoolAllocator,page_size=128)):
            cache,alloc,_=build_fixture(cfg,tree_core_backend="python")
        for row in case:
            key=array("q",[1000+row["sid"]]*row["length"])
            match=match_prefix(cache,key);hit=match.device_prefix_len
            need=len(key)-hit
            lock=cache.inc_lock_ref(match.last_device_node).to_dec_params()
            evicted=0
            if need>alloc.available_size():
                evicted=cache.evict_for_alloc(KVDemand(full_tokens=need)).num_tokens_evicted
            slots=alloc.alloc(need);assert slots is not None and slots.device.type=="npu"
            if hit:
                prev=cache.tree_core.collect_full_device_indices(match.last_device_node,cache.root_node_handle())
                slots=torch.cat([prev,slots])
            cache.insert(InsertParams(key=RadixKey(key),value=slots))
            cache.dec_lock_ref(match.last_device_node,lock)
            assert match_prefix(cache,key).device_prefix_len==len(key)
            records.append(dict(**row,hit=hit,missing=need,evicted=evicted))
        cache.sanity_check();torch.npu.synchronize()
    return records
if sys.argv[1]=="--rank":
    dest=Path(sys.argv[2]);dest.write_text(json.dumps(rank_replay(json.loads(sys.stdin.read()))))
    sys.exit(0)
from sglang.srt.managers import data_parallel_controller as dp
from sglang.srt.managers.io_struct import TokenizedGenerateReqInput
from sglang.srt.sampling.sampling_params import SamplingParams
out=Path(sys.argv[1]);role,arm=sys.argv[2:]
work=[[] for i in range(8)]
lens=[16384,12288,8192,10240,6144,4096,2048,1024]*3
for seed in (11,29):
    rng=random.Random(seed);initial=list(range(24));rng.shuffle(initial)
    shuffled=[initial,rng.sample(initial,24),rng.sample(initial,24)]
    for name,orders in (("fixed",[initial]*3),("permuted",shuffled)):
        for r in work:r.append([])
        ctrl=object.__new__(dp.DataParallelController);ctrl.workers=list(range(8));ctrl._active_workers=list(range(8))
        ctrl.dp_budget=dp.DPBudget(8,track_dispatches=True) if arm=="on" and role=="decode" else dp.DPBudget(8)
        if arm=="on":
            if role=="prefill":ctrl.prefill_sessions=dp.PrefillSessionPlacement(8,slots_per_rank=3)
            else:ctrl.decode_sessions=dp.DecodeSessionPlacement(8,reserved_decode_tokens=128)
        for i,order in enumerate(orders):
            ctrl.dp_budget.update_budget([NS(dp_rank=r,timestamp=i+1,num_running_reqs=0,num_waiting_reqs=0,num_total_tokens=0,max_total_num_tokens=CAP,dp_dispatch_ack_seq=(ctrl.dp_budget.sent_sequences[r] if hasattr(ctrl.dp_budget,"sent_sequences") else None),dp_dispatch_ack_cum_tokens=(ctrl.dp_budget.sent_tokens[r] if hasattr(ctrl.dp_budget,"sent_tokens") else None)) for r in range(8)])
            for sid in order:
                length=lens[sid]+i*512
                req=TokenizedGenerateReqInput(rid=f"{i}-{sid}",input_text=None,input_ids=range(length),input_embeds=None,mm_inputs=None,token_type_ids=None,sampling_params=SamplingParams(max_new_tokens=128),return_logprob=False,logprob_start_len=-1,top_logprobs_num=0,token_ids_logprob=None,stream=False,session_id=str(sid))
                result=[]
                with patch.object(dp,"sock_send",side_effect=lambda worker,request:result.append(worker)):
                    if arm=="on":getattr(ctrl,role+"_session_tokens_scheduler")(req)
                    else:ctrl.total_tokens_scheduler(req)
                assert len(result)==1
                rank=result[0];work[rank][-1].append(dict(seed=seed,order=name,turn=i+1,sid=sid,rank=rank,length=length))
def worker(rank):
    f=out.parent/f"rank-{rank}.json"
    with (out.parent/f"rank-{rank}.log").open("w") as log:
        subprocess.run([sys.executable,__file__,"--rank",str(f)],input=json.dumps(work[rank]),env=dict(os.environ,ASCEND_RT_VISIBLE_DEVICES=str(rank)),text=True,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=420)
    return json.loads(f.read_text())
with ThreadPoolExecutor(max_workers=8) as ex: rows=sum(ex.map(worker,range(8)),[])
assert len(rows)==288
out.write_text(json.dumps(dict(sha=os.environ["ARM_SHA"],role=role,arm=arm,capacity_per_rank=CAP,page_size=128,rows=rows),indent=2)+"\n")
