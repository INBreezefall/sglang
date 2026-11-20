"""Observe process memory, actual VMAs and DMA view lifetime across repeated full GLM host pools."""
import ctypes
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
import torch
import torch_npu
from sglang.srt.mem_cache.pool_host.common import alloc_with_pin_memory_npu

number,flag,output=int(sys.argv[1]),int(sys.argv[2]),Path(sys.argv[3])
assert number==48543
torch.npu.set_device(0)
torch.set_num_threads(4)
os.environ.pop('SGLANG_HICACHE_HOST_MEM_BACKEND',None)
os.environ['SGLANG_NPU_HICACHE_HOST_ALLOCATOR']='aclrt' if flag else 'torch'
# Initialize the context and transfer components before recording the idle process.
torch.zeros(128,device='npu').cpu();torch.npu.synchronize()
config=json.loads(Path('/model/config.json').read_text());config=config.get('text_config',config)
layers=config['num_hidden_layers'];widths=[config[k] for k in ('kv_lora_rank','qk_rope_head_dim','index_head_dim')]
assert [layers,*widths]==[78,512,64,128]
header=re.compile(r'^([0-9a-f]+)-([0-9a-f]+) ')
fields=('Rss','Pss','Locked','Anonymous','AnonHugePages','Private_Dirty','Size')
def measure(addresses=()):
    status={}
    for line in Path('/proc/self/status').read_text().splitlines():
        k,_,v=line.partition(':')
        if k in ('VmRSS','VmSize','VmLck','VmPin','RssAnon','RssFile','VmSwap'):
            status[k]=int(v.strip().split()[0])*1024
    current=None;maps=[]
    for line in Path('/proc/self/smaps').read_text().splitlines():
        match=header.match(line)
        if match:
            lo,hi=(int(x,16) for x in match.groups())
            # One pinned pool may span several kernel VMAs.
            used=sum(max(0,min(hi,a+n)-max(lo,a)) for a,n in addresses)
            current={'requested_covered':used} if used else None
            if current is not None:maps.append(current)
        elif current is not None:
            k,_,value=line.partition(':')
            if k in fields:current[k]=int(value.split()[0])*1024
    return dict(process=status,pool_vmas={k:sum(m.get(k,0) for m in maps) for k in (*fields,'requested_covered')},vma_count=len(maps))
def dma(view,tag):
    view.fill_(tag)
    torch.npu.synchronize()
    start=time.perf_counter()
    value=view.to('npu',non_blocking=True);torch.npu.synchronize()
    mid=time.perf_counter()
    copied=value.to('cpu',non_blocking=True);torch.npu.synchronize()
    end=time.perf_counter()
    assert torch.equal(view,copied)
    return dict(h2d_ms=1000*(mid-start),d2h_ms=1000*(end-mid),
                sha256=hashlib.sha256(copied.view(torch.uint8).numpy().tobytes()).hexdigest())
rows=[]
for it,pages in enumerate((385,385,385)):
    gc.collect()
    before=measure()
    hosts=[];views=[];addresses=[];times=[]
    for width in widths:
        start=time.perf_counter()
        h=alloc_with_pin_memory_npu((pages,layers,128,width),torch.bfloat16,'cpu',True,None)
        times.append((time.perf_counter()-start)*1000);hosts.append(h)
        addresses.append((h.data_ptr(),h.numel()*h.element_size()))
        h.fill_(it+1)
        views.append(h[-1,-1])
    del h
    full=measure(addresses)
    trial=sum(n for a,n in addresses)
    assert full['pool_vmas']['requested_covered']==trial,(full,trial)
    hosts.clear();gc.collect()
    kept=measure(addresses)
    copied=[dma(v,7+it+i) for i,v in enumerate(views)]
    views.clear();gc.collect()
    after=measure(addresses)
    rows.append(dict(pages=pages,payload_bytes=trial,alloc_ms=times,before=before,full=full,
                     kept_by_views=kept,after_last_view=after,transfers=copied))
torch.npu.empty_cache();gc.collect()
output.write_text(json.dumps(dict(feature=number,flag=flag,sha=os.environ['ARM_SHA'],rows=rows,
    after_empty_cache=measure(),device=torch.npu.get_device_name(0),torch=torch.__version__),indent=2)+'\n')
