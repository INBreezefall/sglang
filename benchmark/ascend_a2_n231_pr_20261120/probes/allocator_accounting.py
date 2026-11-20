"""Compare real driver-requested host bytes, view lifetime and cached blocks in fresh main/PR processes."""
import ctypes
import gc
import hashlib
import json
import os
from pathlib import Path
import sys

import torch
import torch_npu
from sglang.srt.mem_cache.pool_host.common import alloc_with_pin_memory_npu

pr,flag,out=int(sys.argv[1]),int(sys.argv[2]),Path(sys.argv[3])
assert pr==48543
torch.npu.set_device(0);torch.set_num_threads(3)
os.environ.pop('SGLANG_HICACHE_HOST_MEM_BACKEND',None)
os.environ['SGLANG_NPU_HICACHE_HOST_ALLOCATOR']='aclrt' if flag else 'torch'
native_live={};events=[]
if flag:
    from sglang.srt.mem_cache.pool_host import npu_host_allocator as module
    vendor=module._acl_host_api()
    class MeasuredACL:
        def aclrtMallocHost(self,out_ptr,size):
            rc=vendor.aclrtMallocHost(out_ptr,size)
            if rc==0:
                ptr=ctypes.cast(out_ptr,ctypes.POINTER(ctypes.c_void_p)).contents.value
                assert ptr and ptr not in native_live
                native_live[ptr]=int(size);events.append(['alloc',int(size)])
            return rc
        def aclrtFreeHost(self,ptr):
            key=ptr.value;size=native_live[key];rc=vendor.aclrtFreeHost(ptr)
            if rc==0:
                del native_live[key];events.append(['free',size])
            return rc
    measured_acl=MeasuredACL()
    module._acl_host_api=lambda:measured_acl
torch.zeros(128,device='npu').cpu();torch.npu.synchronize();gc.collect();torch.npu.host_empty_cache()
config=json.loads(Path('/model/config.json').read_text());config=config.get('text_config',config)
layers=config['num_hidden_layers'];widths=[config[k] for k in ('kv_lora_rank','qk_rope_head_dim','index_head_dim')]
assert [layers,*widths]==[78,512,64,128]
def snap():
    t=torch.npu.host_memory_stats()
    keys=['allocated_bytes.current','active_bytes.current','allocations.current','active_requests.current','num_host_alloc','num_host_free']
    counters={k:int(t.get(k,0)) for k in keys}
    live=sum(native_live.values())
    return dict(torch_pinned=counters,acl_outstanding_bytes=live,acl_outstanding_buffers=len(native_live),
                total_driver_requested_bytes=counters['allocated_bytes.current']+live)
rows=[]
for seq,pages in enumerate((385,385,397)):
    before=snap();hosts=[];views=[];requested=[];allocation_steps=[];hashes=[]
    for i,width in enumerate(widths):
        value=alloc_with_pin_memory_npu((pages,layers,128,width),torch.bfloat16,'cpu',True,None)
        requested.append(value.numel()*value.element_size());value.fill_(i+seq+3)
        hosts.append(value);views.append(value[-1,-1]);allocation_steps.append(snap())
    del value
    full=snap();hosts.clear();gc.collect();torch.npu.synchronize();keep=snap()
    for view in views:
        onchip=view.to('npu',non_blocking=True);torch.npu.synchronize()
        restored=onchip.to('cpu',non_blocking=False);assert torch.equal(view,restored)
        hashes.append(hashlib.sha256(restored.view(torch.uint8).numpy().tobytes()).hexdigest())
        del onchip,restored
    del view
    views.clear();gc.collect();torch.npu.synchronize();released=snap()
    if flag:
        assert full['acl_outstanding_bytes']==keep['acl_outstanding_bytes']==sum(requested) and not native_live
    rows.append(dict(pages=pages,logical_bytes=sum(requested),requested_mapping_bytes=requested,before=before,
                     after_each_allocation=allocation_steps,fully_allocated=full,kept_by_last_views=keep,
                     after_last_views=released,verified_dma=hashes))
torch.npu.host_empty_cache();final=snap()
out.write_text(json.dumps(dict(feature=pr,flag=flag,sha=os.environ['ARM_SHA'],device=torch.npu.get_device_name(0),
    source='torch.npu.host_memory_stats allocated_bytes.current plus currently outstanding successful ACL requests',
    rows=rows,direct_acl_events=events,after_explicit_torch_host_cache_purge=final),indent=2)+'\n')
