"""Exercise the production public allocator on one Ascend card in a fresh Triton cache."""
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import runpy
import sys

import torch
import torch_npu
from sglang.srt.hardware_backend.npu.allocator_npu import NPUPagedTokenToKVPoolAllocator

number,flag,destination=int(sys.argv[1]),int(sys.argv[2]),Path(sys.argv[3])
assert number==48633 and torch.npu.device_count()==1
root=Path(os.environ['TRITON_CACHE_DIR'])
assert not list(root.rglob('*.npubin'))
source=Path(inspect.getfile(NPUPagedTokenToKVPoolAllocator))
assert 'from sgl_kernel_npu.operators.mem_cache import alloc_extend_indices' in source.read_text()
benchmark=Path('/bench/probes/allocator_bench.py')
baseline_output=destination.with_suffix('.timed.json')
sys.argv=[str(benchmark),str(baseline_output)]
runpy.run_path(str(benchmark),run_name='__main__')
sample=json.loads(baseline_output.read_text())
assert sample['sha']==os.environ['ARM_SHA']
public=importlib.import_module('sgl_kernel_npu.operators.mem_cache.alloc_extend_indices')
actual=public._triton_fn
assert callable(actual)
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
binary_hashes={path:digest(root/path) for path in sample['artifacts'] if path.endswith('.npubin')}
assert binary_hashes

# These boundary and mixed-prefix cases run after snapshotting the timed cache.
page=128
def verify(rows):
    allocator=NPUPagedTokenToKVPoolAllocator(65536,page,torch.bfloat16,'npu:0',None,False)
    free=list(range(64,512))
    allocator.free_pages=torch.tensor(free,dtype=torch.int64,device='npu:0')
    prefix=torch.tensor([p for p,n in rows],dtype=torch.int64)
    seq=torch.tensor([p+n for p,n in rows],dtype=torch.int64)
    tails=[(1+i)*page+(p-1)%page if p else -1 for i,(p,n) in enumerate(rows)]
    cursor=0;expected=[]
    for (p,n),tail in zip(rows,tails):
        take=min(n,(-p)%page) if p else 0
        expected.extend(range(tail+1,tail+1+take))
        left=n-take
        expected.extend(free[cursor+j//page]*page+j%page for j in range(left))
        cursor+=(left+page-1)//page
    result=allocator.alloc_extend(prefix.to('npu:0'),prefix,seq.to('npu:0'),seq,
              torch.tensor(tails,dtype=torch.int64,device='npu:0'),len(expected),num_new_pages=cursor)
    reference=torch.tensor(expected,dtype=torch.int32)
    assert result.dtype==torch.int32 and torch.equal(result.cpu(),reference)
    assert allocator.free_pages.cpu().tolist()==free[cursor:]
    return dict(prefix_and_extend=rows,output_slots=len(expected),new_pages=cursor,
                output_sha256=hashlib.sha256(reference.numpy().tobytes()).hexdigest())
cases=[[(0,0)],[(0,0),(13,1),(127,1),(130,128),(256,2049),(31,4097)],
       [(128,1),(129,0),(256,2048),(33,17)],[(0,200*128+1)]]
checks=[verify(case) for case in cases]
sample.update(feature=number,flag=flag,protocol_sha256=digest(benchmark),binary_hashes=binary_hashes,
              public_operator_sha256=digest(inspect.getfile(public)),triton_adapter_sha256=digest(inspect.getfile(actual)),
              mixed_and_boundary=checks,visible_devices=torch.npu.device_count())
destination.write_text(json.dumps(sample,indent=2)+'\n')
