"""Hash the actual live post-load tensors before first inference, outside all request timings."""
import hashlib
import json
from pathlib import Path
import torch
import torch_npu
from sglang.srt.model_loader.loader import PreshardedModelLoader
from sglang.srt.hardware_backend.npu.utils import NPUACLFormat
from sglang.srt.runtime_context import get_parallel

def logical_value(x):
    flat=x.detach()
    if flat.numel()==0:return torch.empty(0,dtype=torch.uint8)
    if flat.device.type=='npu' and flat.dtype==torch.int32 and torch_npu.get_npu_format(flat)==NPUACLFormat.ACL_FORMAT_FRACTAL_NZ:
        flat=(flat if flat.ndim and flat.stride(-1)==1 else flat.unsqueeze(-1)).view(torch.int8)
    return flat.to(device='cpu',copy=True).contiguous().reshape(-1).view(torch.uint8)
def audit(runner,folder):
    rank=get_parallel().tp_rank
    # The already-running dump loaded the earlier observation wrapper. Its only
    # use of this name is labelling records with the global tensor-parallel rank.
    if not hasattr(runner,'tp_rank'):
        runner.tp_rank=rank
    dst=Path(folder)/f'live-weights-rank-{rank}.json'
    if dst.exists():return
    model=runner.model
    named=dict(model.state_dict())
    more=PreshardedModelLoader._collect_extra_tensors(model)
    for key,val in more.items():
        key='extra:'+key
        assert key not in named
        named[key]=val
    rows=[]
    for name,x in sorted(named.items()):
        data=logical_value(x)
        h=hashlib.sha256(memoryview(data.numpy())).hexdigest()
        rows.append(dict(name=name,shape=list(x.shape),dtype=str(x.dtype),bytes=x.numel()*x.element_size(),sha256=h))
        del data
    temp=dst.with_suffix('.partial')
    temp.write_text(json.dumps(dict(rank=rank,tensors=rows,sha256=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest())))
    temp.replace(dst)
