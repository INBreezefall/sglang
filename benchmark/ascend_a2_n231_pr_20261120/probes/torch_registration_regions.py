"""Register the mapped prefix of each PyTorch NPU allocation covering SGLang's transfer descriptors."""
from bisect import bisect_right
import torch

def registration_regions(regions):
    segments=[]
    for item in torch.npu.memory_snapshot():
        if item.get('is_expandable'):
            for block in item['blocks']:
                if block['state'].startswith('active'):
                    segments.append((int(block['address']),int(block['size'])))
        else:
            segments.append((int(item['address']),int(item['total_size'])))
    segments.sort()
    starts=[addr for addr,_ in segments]
    spans={};mapped=0;passthrough=0
    for address,length in regions:
        address,length=int(address),int(length)
        if length==0:continue
        if address<=0 or length<0:raise ValueError('Invalid memory region')
        end=address+length;idx=bisect_right(starts,address)-1
        base=address
        if idx>=0 and end<=segments[idx][0]+segments[idx][1]:
            base=segments[idx][0];mapped+=1
        else:passthrough+=1
        spans[base]=max(spans.get(base,base),end)
    print(f'ASCEND_TORCH_REGISTRATION mapped_views={mapped} passthrough={passthrough} allocations={len(spans)}',flush=True)
    return [(base,end-base) for base,end in spans.items()]
