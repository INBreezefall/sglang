"""Record real vendor INT8 indexer dispatch during graph construction without reading any device value."""
import functools
import importlib.machinery
import json
import os
from pathlib import Path
import sys

import base_observe

target='sglang.srt.layers.attention.dsa.dsa_npu_indexer'
seen=set()

def install(module):
    if os.environ.get('GLM_PR_ROLE')!='decode':return
    cls=module.DSANPUIndexerMixin
    original=getattr(cls,'_npu_vendor_quant_indexer',None)
    if original is None:return
    import torch
    from sglang.srt.model_executor.runner_utils.capture_mode import get_is_capture_mode
    from sglang.srt.runtime_context import get_parallel
    output=Path(os.environ['GLM_PR_POOL_PROBE'])/f'forward-rank-indexer-{os.getpid()}.jsonl'
    @functools.wraps(original)
    def forward(self,**kwargs):
        result=original(self,**kwargs)
        context=bool(get_is_capture_mode())
        query=kwargs['query']
        if len(seen)>=24:return result
        state=getattr(torch.npu,'is_current_stream_capturing',None)
        try:capturing=bool(state()) if state else None
        except RuntimeError:capturing=None
        key=(context,capturing,query.shape[0])
        if key in seen:return result
        seen.add(key)
        values={name:dict(shape=list(kwargs[name].shape),dtype=str(kwargs[name].dtype))
                for name in ('query','key','key_scale','metadata','cu_q_lens','kv_lens')}
        row=dict(event='vendor-int8-indexer',rank=get_parallel().tp_rank,capture_context=context,
                 stream_capturing=capturing,tensors=values,output_shape=list(result.shape))
        with output.open('a') as f:print(json.dumps(row),file=f)
        return result
    cls._npu_vendor_quant_indexer=forward

class Loader:
    def __init__(self,original):self.original=original
    def create_module(self,spec):return self.original.create_module(spec)
    def exec_module(self,module):
        self.original.exec_module(module)
        install(module)

class Finder:
    @staticmethod
    def find_spec(name,path,target_module=None):
        if name!=target:return None
        spec=importlib.machinery.PathFinder.find_spec(name,path)
        if spec is not None:spec.loader=Loader(spec.loader)
        return spec

if os.environ.get('GLM_PR_POOL_PROBE'):sys.meta_path.insert(0,Finder())
