"""Record real pool geometry and optional per-forward allocator peaks in both arms."""
import functools
import importlib.machinery
import json
import os
from pathlib import Path
import sys
import time

folder = os.environ.get('GLM_PR_POOL_PROBE')
runner_module = 'sglang.srt.model_executor.model_runner'
tile_module = 'sglang.srt.layers.moe.prefill_moe_tile'
host_module = 'sglang.srt.mem_cache.pool_host.mla'
instrument_memory = os.environ.get('GLM_PR_FORWARD_MEMORY')=='1'
tile_calls = 0

def install(module):
    if module.__name__==host_module:
        from restore_audit import install as observe_restore
        observe_restore(module,folder)
        return
    if module.__name__=='sglang.srt.managers.data_parallel_controller':
        from route_audit import install as instrument_dispatch
        instrument_dispatch(module,folder)
        return
    if module.__name__ == tile_module:
        original=module.tiled_experts
        @functools.wraps(original)
        def tiled(*args,**kwargs):
            global tile_calls
            tile_calls+=1
            return original(*args,**kwargs)
        module.tiled_experts=tiled
        return
    import torch
    from sglang.srt.runtime_context import get_parallel
    cls=module.ModelRunner
    original=cls.forward
    @functools.wraps(original)
    def forward(self, forward_batch, *args, **kwargs):
        if os.environ.get('GLM_PR_AUDIT_WEIGHTS') and not getattr(self,'_pr_live_weights_audited',False):
            from weight_audit import audit
            audit(self,folder)
            self._pr_live_weights_audited=True
        rank=get_parallel().tp_rank
        mode=forward_batch.forward_mode
        tokens=forward_batch.input_ids.numel() if forward_batch.input_ids is not None else 0
        file=Path(folder)/('forward-rank-%s-%d.jsonl' % (rank,os.getpid()))
        if not getattr(self,'_pr_pool_recorded',False):
            pool=self.token_to_kv_pool
            fields={}
            for key in ('kv_buffer','k_buffer','v_buffer','index_k_buffer','index_k_scale_buffer'):
                value=getattr(pool,key,None)
                if isinstance(value,torch.Tensor):
                    fields[key]=dict(shape=list(value.shape),dtype=str(value.dtype),bytes=value.numel()*value.element_size())
                elif isinstance(value,(list,tuple)) and len(value) and isinstance(value[0],torch.Tensor):
                    fields[key]=dict(count=len(value),shape=list(value[0].shape),dtype=str(value[0].dtype),
                                     bytes=sum(t.numel()*t.element_size() for t in value))
            with file.open('a') as f:
                print(json.dumps(dict(event='pool',rank=rank,pool_type=type(pool).__name__,buffers=fields,
                                      capacity=getattr(self,'max_total_num_tokens',None),role=os.environ['GLM_PR_ROLE'])),file=f)
            self._pr_pool_recorded=True
        measure=instrument_memory and mode.is_extend() and tokens>0 and (Path(folder)/'measure-memory').is_file()
        before=0
        if measure:
            torch.npu.synchronize()
            before=torch.npu.memory_allocated()
            torch.npu.reset_peak_memory_stats()
        global tile_calls
        prior_calls=tile_calls
        result=original(self,forward_batch,*args,**kwargs)
        if measure:
            torch.npu.synchronize()
            row=dict(event='memory',tokens=tokens,rank=rank,batch=forward_batch.batch_size,
                     before=before,peak=torch.npu.max_memory_allocated(),reserved=torch.npu.max_memory_reserved(),
                     tiled_calls=tile_calls-prior_calls,epoch=time.time())
            with file.open('a') as f:print(json.dumps(row),file=f)
        if result.can_run_graph and not getattr(self,'_pr_observed_graph',False):
            with file.open('a') as f:print(json.dumps(dict(event='graph',rank=rank,tokens=tokens)),file=f)
            self._pr_observed_graph=True
        return result
    cls.forward=forward

class Loader:
    def __init__(self,loader):self.loader=loader
    def create_module(self,spec):return self.loader.create_module(spec)
    def exec_module(self,module):
        self.loader.exec_module(module)
        install(module)
class Finder:
    @staticmethod
    def find_spec(name,path,target=None):
        tracked=(runner_module,tile_module)
        if os.environ.get('GLM_PR_CHECK_HOST_RESTORES')=='1':tracked+=(host_module,)
        if os.environ.get('GLM_PR_ROUTE_AUDIT')=='1':tracked+=('sglang.srt.managers.data_parallel_controller',)
        if name not in tracked:return None
        spec=importlib.machinery.PathFinder.find_spec(name,path)
        if spec is not None:spec.loader=Loader(spec.loader)
        return spec
if folder:
    sys.meta_path.insert(0,Finder())
    # Mooncake's per-thread device fixture also uses sitecustomize.
    if os.environ.get('ASCEND_BENCH_MOONCAKE_SUPPORT'):
        import runpy
        runpy.run_path('/bench/moon-fixture/sitecustomize.py')
