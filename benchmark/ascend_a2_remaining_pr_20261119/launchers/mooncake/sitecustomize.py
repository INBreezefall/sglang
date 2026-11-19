"""Common launch fixture for two matched SGLang services using the Ascend Direct wheel."""
import functools
import importlib.machinery
import os
import sys
import threading

TARGET='sglang.srt.distributed.device_communicators.mooncake_transfer_engine'
mode=os.environ.get('ASCEND_BENCH_MOONCAKE_SUPPORT','')
def install(module):
    import torch
    cls=module.MooncakeTransferEngine
    context=threading.local()
    original_init=cls.initialize
    original_deadline=module.run_with_deadline
    @functools.wraps(original_init)
    def init(self,*args,**kwargs):
        context.gpu_id=self.gpu_id
        try:return original_init(self,*args,**kwargs)
        finally:del context.gpu_id
    @functools.wraps(original_deadline)
    def deadline(fn,**kwargs):
        if not hasattr(context,'gpu_id'):
            return original_deadline(fn,**kwargs)
        device=context.gpu_id
        def initialize_on_requested_device():
            torch.npu.set_device(device)
            return fn()
        return original_deadline(initialize_on_requested_device,**kwargs)
    cls.initialize=init;module.run_with_deadline=deadline
    if mode=='thread-regions':
        from acl_regions import registration_regions
        orig_register=cls.batch_register;orig_deregister=cls.batch_deregister
        @functools.wraps(orig_register)
        def register(self,ptrs,lengths):
            spans=registration_regions(list(zip(ptrs,lengths)))
            if not spans:return orig_register(self,ptrs,lengths)
            base,sizes=map(list,zip(*spans))
            result=orig_register(self,base,sizes)
            if not hasattr(self,'_bench_allocation_spans'):self._bench_allocation_spans={}
            self._bench_allocation_spans[tuple(ptrs)]=base
            print(f'ASCEND_BENCH_MOON_REGISTRATION views={len(ptrs)} spans={len(base)}',flush=True)
            return result
        @functools.wraps(orig_deregister)
        def deregister(self,ptrs):
            spans=getattr(self,'_bench_allocation_spans',{}).pop(tuple(ptrs),ptrs)
            return orig_deregister(self,spans)
        cls.batch_register=register;cls.batch_deregister=deregister
    print('ASCEND_BENCH_MOON_SUPPORT '+mode,flush=True)
class Loader:
    def __init__(self,upstream):self.upstream=upstream
    def create_module(self,spec):return self.upstream.create_module(spec)
    def exec_module(self,module):
        self.upstream.exec_module(module)
        install(module)
class Finder:
    @staticmethod
    def find_spec(name,path,target=None):
        if name!=TARGET:return None
        spec=importlib.machinery.PathFinder.find_spec(name,path)
        if spec is not None:spec.loader=Loader(spec.loader)
        return spec
if mode in ('thread','thread-regions'):sys.meta_path.insert(0,Finder())
