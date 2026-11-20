"""Single-feature comparisons using real paged NPU allocators and host or device storage."""
import contextlib,hashlib,json,os,sys,tempfile,time
from array import array
from functools import partial
from pathlib import Path
from types import SimpleNamespace as N
from unittest.mock import patch
import torch,torch_npu
from sglang.srt.runtime_context import get_context,get_memory,get_parallel
from sglang.srt.hardware_backend.npu.allocator_npu import NPUPagedTokenToKVPoolAllocator
from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMLATokenToKVPool
from sglang.srt.mem_cache.base_prefix_cache import InsertParams,MatchPrefixParams
from sglang.srt.mem_cache.kv.alloc.demand import KVDemand
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.test.kits.unified_radix_cache.fixtures import build_fixture,match_prefix
from sglang.test.kits.unified_radix_cache.config import CacheConfig
feature,flag,out=int(sys.argv[1]),int(sys.argv[2]),Path(sys.argv[3])
torch.npu.set_device(0);torch.set_num_threads(1);torch.manual_seed(422)
def ms(fn):
 torch.npu.synchronize();start=time.perf_counter_ns();v=fn();torch.npu.synchronize();return v,(time.perf_counter_ns()-start)/1e6
def digest(x):return hashlib.sha256(x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
def fixture(cap):
 with patch('sglang.test.kits.unified_radix_cache.fixtures.TokenToKVPoolAllocator',partial(NPUPagedTokenToKVPoolAllocator,page_size=128)):
  return build_fixture(CacheConfig(page_size=128,kv_size=cap,max_context_len=cap,head_num=1,head_dim=64,num_layers=1),tree_core_backend='python')
def override(**kw):return get_context().override_server_args(model_path='dummy',device='npu',page_size=128,**kw)
rows=[]
if feature==48096:
 os.environ['SGLANG_PYTHON_RADIX_EVICT_TAIL_CHUNK_SIZE']='256' if flag else '0'
 for trial in range(12):
  length=(16+trial%4*4)*128;cap=8192;need=(1+trial%3)*128
  cache,alloc,_=fixture(cap);slots=alloc.alloc(length)
  assert slots.device.type=='npu'
  kv=torch.zeros((cap+128,32),device='npu',dtype=torch.int32)
  payload=(torch.arange(length,dtype=torch.int32).unsqueeze(1).expand(length,32).contiguous()+trial).to('npu')
  kv[slots.long()]=payload
  tokens=array('q',range(1000,1000+length));cache.insert(InsertParams(key=RadixKey(tokens),value=slots))
  before=alloc.available_size();res,t=ms(lambda:cache.evict_for_alloc(KVDemand(full_tokens=before+need)))
  hit=match_prefix(cache,tokens);n=hit.device_prefix_len
  if n:
   actual=cache.tree_core.collect_full_device_indices(hit.last_device_node,cache.root_node_handle())
   assert torch.equal(kv[actual.long()],payload[:n])
  assert alloc.available_size()==before+res.num_tokens_evicted and res.num_tokens_evicted+n==length
  cache.sanity_check()
  rows.append(dict(tokens=length,shortfall=need,evicted=res.num_tokens_evicted,hit=n,ms=t,preserved_values=n*32))
elif feature==48520:
 from sglang.srt.managers.schedule_batch import Req
 from sglang.srt.managers.req_types import FINISH_LENGTH
 from sglang.srt.sampling.sampling_params import SamplingParams
 from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool,ReqToTokenPool
 from sglang.srt.mem_cache.cache_init_params import CacheInitParams
 from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
 from sglang.srt.mem_cache.common import release_kv_cache
 from sglang.srt.mem_cache.unified_cache.components import ComponentType
 from sglang.test.tree_slots import tree_slots_of
 os.environ['SGLANG_PD_DECODE_CACHE_INPUT_ONLY']='1' if flag else '0'
 with override(disaggregation_mode='decode',speculative_algorithm='EAGLE',disaggregation_transfer_backend='mooncake',disaggregation_decode_enable_radix_cache=True):
  for trial in range(12):
   capacity=8192;prompt=(512,1025,2049,3073)[trial%4];outlen=(256,512,768)[trial%3];total=prompt+outlen
   rp=ReqToTokenPool(4,8192,'npu',enable_memory_saver=False)
   pool=MHATokenToKVPool(size=capacity,page_size=128,dtype=torch.float32,head_num=1,head_dim=16,layer_num=1,device='npu',enable_memory_saver=False)
   alloc=NPUPagedTokenToKVPoolAllocator(size=capacity,page_size=128,dtype=torch.float32,device='npu',kvcache=pool,need_sort=False)
   cache=UnifiedRadixCache(CacheInitParams(disable=False,req_to_token_pool=rp,token_to_kv_pool_allocator=alloc,page_size=128,is_eagle=True,tree_slots=tree_slots_of((ComponentType.FULL,))))
   ids=array('q',range(1,total+1));req=Req(rid='r'+str(trial),origin_input_text='',origin_input_ids=ids[:prompt],sampling_params=SamplingParams(temperature=0,max_new_tokens=outlen))
   assert rp.alloc([req]) is not None
   req.output_ids=ids[prompt:];req.full_untruncated_fill_ids=ids;req.extend_end=total
   slots=alloc.alloc((total+127)//128*128);rp.write((req.kv.req_pool_idx,slice(0,total)),slots[:total])
   req.kv.kv_committed_len=req.kv.kv_allocated_len=total;req.last_node=cache.root_node_handle();req.lock=None
   marker=pool.get_key_buffer(0);marker[slots[:total].long()]=trial+1
   req.finished_reason=FINISH_LENGTH(length=outlen);_,lat=ms(lambda:release_kv_cache(req,cache,checkpoint=True))
   def matched(seq):return cache.match_prefix(MatchPrefixParams(key=RadixKey(array('q',seq),is_bigram=True)))
   full=matched(ids);input_hit=matched(ids[:prompt]).device_prefix_len
   if full.device_prefix_len:
    keep=cache.tree_core.collect_full_device_indices(full.last_device_node,cache.root_node_handle())
    assert bool((marker[keep.long()]==trial+1).all().item())
   retained=capacity-alloc.available_size();assert retained==full.device_prefix_len
   cache.sanity_check();rows.append(dict(input_tokens=prompt,output_tokens=outlen,full_hit=full.device_prefix_len,input_hit=input_hit,retained_slots=retained,free_slots=alloc.available_size(),finish_ms=lat))
elif feature==48538:
 from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
 from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
 with override(enable_hierarchical_cache=True,speculative_algorithm='EAGLE'),get_parallel().override(attn_dcp_rank=0):
  def pool(dtype,rope,index=False,layers=1):
   return NPUMLATokenToKVPool(size=384,page_size=128,dtype=dtype,kv_lora_rank=512,qk_rope_head_dim=rope,layer_num=layers,device='npu',enable_memory_saver=False,index_head_dim=128 if index else None)
  target=pool(torch.bfloat16,64,False,2)
  for dtype,rope,index in [(torch.bfloat16,64,False),(torch.float16,64,False),(torch.bfloat16,32,False),(torch.bfloat16,64,True)]:
   draft=pool(dtype,rope,index);runner=N(token_to_kv_pool=target,spec_algorithm=SpeculativeAlgorithm.EAGLE)
   worker=N(target_worker=N(model_runner=runner),_draft_model_runners=lambda:(N(token_to_kv_pool=draft,model_config=N(num_nextn_predict_layers=1),runner_ids=N(is_fused_draft=False)),))
   plan,t=ms(lambda:BaseSpecWorker._build_hicache_draft_plan(worker));buffer=draft.get_key_buffer(0)
   # Exercise the allocated main K payload independently of the metadata decision.
   buffer.fill_(7);raw,d2h=ms(lambda:buffer.to('cpu',non_blocking=True));r,h2d=ms(lambda:raw.to('npu',non_blocking=True))
   assert torch.equal(buffer,r)
   rows.append(dict(dtype=str(dtype),rope=rope,indexer=index,plan=plan.mode.name,selection_ms=t,device_shape=list(buffer.shape),bytes=buffer.numel()*buffer.element_size(),d2h_ms=d2h,h2d_ms=h2d,hash=digest(r),compatible=(dtype==torch.bfloat16 and rope==64 and not index)))
elif feature==49183:
 from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler as asm
 from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
 with override(enable_hierarchical_cache=True,speculative_algorithm='EAGLE',hicache_mem_layout='layer_first'):
  for cap in (1024,2048,4096):
   pool=MHATokenToKVPool(size=256,page_size=128,dtype=torch.float16,head_num=2,head_dim=64,layer_num=1,device='npu',enable_memory_saver=False)
   ctl=N(mem_pool_host=N(logical_size=cap,size=cap),page_size=128,transfer_layer_id_max=1)
   budget=0.004
   with (get_memory().override(hicache_draft_size=budget) if flag else contextlib.nullcontext()):
    (specs,entries),tm=ms(lambda:asm.build_full_draft_pools(draft_kv_pool=pool,tree_cache=N(cache_controller=ctl)))
   hp=entries[0].host_pool;h=hp.kv_buffer
   h.fill_(5);d,t=ms(lambda:h.to('npu',non_blocking=True));back,rev=ms(lambda:d.to('cpu',non_blocking=True))
   assert torch.equal(h,back) and hp.logical_size>=cap
   rows.append(dict(target_tokens=cap,budget_gb=budget if flag else 0,actual_tokens=hp.logical_size,host_bytes=h.numel()*h.element_size(),build_ms=tm,h2d_ms=t,d2h_ms=rev,hash=digest(back)))
elif feature==48599:
 from sglang.srt.mem_cache.unified_cache.components import ComponentType
 os.environ['SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND']='python'
 for k in range(8):
  page=128;n=(1+k%4)*page;cache,allocator,_=fixture(4096);core=cache.tree_core;core.set_hicache_enabled()
  token_a=array('q',range(n));token_b=array('q',range(9000,9000+n));slots=allocator.alloc(n)
  cache.insert(InsertParams(key=RadixKey(token_a),value=slots));found=match_prefix(cache,token_a);node=core.node_by_id(found.last_device_node)
  value=torch.arange(n*32,dtype=torch.int32).reshape(n,32).to('npu')
  host=torch.empty((n,32),dtype=torch.int32,pin_memory=True);saved_slots=torch.arange(n,dtype=torch.int64)
  def writeback(*args,**kwargs):
   host.copy_(value,non_blocking=True);torch.npu.synchronize();return saved_slots
  cache._execute_kv_backup=writeback
  action=core._build_backup_kv_action(node,write_back=True)
  count,wb=ms(lambda:cache._execute_and_commit_kv_backup(action,write_back=True))
  assert count==n
  cache._finish_write_through_ack(node.id)
  core.demote(node.id)
  with tempfile.TemporaryFile(dir=out.parent) as file:
   file.write(host.numpy().tobytes());file.seek(0);separate=torch.empty_like(host,pin_memory=True)
   assert file.readinto(memoryview(separate.numpy()).cast('B'))==n*32*4
   ins,pre=ms(lambda:core.insert_host(core.root_node.id,RadixKey(token_b),saved_slots+n,[str(i) for i in range(n//page)]))
  a,b=match_prefix(cache,token_a),match_prefix(cache,token_b)
  assert a.host_hit_length==b.host_hit_length==n
  aa,tt=ms(lambda:host.to('npu',non_blocking=True));bb,uu=ms(lambda:separate.to('npu',non_blocking=True))
  assert torch.equal(aa,value) and torch.equal(bb,value)
  rows.append(dict(tokens=n,writeback_origin=[getattr(a,'host_hit_write_back',0),getattr(a,'host_hit_prefetch',0)],prefetch_origin=[getattr(b,'host_hit_write_back',0),getattr(b,'host_hit_prefetch',0)],host_hits=[a.host_hit_length,b.host_hit_length],backup_ms=wb,insert_ms=pre,restore_ms=[tt,uu],hash=digest(aa)))
else:raise ValueError(feature)
out.write_text(json.dumps(dict(feature=feature,flag=flag,sha=os.environ['ARM_SHA'],device=torch.npu.get_device_name(0),rows=rows),indent=2)+'\n')
