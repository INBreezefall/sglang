"""Exercise eager and captured Ascend DSA indexer calls on two requests with changing KV lengths."""
import json,os,sys,time
from pathlib import Path
from types import SimpleNamespace as N
import torch,torch_npu
torch.npu.set_device(0);torch.set_num_threads(1)
import custom_ops
from sglang.srt.hardware_backend.npu.attention.ascend_backend import AscendAttnBackend,ForwardMetadata
from sglang.srt.layers.attention.dsa.dsa_npu_indexer import DSANPUIndexerMixin
from sglang.srt.model_executor.forward_mode import ForwardMode
mode=sys.argv[2];dest=Path(sys.argv[1]);layer_count=int(sys.argv[3]) if len(sys.argv)>3 else 1
g=torch.Generator().manual_seed(48562);bs=2;heads=16;dim=128;page=128;pages=20;top=2048
q=torch.randint(-8,9,(bs,heads,dim),dtype=torch.int8,generator=g).to("npu")
k=torch.randint(-8,9,(bs*pages,page,1,dim),dtype=torch.int8,generator=g).to("npu")
weight=(torch.rand((bs,heads),generator=g)+0.5).to("npu").half()
qscale=torch.ones((bs,heads),dtype=torch.float16,device="npu")
kscale=torch.ones((bs*pages,page,1),dtype=torch.float16,device="npu")
queries=[q]+[torch.randint(-8,9,q.shape,dtype=torch.int8,generator=g).to("npu") for _ in range(layer_count-1)]
keys=[k]+[torch.randint(-8,9,k.shape,dtype=torch.int8,generator=g).to("npu") for _ in range(layer_count-1)]
weights=[weight]+[(torch.rand(weight.shape,generator=g)+.5).to("npu").half() for _ in range(layer_count-1)]
bfqs=[x.bfloat16() for x in queries];bfks=[x.bfloat16() for x in keys];bfws=[x.bfloat16() for x in weights]
table=torch.arange(bs*pages,dtype=torch.int32,device="npu").reshape(bs,pages)
cu=torch.arange(1,bs+1,dtype=torch.int32,device="npu")
lens=torch.tensor([2304,2464],dtype=torch.int32,device="npu")
def ref(i):
    out=torch_npu.npu_lightning_indexer(query=bfqs[i],key=bfks[i],weights=bfws[i],actual_seq_lengths_query=cu,actual_seq_lengths_key=lens,block_table=table,layout_query="TND",layout_key="PA_BSND",sparse_count=top,sparse_mode=3)
    return out[0].view(bs,top)
if mode=="int8":
    from sglang.srt.hardware_backend.npu.attention.ascend_backend import _quant_indexer_metadata_static_kwargs
    backend=object.__new__(AscendAttnBackend);backend.use_int8_quant_indexer=True;backend.device="npu:0"
    backend.quant_indexer_metadata_kwargs=_quant_indexer_metadata_static_kwargs(index_topk=top,head_dim=dim)
    fm=ForwardMetadata(actual_seq_lengths_q=cu,seq_lens=lens,seq_lens_cpu_int=None)
    backend.forward_metadata=fm
    indexer=N(n_heads=heads,index_topk=top)
    backend._init_graph_quant_indexer_metadata(metadata=fm,bs=bs,seq_lens=lens)
    hp=fm.quant_indexer_metadata.data_ptr()
    def call(i):
        return DSANPUIndexerMixin._npu_vendor_quant_indexer(indexer,query=queries[i],query_scale=qscale,key=keys[i],key_scale=kscale,weights=weights[i],block_table=table,metadata=fm.quant_indexer_metadata,cu_q_lens=fm.quant_indexer_cu_q_lens,kv_lens=fm.quant_indexer_kv_lens)
    def refresh():
        backend._refresh_graph_quant_indexer_metadata(metadata=fm,bs=bs)
        assert hp==fm.quant_indexer_metadata.data_ptr()
else:
    call=ref
    def refresh():pass
def group():return [call(i) for i in range(layer_count)]
refresh();a=group();torch.npu.synchronize()
graph=torch.npu.NPUGraph()
with torch.npu.graph(graph):
    captured=group()
rows=[]
for lens_cpu in ([65,127],[2304,2464],[2513,2113],[333,987]):
    lens.copy_(torch.tensor(lens_cpu,dtype=torch.int32));refresh()
    oracles=[ref(i).cpu() for i in range(layer_count)];current=[x.cpu() for x in group()]
    graph.replay();torch.npu.synchronize();candidates=[x.cpu() for x in captured]
    overlap=[];counts=[]
    for oracle,eager,candidate in zip(oracles,current,candidates):
        for j,l in enumerate(lens_cpu):
            def valid(t):
                vs=t[j][t[j]>=0].tolist();assert len(vs)==min(l,top) and len(set(vs))==len(vs) and max(vs)<l,(l,len(vs),vs[:8]);return set(vs)
            a,b,c=valid(oracle),valid(eager),valid(candidate)
            assert b==c
            overlap.append(len(a&b)/len(a));counts.append(len(b))
    ts=[]
    for r in range(14):
        torch.npu.synchronize();t0=time.perf_counter_ns();refresh();graph.replay();torch.npu.synchronize()
        if r>=3:ts.append((time.perf_counter_ns()-t0)/1e6)
    rows.append(dict(kv_lengths=lens_cpu,valid=counts,topk_overlap=overlap,graph_equals_eager=True,refresh_and_replay_ms=ts))
dest.write_text(json.dumps(dict(sha=os.environ["ARM_SHA"],mode=mode,device=torch.npu.get_device_name(),layers=layer_count,heads=heads,topk=top,rows=rows),indent=2)+"\n")
