"""Exercise the 78-layer K/V/indexer layouts, tensor views, transfer batching and a nonzero NPU."""
import hashlib,json,os,socket,sys,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import torch,torch_npu
role,host,peer,port,kind,layout,npu,remote_npu,out=sys.argv[1:];port=int(port);sender=role=='send';card=int(npu)
torch.npu.set_device(card);torch.set_num_threads(1)
def server(addr,p):
 s=socket.socket();s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);s.bind((addr,p));return s
if sender:
 s=server(host,port);s.listen(1);s.settimeout(48);sock,w=s.accept();assert w[0]==peer
else:
 for i in range(100):
  try:sock=socket.create_connection((peer,port),timeout=3);break
  except OSError:time.sleep(.3)
 else:raise TimeoutError()
sock.settimeout(100);f=sock.makefile('rwb')
def send(v):f.write((json.dumps(v)+'\n').encode());f.flush()
def recv():return json.loads(f.readline(1<<20))
s=server(host,0);rpc=s.getsockname()[1];s.close()
if kind=='mfzero':
 from memfabric_hybrid import TransferEngine
 en=TransferEngine();rc=int(en.initialize('tcp://'+peer+':20883',host+':'+str(rpc),'Prefill' if sender else 'Decode',card,TransferEngine.TransDataOpType.DEVICE_RDMA))
else:
 from mooncake.engine import TransferEngine
 en=TransferEngine();rc=int(en.initialize(host+':'+str(rpc)+':npu_'+str(card),'P2PHANDSHAKE','ascend',''))
assert rc==0,rc
arrays=[torch.full((78,9,128,width),17+k if sender else 0,device='npu',dtype=torch.uint8) for k,width in enumerate((1024,128,256))]
torch.npu.synchronize()
views=[x[j] for x in arrays for j in range(78)]
regions=views if layout=='views' else arrays
ptr=[t.data_ptr() for t in regions];lens=[t.numel() for t in regions]
if layout=='segments':
 from acl_regions import registration_regions
 spans=registration_regions([(t.data_ptr(),t.numel()) for t in views])
 ptr,lens=map(list,zip(*spans))
if hasattr(en,'batch_register_memory'):reg=int(en.batch_register_memory(ptr,lens))
else:reg=max(int(en.register_memory(a,n)) for a,n in zip(ptr,lens))
desc=[(t.data_ptr(),t[0].numel()*6) for t in views]
send(dict(sid=host+':'+str(en.get_rpc_port()),desc=desc,reg=reg));other=recv()
ans=dict(role=role,card=card,peer_card=int(remote_npu),kind=kind,layout=layout,registration=reg,peer_registration=other['reg'],regions=len(regions),descriptors=len(desc),bytes=sum(n for _,n in desc))
if sender:
 def do_write(batch_size):
  torch.npu.set_device(card)
  t0=time.monotonic();codes=[]
  for i in range(0,len(desc),batch_size):
   a=desc[i:i+batch_size];b=other['desc'][i:i+batch_size]
   code=int(en.batch_transfer_sync_write(other['sid'],[p for p,_ in a],[p for p,_ in b],[n for _,n in a]))
   codes.append(code)
   if code:break
  return dict(batch_size=batch_size,rc=codes,seconds=time.monotonic()-t0)
 results=[]
 if reg==other['reg']==0:
  with ThreadPoolExecutor(1) as pool:
   for size in [234,16,1]:
    result=pool.submit(do_write,size).result(timeout=90);results.append(result)
    if all(v==0 for v in result['rc']):break
 send(results);ans.update(writes=results,verified=recv())
else:
 ans['writes']=recv();torch.npu.synchronize()
 verified=[sum(bool((x[j,:6]==17+k).all().cpu().item()) for j in range(78)) for k,x in enumerate(arrays)]
 ans['verified']=verified;send(verified)
send(1);assert recv()==1;ans['registered_spans']=len(ptr);Path(out).write_text(json.dumps(ans))
