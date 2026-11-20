"""Feed fresh long inputs at one prefill rank and measure full-model execution under the fixed 8192-token cap."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import random
import secrets
import time
from urllib.parse import urlsplit
import requests

def main(a):
    rows=[];nonce=secrets.randbelow(1<<28)+1
    def post(ep,body):
        r=requests.post(ep+'/generate',json=body,timeout=(3,180));r.raise_for_status()
        v=r.json();m=v['meta_info'];assert (m.get('finish_reason') or {}).get('type')!='abort'
        return v
    for it,length in enumerate((3073,4097,6145,7169,7169,6145,4097,3073)):
        for ep in (a.prefill,a.decode):
            requests.post(ep+'/flush_cache?timeout=20',timeout=(3,30)).raise_for_status()
        rng=random.Random(485400+it)
        seq=[rng.randrange(512,22000) for _ in range(length)]
        body=dict(input_ids=seq,routed_dp_rank=0,stream=False,sampling_params=dict(temperature=0,ignore_eos=True,max_new_tokens=4),
                  rid=f'moe-memory-{nonce}-{it}',bootstrap_host=urlsplit(a.prefill).hostname,bootstrap_port=a.bootstrap,
                  bootstrap_room=(nonce<<22)+it)
        start=time.monotonic()
        with ThreadPoolExecutor(max_workers=2) as ex:
            pf,df=[ex.submit(post,ep,body) for ep in (a.prefill,a.decode)]
            pre,dec=pf.result(),df.result()
        ids=dec.get('output_ids');m=dec['meta_info']
        assert len(ids)==m['completion_tokens']==4
        if a.single_rank:assert m.get('dp_rank') in (None,0) and pre['meta_info'].get('dp_rank') in (None,0)
        else:assert m['dp_rank']==pre['meta_info']['dp_rank']==0
        rows.append(dict(input_tokens=length,input_hash=hashlib.sha256(json.dumps(seq,separators=(',',':')).encode()).hexdigest(),
           output_ids=ids,prefill_cache=pre['meta_info'].get('cached_tokens'),wall_s=time.monotonic()-start))
        Path(a.output).write_text(json.dumps(dict(rows=rows),indent=2)+'\n')
    print('Completed',len(rows),'uncached full-model requests',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prefill',required=True);p.add_argument('--decode',required=True);p.add_argument('--bootstrap',type=int,default=18784);p.add_argument('--output',required=True);p.add_argument('--single-rank',action='store_true');main(p.parse_args())
