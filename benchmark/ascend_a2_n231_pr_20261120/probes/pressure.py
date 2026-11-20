"""Recreate a mixture of host- and device-resident prefixes on an isolated, live GLM-5.2 rank."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import random
import secrets
import statistics
import time
from urllib.parse import urlsplit
import aiohttp

def digest(seq):
    return hashlib.sha256(json.dumps(seq,separators=(',',':')).encode()).hexdigest()
def prefix(seed,length):
    rng=random.Random(seed)
    return [rng.randrange(512,22000) for _ in range(length)]
async def main(a):
    nonce=secrets.randbelow(1<<25)+1;counter=0;rows=[];waves=[]
    urls=[a.prefill,a.decode]
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=210)) as client:
        async def flush(ep):
            async with client.post(ep+'/flush_cache?timeout=20') as r:r.raise_for_status()
        async def ask(ids,phase,rep,slot,count=4):
            nonlocal counter
            counter+=1;rid=f'pressure-{nonce}-{counter}'
            body=dict(input_ids=ids,routed_dp_rank=0,rid=rid,bootstrap_host=urlsplit(a.prefill).hostname,
                      bootstrap_port=a.bootstrap,bootstrap_room=(nonce<<24)+counter,
                      sampling_params=dict(max_new_tokens=count,ignore_eos=True,temperature=0))
            start=time.monotonic()
            async def pre():
                async with client.post(urls[0]+'/generate',json={**body,'stream':False}) as r:
                    r.raise_for_status();v=await r.json();m=v['meta_info']
                    assert (m.get('finish_reason') or {}).get('type')!='abort'
                    return m
            async def dec():
                first=None;last=None;value=None;token_ids=[]
                async with client.post(urls[1]+'/generate',json={**body,'stream':True}) as r:
                    r.raise_for_status()
                    async for line in r.content:
                        if not line.startswith(b'data:'):continue
                        raw=line[5:].strip()
                        if raw==b'[DONE]':break
                        if not raw:continue
                        v=json.loads(raw)
                        if v.get('error'):raise RuntimeError(str(v['error'])[:240])
                        m=v['meta_info'];assert (m.get('finish_reason') or {}).get('type')!='abort'
                        count_so_far=m.get('completion_tokens',0);piece=v.get('output_ids') or []
                        if piece and len(piece)==count_so_far:token_ids=list(piece)
                        elif piece and len(token_ids)+len(piece)==count_so_far:token_ids.extend(piece)
                        if count_so_far>0 and first is None:first=time.monotonic()
                        last=time.monotonic();value=v
                assert first is not None and value['meta_info']['completion_tokens']==count
                assert len(token_ids)==count,(len(token_ids),count)
                return dict(ttft_s=first-start,decode_s=last-first,output_ids=token_ids,rank=value['meta_info'].get('dp_rank'))
            p,d=await asyncio.gather(pre(),dec())
            if a.single_rank:
                assert p.get('dp_rank') in (0,None) and d['rank'] in (0,None)
            else:
                assert p['dp_rank']==d['rank']==0
            entry=dict(rep=rep,phase=phase,slot=slot,length=len(ids),input_hash=digest(ids),elapsed_s=time.monotonic()-start,
                       cache=p.get('cached_tokens_details') or {},cached=p.get('cached_tokens'),**d)
            rows.append(entry)
        for rep in range(a.repeats):
            await asyncio.gather(*(flush(ep) for ep in urls))
            base=864201+rep*100
            # The 8192-token device pool holds four long prefixes, while the 4x host pool holds sixteen.
            for wave in range(4):
                await asyncio.gather(*(ask(prefix(base+i,2049),'seed',rep,i) for i in (2*wave,2*wave+1)))
            await asyncio.sleep(1)
            await flush(urls[1])
            sequence=list(range(8));random.Random(873+rep).shuffle(sequence)
            start=time.monotonic()
            await asyncio.gather(*(ask(prefix(base+i,2049)+prefix(base+200+i,48+(i%3)*48),'mixed',rep,i,8) for i in sequence))
            took=time.monotonic()-start;match=[r for r in rows if r['rep']==rep and r['phase']=='mixed']
            waves.append(dict(rep=rep,wall_s=took,mean_ttft_s=statistics.mean(r['ttft_s'] for r in match),
                              mean_finish_s=statistics.mean(r['elapsed_s'] for r in match),
                              host_hit_tokens=sum((r['cache'].get('host') or 0) for r in match),
                              device_hit_tokens=sum((r['cache'].get('device') or 0) for r in match)))
            await asyncio.gather(*(flush(ep) for ep in urls))
            await asyncio.gather(*(ask(prefix(base+i,2049)+prefix(base+200+i,48+(i%3)*48),'cold-reference',rep,i,8) for i in sequence))
            cold={r['slot']:r for r in rows if r['rep']==rep and r['phase']=='cold-reference'}
            waves[-1].update(cold_reference_requests=len(cold),cold_reference_cached=sum(r['cached'] or 0 for r in cold.values()),
                restored_output_matches_cold=sum(r['output_ids']==cold[r['slot']]['output_ids'] for r in match))
            Path(a.output).write_text(json.dumps(dict(waves=waves,rows=rows),indent=2)+'\n')
        print(json.dumps(waves),flush=True)
p=argparse.ArgumentParser();p.add_argument('--prefill',required=True);p.add_argument('--decode',required=True);p.add_argument('--bootstrap',type=int,default=18784);p.add_argument('--output',required=True);p.add_argument('--repeats',type=int,default=3);p.add_argument('--single-rank',action='store_true')
if __name__=='__main__':asyncio.run(main(p.parse_args()))
