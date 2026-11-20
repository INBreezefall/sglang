"""Fixed-token PD replay. Send matching native requests concurrently to the two roles."""
import argparse,asyncio,hashlib,json,random,secrets,statistics,time
from pathlib import Path
from urllib.parse import urlsplit
import aiohttp

def digest(value):
    return hashlib.sha256(json.dumps(value,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()
def workload(sid,turn):
    sizes=[4096,3072,2560,2048,1536,1280,1024,768]
    rng=random.Random(119_000+sid)
    return [rng.randrange(512,22000) for _ in range(sizes[sid%8]+turn*128)]
def qtile(x,f):
    x=sorted(x);i=f*(len(x)-1);l=int(i);return x[l]+(i-l)*(x[min(l+1,len(x)-1)]-x[l])
async def main(args):
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    nonce=secrets.randbelow(1<<27)+1;clock=time.monotonic;rows=[]
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=420),connector=aiohttp.TCPConnector(limit=96)) as client:
        async def ask(sid,turn,warm=False):
            ids=workload(sid,turn) if not warm else workload(sid+8192,turn)
            session=("warm-" if warm else "s-")+str(sid)
            body={"input_ids":ids,"session_id":session,"sampling_params":{"temperature":0,"max_new_tokens":args.output_tokens,"ignore_eos":True}}
            h=digest(body);rid=f"ab-{nonce}-{int(warm)}-{turn}-{sid}"
            body.update(rid=rid,bootstrap_host=urlsplit(args.prefill).hostname,bootstrap_port=args.bootstrap,bootstrap_room=(nonce<<24)+(int(warm)<<20)+turn*1024+sid)
            t0=clock()
            async def prefill():
                async with client.post(args.prefill+"/generate",json=dict(body,stream=False)) as r:
                    if r.status!=200:raise RuntimeError("prefill HTTP "+str(r.status)+" "+(await r.text())[-400:])
                    data=await r.json();meta=data.get("meta_info",{})
                    if data.get("error") or (meta.get("finish_reason") or {}).get("type")=="abort":raise RuntimeError("prefill aborted "+str(meta.get("finish_reason")))
                    return {k:meta.get(k) for k in ("dp_rank","cached_tokens","cached_tokens_details","prompt_tokens","completion_tokens")}
            async def decode():
                first=None;last=None;received=None;pieces=0;token_ids=[]
                async with client.post(args.decode+"/generate",json=dict(body,stream=True)) as r:
                    if r.status!=200:raise RuntimeError("decode HTTP "+str(r.status)+" "+(await r.text())[-400:])
                    async for line in r.content:
                        if not line.startswith(b"data:"):continue
                        msg=line[5:].strip()
                        if msg==b"[DONE]":break
                        if not msg:continue
                        data=json.loads(msg)
                        if "error" in data:raise RuntimeError(str(data["error"])[:400])
                        meta=data.get("meta_info",{})
                        if (meta.get("finish_reason") or {}).get("type")=="abort":raise RuntimeError("decode aborted "+str(meta.get("finish_reason")))
                        ids=data.get("output_ids") or []
                        count=meta.get("completion_tokens",0)
                        if ids and len(ids)==count:token_ids=list(ids)
                        elif ids and len(token_ids)+len(ids)==count:token_ids.extend(ids)
                        if count>0 and first is None:first=clock()
                        last=clock();received=data;pieces+=1
                assert first is not None and received is not None,(sid,turn,"empty stream")
                m=received["meta_info"];count=m.get("completion_tokens");assert count==args.output_tokens,(count,args.output_tokens)
                return dict(first_ms=(first-t0)*1000,decode_ms=(last-first)*1000,tokens=count,events=pieces,output_hash=digest(token_ids if len(token_ids)==count else received.get("text")),hashed_tokens=len(token_ids),rank=m.get("dp_rank"),cached=m.get("cached_tokens"))
            p,d=await asyncio.gather(prefill(),decode())
            return dict(sid=sid,turn=turn,input_hash=h,input_tokens=len(ids),p=p,d=d,completed_ms=(clock()-t0)*1000)
        for ep in (args.prefill,args.decode):
            async with client.post(ep+"/flush_cache?timeout=10") as r:
                if r.status!=200:raise RuntimeError("cache flush failed "+str(r.status))
        warm_count=min(args.sessions,8)
        await asyncio.gather(*(ask(s,0,True) for s in range(warm_count)))
        for ep in (args.prefill,args.decode):
            async with client.post(ep+"/flush_cache?timeout=10") as r:
                if r.status!=200:raise RuntimeError("post-warmup flush failed "+str(r.status))
        formal_start_epoch=time.time()
        t0=clock()
        requested=args.total_requests or args.turns*args.sessions
        if not 1<=requested<=args.turns*args.sessions:raise ValueError('Invalid request count')
        for turn in range(args.turns):
            order=list(range(args.sessions));random.Random(9100+turn).shuffle(order)
            order=order[:requested-len(rows)]
            if not order:break
            wave=await asyncio.gather(*(ask(s,turn) for s in order))
            rows.extend(wave)
            out.with_suffix(".partial.json").write_text(json.dumps(rows)+"\n")
        wall=clock()-t0
        tt=[r["d"]["first_ms"] for r in rows];decode=[(r["d"]["tokens"]-1)*1000/r["d"]["decode_ms"] for r in rows if r["d"]["tokens"]>1 and r["d"]["decode_ms"]>0]
        result=dict(requested=requested,formal_start_epoch=formal_start_epoch,formal_end_epoch=time.time(),prefill_sha=args.prefill_sha,decode_sha=args.decode_sha,sessions=args.sessions,turns=args.turns,output_limit=args.output_tokens,warmups=warm_count,wall_s=wall,output_tps=sum(r["d"]["tokens"] for r in rows)/wall,mean_ttft_ms=statistics.mean(tt),p95_ttft_ms=qtile(tt,.95),mean_decode_tps_per_request=statistics.mean(decode) if decode else None,rows=rows)
        out.write_text(json.dumps(result,indent=2)+"\n")
        print(json.dumps({k:v for k,v in result.items() if k!="rows"}),flush=True)
p=argparse.ArgumentParser()
for k in ("prefill","decode","prefill-sha","decode-sha","output"):p.add_argument("--"+k,required=True)
p.add_argument("--bootstrap",type=int,required=True);p.add_argument("--sessions",type=int,default=16);p.add_argument("--turns",type=int,default=3);p.add_argument("--output-tokens",type=int,default=24)
p.add_argument('--total-requests',type=int,default=0)
asyncio.run(main(p.parse_args()))
