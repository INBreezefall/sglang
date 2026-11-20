"""Exercise short and long scored retrieval on every PD attention rank of a graph-enabled service."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import secrets
from urllib.parse import urlsplit
import aiohttp
from transformers import AutoTokenizer


async def run(args):
    tok = AutoTokenizer.from_pretrained('/model', trust_remote_code=True, local_files_only=True)
    rows = []
    nonce = 1+secrets.randbelow(1<<26)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as client:
        async def ask(rank, group, length):
            count=length+rank%3*5
            entry=19+rank*3
            names=('Apricot','Beryl','Chestnut','Dove','Elder','Fawn','Ginger','Hazel')
            def label(i):return f'{(i*761+4937+rank*911+group*3017)%100000:05d}'
            prompt=names[rank]+' catalogue.\n'+'\n'.join(f'Entry {i:03d} is assigned code {label(i)}.' for i in range(count))
            prompt+=f'\nGive the five-digit code assigned to entry {entry:03d}. Answer with only the five digits.'
            ids=tok.apply_chat_template([dict(role='user',content=prompt)],tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False)
            if isinstance(ids,dict):ids=ids['input_ids']
            if hasattr(ids,'tolist'):ids=ids.tolist()
            assert (len(ids)<1800 if group==0 else len(ids)>3000)
            pos=group*8+rank
            body=dict(input_ids=ids,stream=False,routed_dp_rank=rank,rid=f'graph-sweep-{nonce}-{pos}',
                      bootstrap_host=urlsplit(args.prefill).hostname,bootstrap_port=args.bootstrap,
                      bootstrap_room=(nonce<<20)+pos,sampling_params=dict(temperature=0,max_new_tokens=16))
            async def post(ep):
                async with client.post(ep+'/generate',json=body) as response:
                    response.raise_for_status();v=await response.json()
                assert (v['meta_info'].get('finish_reason') or {}).get('type')!='abort'
                return v
            pre,dec=await asyncio.gather(post(args.prefill),post(args.decode))
            assert pre['meta_info']['dp_rank']==dec['meta_info']['dp_rank']==rank
            answer=(dec.get('text') or '').rsplit('</think>',1)[-1].strip().strip(' .\n\t`')
            return dict(rank=rank,group=group,input_tokens=len(ids),input_sha256=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest(),
                        gold=label(entry),answer=answer,correct=answer==label(entry),output_ids=dec['output_ids'])
        for group, size in enumerate((64,360)):
            for ep in (args.prefill,args.decode):
                async with client.post(ep+'/flush_cache?timeout=20') as reply:reply.raise_for_status()
            rows.extend(await asyncio.gather(*(ask(rank,group,size) for rank in range(8))))
    result=dict(rows=rows,correct=sum(r['correct'] for r in rows),requests=len(rows),longest=max(r['input_tokens'] for r in rows))
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    print('Explicit DP rank graph sweep',result['correct'],result['requests'],result['longest'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prefill',required=True);p.add_argument('--decode',required=True)
    p.add_argument('--bootstrap',type=int,required=True);p.add_argument('--output',required=True)
    asyncio.run(run(p.parse_args()))
