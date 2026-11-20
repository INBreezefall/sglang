"""Score known answers on genuine host-restored and cold full-model prompts using identical token IDs."""
import argparse,asyncio,hashlib,json,secrets
from pathlib import Path
from urllib.parse import urlsplit
import aiohttp
from transformers import AutoTokenizer

async def main(a):
    tok=AutoTokenizer.from_pretrained('/model',local_files_only=True,trust_remote_code=True)
    rows=[];items=[];nonce=secrets.randbelow(1<<26)+1;counter=0
    for slot in range(8):
        name=('Aspen','Birch','Cedar','Elm','Fir','Larch','Maple','Oak')[slot]
        count=182+slot%3*11;index=(37+17*slot)%count
        answer=f'{(index*713+9327+slot*418)%100000:05d}'
        question=f'{name} log.\n'+'\n'.join(f'Record {j:03d} has value {(j*713+9327+slot*418)%100000:05d}.' for j in range(count))
        question+=f'\nCopy the five-digit value of record {index:03d}. Reply only with the five digits.'
        ids=tok.apply_chat_template([dict(role='user',content=question)],tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False)
        if isinstance(ids,dict):ids=ids['input_ids']
        if hasattr(ids,'tolist'):ids=ids.tolist()
        prefix=((len(ids)-60)//128)*128+1
        assert 1450<prefix<len(ids)
        items.append((slot,ids,prefix,answer))
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=230)) as conn:
        async def flush(ep):
            async with conn.post(ep+'/flush_cache?timeout=20') as r:r.raise_for_status()
        async def post(ep,body):
            async with conn.post(ep+'/generate',json=body) as r:
                r.raise_for_status();v=await r.json()
            m=v['meta_info'];assert (m.get('finish_reason') or {}).get('type')!='abort'
            return v
        async def ask(item,phase):
            nonlocal counter
            slot,full,prefix,answer=item
            seed=phase=='seed';ids=full[:prefix] if seed else full
            counter+=1
            body=dict(input_ids=ids,rid=f'cache-answer-{nonce}-{counter}',routed_dp_rank=0,
                      bootstrap_room=(nonce<<22)+counter,bootstrap_port=a.bootstrap,bootstrap_host=urlsplit(a.prefill).hostname,
                      stream=False,sampling_params=dict(temperature=0,max_new_tokens=1 if seed else 16,ignore_eos=seed))
            p,d=await asyncio.gather(post(a.prefill,body),post(a.decode,body))
            pm=p['meta_info'];dm=d['meta_info'];assert pm['dp_rank']==dm['dp_rank']==0
            text=(d.get('text') or '').rsplit('</think>',1)[-1].strip().strip(' .\n\t`')
            row=dict(slot=slot,phase=phase,length=len(ids),input_sha256=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest(),
                     expected=answer if not seed else None,correct=text==answer if not seed else None,output_ids=d.get('output_ids'),answer=text,
                     cache=pm.get('cached_tokens_details') or {},cached=pm.get('cached_tokens'))
            rows.append(row);return row
        await asyncio.gather(flush(a.prefill),flush(a.decode))
        for j in range(0,8,2):await asyncio.gather(*(ask(v,'seed') for v in items[j:j+2]))
        await asyncio.sleep(1);await flush(a.decode)
        hot=await asyncio.gather(*(ask(v,'restored') for v in items))
        await asyncio.gather(flush(a.prefill),flush(a.decode))
        cold=await asyncio.gather(*(ask(v,'cold') for v in items))
        assert [x['input_sha256'] for x in hot]==[x['input_sha256'] for x in cold]
        result=dict(rows=rows,restored_correct=sum(x['correct'] for x in hot),cold_correct=sum(x['correct'] for x in cold),
          host_hit_requests=sum(bool(x['cache'].get('host')) for x in hot),host_hit_tokens=sum(x['cache'].get('host') or 0 for x in hot),
          device_hit_tokens=sum(x['cache'].get('device') or 0 for x in hot),cold_cached=sum(x['cached'] or 0 for x in cold),
          exact_sequences=sum(x['output_ids']==y['output_ids'] for x,y in zip(hot,cold)))
        Path(a.output).write_text(json.dumps(result));print({k:v for k,v in result.items() if k!='rows'},flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prefill',required=True);p.add_argument('--decode',required=True);p.add_argument('--bootstrap',type=int,default=18784);p.add_argument('--output',required=True);asyncio.run(main(p.parse_args()))
