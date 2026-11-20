"""Repeat the original eight-question ARC cohort containing the one changed TP8 choice."""
import argparse,asyncio,hashlib,json,math,secrets
from pathlib import Path
from urllib.parse import urlsplit
import aiohttp
from transformers import AutoTokenizer

async def main(a):
    raw=Path('arc-256.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest()=='aab26bca3e6f5623d3ad55fc4e1e8b093f9920b9bff92db8e97193960aa6d856'
    data=json.loads(raw);ix=next(i for i,r in enumerate(data) if r['id']==a.focus_id)
    start=(ix//8)*8
    tok=AutoTokenizer.from_pretrained('/model',local_files_only=True,trust_remote_code=True)
    for space in (' ',''):
        opts=[tok.encode(space+c,add_special_tokens=False) for c in 'ABCD']
        if all(len(q)==1 for q in opts) and len({q[0] for q in opts})==4:break
    else:raise RuntimeError('Invalid option tokens')
    option_ids=[q[0] for q in opts];nonce=secrets.randbelow(1<<26)+1;rows=[]
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=180)) as conn:
        async def post(url,body):
            async with conn.post(url+'/generate',json=body) as res:
                res.raise_for_status();v=await res.json()
                assert (v['meta_info'].get('finish_reason') or {}).get('type')!='abort'
                return v
        async def ask(i,rep):
            r=data[i];s=r['question']+'\n'+'\n'.join(f'{l}. {choice}' for l,choice in zip('ABCD',r['choices']['text']))+'\nAnswer with the letter of the correct option.'
            prompt=tok.apply_chat_template([dict(role='user',content=s)],add_generation_prompt=True,tokenize=False,enable_thinking=False)+'The answer is'
            ids=tok.encode(prompt,add_special_tokens=False)
            body=dict(input_ids=ids,stream=False,return_logprob=True,logprob_start_len=-1,token_ids_logprob=option_ids,sampling_params=dict(temperature=0,max_new_tokens=2,ignore_eos=True))
            if a.decode:
                body.update(rid=f'arc-repeat-{nonce}-{rep}-{i}',routed_dp_rank=i%a.ranks,bootstrap_host=urlsplit(a.prefill).hostname,bootstrap_port=a.bootstrap,bootstrap_room=(nonce<<22)+rep*1024+i)
                vals=await asyncio.gather(post(a.prefill,body),post(a.decode,body))
            else:vals=[await post(a.prefill,body)]
            for role,v in enumerate(vals):
                scores=v['meta_info'].get('output_token_ids_logprobs') or []
                if not scores or not scores[0]:continue
                lookup={int(z[1]):z[0] for z in scores[0] if z and z[0] is not None}
                if set(option_ids)<=lookup.keys():break
            else:raise RuntimeError('Option log probabilities missing')
            logps=[float(lookup[t]) for t in option_ids];peak=max(logps);p=[math.exp(t-peak) for t in logps];norm=sum(p)
            return dict(id=r['id'],rep=rep,input_sha256=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest(),gold=r['answerKey'],
                choice='ABCD'[logps.index(peak)],option_probs=[v/norm for v in p],option_logprobs=logps,
                score_source=('prefill' if role==0 else 'decode') if a.decode else 'single',top_logprob_tied=logps.count(peak)>1)
        for rep in range(6):
            for ep in filter(None,(a.prefill,a.decode)):
                async with conn.post(ep+'/flush_cache?timeout=20') as res:res.raise_for_status()
            rows.extend(await asyncio.gather(*(ask(i,rep) for i in range(start,start+8))))
            Path(a.output).write_text(json.dumps(dict(cohort_offset=start,rows=rows,focus_id=data[ix]['id'])))
    print(len(rows),'repeated scored requests')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prefill',required=True);p.add_argument('--decode');p.add_argument('--output',required=True);p.add_argument('--bootstrap',type=int,default=18784);p.add_argument('--ranks',type=int,default=8);p.add_argument('--focus-id',default='Mercury_SC_400134');asyncio.run(main(p.parse_args()))
