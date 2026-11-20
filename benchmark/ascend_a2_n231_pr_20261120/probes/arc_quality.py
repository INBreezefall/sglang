"""Score the same 256 held-out ARC-Challenge questions by four option-token log probabilities."""
import argparse
import asyncio
import hashlib
import json
import math
from pathlib import Path
import secrets
from urllib.parse import urlsplit

import aiohttp
from transformers import AutoTokenizer

async def main(a):
    data=Path(a.data).read_bytes()
    sha=hashlib.sha256(data).hexdigest()
    assert sha=='aab26bca3e6f5623d3ad55fc4e1e8b093f9920b9bff92db8e97193960aa6d856'
    cases=json.loads(data);tok=AutoTokenizer.from_pretrained('/model',local_files_only=True,trust_remote_code=True)
    letters='ABCD'
    for space in (' ',''):
        opts=[tok.encode(space+letter,add_special_tokens=False) for letter in letters]
        if all(len(v)==1 for v in opts) and len(set(v[0] for v in opts))==4:break
    else:raise ValueError('Expected four distinct single-token answer choices')
    ids=[v[0] for v in opts];nonce=secrets.randbelow(1<<27)+1;rows=[]
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=180)) as session:
        async def post(ep,b):
            async with session.post(ep+'/generate',json=b) as response:
                raw=await response.text()
                if response.status!=200:raise RuntimeError(f'HTTP {response.status}: '+raw[-220:])
                value=json.loads(raw)
                assert (value['meta_info'].get('finish_reason') or {}).get('type')!='abort'
                return value
        async def one(i,r):
            user=r['question']+'\n'+'\n'.join(f'{l}. {x}' for l,x in zip(letters,r['choices']['text']))+'\nAnswer with the letter of the correct option.'
            prompt=tok.apply_chat_template([dict(role='user',content=user)],add_generation_prompt=True,tokenize=False,enable_thinking=False)+'The answer is'
            seq=tok.encode(prompt,add_special_tokens=False)
            body=dict(input_ids=seq,stream=False,return_logprob=True,logprob_start_len=-1,token_ids_logprob=ids,
                      sampling_params=dict(temperature=0,max_new_tokens=2,ignore_eos=True))
            if a.decode:
                body.update(rid=f'arc-{nonce}-{i}',routed_dp_rank=i%a.ranks,bootstrap_host=urlsplit(a.prefill).hostname,
                            bootstrap_port=a.bootstrap,bootstrap_room=(nonce<<22)+i)
                pre,dec=await asyncio.gather(post(a.prefill,body),post(a.decode,body))
            else:pre=dec=await post(a.prefill,body)
            def scores(result):
                z=result['meta_info'].get('output_token_ids_logprobs') or []
                if not z or not z[0]:return None
                vals={int(v[1]):v[0] for v in z[0] if v and v[0] is not None}
                if set(ids)-vals.keys():return None
                return [float(vals[x]) for x in ids]
            # The first generated token is owned by prefill in a PD deployment.
            vals=scores(pre)
            from_role='prefill' if a.decode else 'single'
            if vals is None:
                vals=scores(dec);from_role='decode' if a.decode else 'single'
            if vals is None:raise ValueError('No first-position log probabilities for the four requested option IDs')
            assert all(math.isfinite(x) for x in vals), (i,vals)
            maximum=max(vals);den=sum(math.exp(x-maximum) for x in vals)
            pred=letters[vals.index(maximum)]
            return dict(id=r['id'],gold=r['answerKey'],prediction=pred,correct=pred==r['answerKey'],score_source=from_role,
                        input_sha256=hashlib.sha256(json.dumps(seq,separators=(',',':')).encode()).hexdigest(),
                        input_tokens=len(seq),option_logprobs=vals,option_probs=[math.exp(x-maximum)/den for x in vals])
        for offset in range(0,len(cases),8):
            batch=await asyncio.gather(*(one(i,cases[i]) for i in range(offset,min(offset+8,len(cases)))))
            rows.extend(batch)
            Path(a.output+'.partial').write_text(json.dumps(rows))
    assert len(rows)==256
    Path(a.output).write_text(json.dumps(dict(dataset_sha256=sha,option_token_ids=ids,option_space_prefix=space,
        items=len(rows),correct=sum(r['correct'] for r in rows),rows=rows),indent=2)+'\n')
    print('ARC-Challenge',sum(r['correct'] for r in rows),'/',len(rows),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prefill',required=True);p.add_argument('--decode');p.add_argument('--bootstrap',type=int,default=18784);p.add_argument('--data',default='arc-256.json');p.add_argument('--output',required=True);p.add_argument('--ranks',type=int,default=8);asyncio.run(main(p.parse_args()))
