"""Fixed natural-language answers and fixed-context next-token scores for single or PD serving."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import secrets
import time
from urllib.parse import urlsplit
import requests
from transformers import AutoTokenizer

def digest(x):
    return hashlib.sha256(json.dumps(x,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
def cases():
    for i in range(8):
        a=17+i*4;b=19+i*7
        yield f'add-{i}',f'Reply only with the integer. What is {a} + {b}?',str(a+b)
        a=82+i*8;b=7+i
        yield f'sub-{i}',f'Reply only with the integer. What is {a} minus {b}?',str(a-b)
    for i in range(4):
        yield f'copy-{i}',f'Repeat exactly this identifier, without explanation or punctuation: Atlas-{4871+i*17}',f'Atlas-{4871+i*17}'
        a=6+i;b=7+i
        yield f'mul-{i}',f'Answer with the integer alone. What is {a} times {b}?',str(a*b)
    for i in range(8):
        # Use both short and sparse-indexer-length contexts, with the target at fixed but varied positions.
        count=(32,96,260,460)[i%4]; key=(count*(3+i%5)//10)
        records=[f'Record {j:03d} has value {(j*713+9327+i*418)%100000:05d}.' for j in range(count)]
        answer=f'{(key*713+9327+i*418)%100000:05d}'
        yield f'lookup-{i}','\n'.join(records)+f'\nCopy the five-digit value of record {key:03d}. Reply only with the five digits.',answer

def main():
    p=argparse.ArgumentParser();p.add_argument('--prefill',required=True);p.add_argument('--decode');p.add_argument('--bootstrap',type=int,default=18784);p.add_argument('--output',required=True);p.add_argument('--short',action='store_true');p.add_argument('--repeats',type=int,default=1);p.add_argument('--rank',type=int,default=0)
    a=p.parse_args();tok=AutoTokenizer.from_pretrained('/model',trust_remote_code=True,local_files_only=True)
    nonce=secrets.randbelow(1<<24)+1
    def post(ep,body):
        r=requests.post(ep+'/generate',json=body,timeout=(4,170));r.raise_for_status();value=r.json()
        m=value['meta_info']
        if value.get('error') or (m.get('finish_reason') or {}).get('type')=='abort':raise RuntimeError(str(m.get('finish_reason')))
        return value
    def ask(index,ids,max_output):
        body=dict(input_ids=ids,stream=False,sampling_params=dict(temperature=0,max_new_tokens=max_output),
                  return_logprob=True,top_logprobs_num=5,logprob_start_len=-1)
        if a.decode:
            body.update(rid=f'quality-{nonce}-{index}',routed_dp_rank=a.rank,bootstrap_host=urlsplit(a.prefill).hostname,
                        bootstrap_port=a.bootstrap,bootstrap_room=(nonce<<24)+index)
            with ThreadPoolExecutor(max_workers=2) as ex:
                pf,df=[ex.submit(post,ep,body) for ep in (a.prefill,a.decode)];pf.result();d=df.result()
        else:d=post(a.prefill,body)
        m=d['meta_info']
        return dict(ids=d.get('output_ids'),text=d.get('text'),chosen=m.get('output_token_logprobs'),top=m.get('output_top_logprobs'))
    records=[];q=list(cases())
    if a.short:q=[row for row in q if not row[0].startswith('lookup-') or int(row[0].split('-')[-1])<2]
    for i,(name,question,answer) in enumerate(q):
        ids=tok.apply_chat_template([dict(role='user',content=question)],tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False)
        if isinstance(ids,dict):ids=ids['input_ids']
        if hasattr(ids,'tolist'):ids=ids.tolist()
        value=ask(i,ids,16);norm=(value['text'] or '').rsplit('</think>',1)[-1].strip().strip(' `\n\t.')
        records.append(dict(name=name,expected=answer,correct=norm==answer,token_count=len(ids),input_hash=digest(ids),**value))
    free=[]
    prefix=tok.encode('2 + 2 =',add_special_tokens=True)
    for i in range(3*a.repeats):
        val=ask(1000+i,prefix,12)
        free.append(dict(prefix_hash=digest(prefix),**val))
    fixed=[]
    # Same pretokenized context in every service, independent of earlier free generation.
    for i,text in enumerate(('2 + 2 =','2 + 2 = 4','The capital city of Japan is','Carbon dioxide contains','One two three four five','Water freezes at','The seven days of the','Compute 17 plus 19. The result is')):
        ids=tok.encode(text,add_special_tokens=True)
        fixed.append(dict(context=text,input_hash=digest(ids),**ask(2000+i,ids,1)))
    dest=Path(a.output);dest.write_text(json.dumps(dict(cases=records,free=free,fixed=fixed,started_nonce=nonce),indent=2)+'\n')
    print(json.dumps(dict(count=len(records),correct=sum(r['correct'] for r in records),free=len(free),fixed=len(fixed))),flush=True)
if __name__=='__main__':main()
