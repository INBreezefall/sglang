"""Exercise real target/draft model prefill and HiCache copies with SGLang's prefill-only sender."""
import argparse,asyncio,hashlib,json,random,secrets,time
from statistics import mean
from pathlib import Path
import aiohttp
from transformers import AutoTokenizer

async def main(a):
    rows=[];waves=[];nonce=secrets.randbelow(1<<25)+1;counter=0
    tokenizer=AutoTokenizer.from_pretrained('/model',local_files_only=True,trust_remote_code=True)
    letters='ABCD'
    for space in (' ',''):
        choice_ids=[tokenizer.encode(space+letter,add_special_tokens=False) for letter in letters]
        if all(len(v)==1 for v in choice_ids) and len({v[0] for v in choice_ids})==4:break
    else:raise ValueError('Four individual option tokens required')
    choice_ids=[v[0] for v in choice_ids]
    def tokens(seed,length):
        rnd=random.Random(seed);return [rnd.randrange(512,22000) for _ in range(length)]
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=210)) as conn:
        async def flush():
            async with conn.post(a.prefill+'/flush_cache?timeout=20') as reply:reply.raise_for_status()
        async def ask(ids,rep,phase,slot,gold=None):
            nonlocal counter
            counter+=1
            # The SGLang native warmup and health path uses this special bootstrap sentinel.
            msg=dict(input_ids=ids,rid=f'prefill-only-{nonce}-{counter}',routed_dp_rank=0,bootstrap_host='2.2.2.2',bootstrap_room=(nonce<<22)+counter,
                sampling_params=dict(temperature=0,max_new_tokens=1,ignore_eos=True),return_logprob=True,top_logprobs_num=5,logprob_start_len=-1)
            if gold is not None:msg['token_ids_logprob']=choice_ids
            begin=time.monotonic()
            async with conn.post(a.prefill+'/generate',json=msg) as reply:
                reply.raise_for_status();v=await reply.json()
            m=v['meta_info'];assert (m.get('finish_reason') or {}).get('type')!='abort' and m.get('dp_rank')==0
            ids_out=v.get('output_ids') or [];chosen=m.get('output_token_logprobs') or []
            if not ids_out and chosen:ids_out=[int(chosen[0][1])]
            assert len(ids_out)==1, (len(ids_out),m.get('completion_tokens'))
            item=dict(input_sha256=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest(),length=len(ids),rep=rep,phase=phase,slot=slot,
                first_token=int(ids_out[0]),completion_s=time.monotonic()-begin,cached=m.get('cached_tokens'),cache=m.get('cached_tokens_details') or {},top=m.get('output_top_logprobs'))
            if gold is not None:
                scores={int(r[1]):float(r[0]) for r in m['output_token_ids_logprobs'][0] if r[0] is not None}
                values=[scores[k] for k in choice_ids]
                choice=letters[values.index(max(values))]
                item.update(gold=gold,choice=choice,correct=choice==gold,option_logprobs=values)
            rows.append(item);return item
        for rep in range(2):
            await flush();seed=873903+rep*32
            for wave in range(4):
                await asyncio.gather(*(ask(tokens(seed+i,2049),rep,'seed',i) for i in (2*wave,2*wave+1)))
            await asyncio.sleep(1)
            order=list(range(8));random.Random(seed).shuffle(order)
            begin=time.monotonic()
            mixed=await asyncio.gather(*(ask(tokens(seed+i,2049)+tokens(seed+i+12,48+i*7),rep,'restore',i) for i in order))
            wave_s=time.monotonic()-begin
            await flush()
            cold=await asyncio.gather(*(ask(tokens(seed+i,2049)+tokens(seed+i+12,48+i*7),rep,'cold',i) for i in order))
            assert [r['input_sha256'] for r in mixed]==[r['input_sha256'] for r in cold]
            waves.append(dict(host_hit_tokens=sum(v['cache'].get('host') or 0 for v in mixed),device_hit_tokens=sum(v['cache'].get('device') or 0 for v in mixed),
                cold_cached=sum(v['cached'] or 0 for v in cold),first_token_matches=sum(x['first_token']==y['first_token'] for x,y in zip(mixed,cold)),
                mean_prefill_response_s=mean(x['completion_s'] for x in mixed),wave_s=wave_s))
            Path(a.output).write_text(json.dumps(dict(waves=waves,rows=rows)))
        prompts=[]
        for slot,name in enumerate(('Amber','Bronze','Cobalt','Denim','Emerald','Flint','Gold','Hazel')):
            count=210+slot%3*8
            codes=[letters[(j*7+j//9+slot)%4] for j in range(count)]
            index=34+slot*17
            q=name+' reference file.\n'+'\n'.join(f'Entry {j:03d} has code {c}.' for j,c in enumerate(codes))
            q+=f'\nWhich single letter is recorded for entry {index:03d}? Choose A, B, C, or D.'
            t=tokenizer.apply_chat_template([dict(role='user',content=q)],tokenize=False,add_generation_prompt=True,enable_thinking=False)+'The recorded letter is'
            seq=tokenizer.encode(t,add_special_tokens=False)
            prefix=((len(seq)-60)//128)*128+1
            assert 1400<prefix<len(seq)
            prompts.append((slot,seq,prefix,codes[index]))
        await flush()
        for start in range(0,8,2):
            await asyncio.gather(*(ask(seq[:prefix],2,'known-seed',slot) for slot,seq,prefix,_ in prompts[start:start+2]))
        await asyncio.sleep(1)
        hot=await asyncio.gather(*(ask(seq,2,'known-restored',slot,gold) for slot,seq,_,gold in prompts))
        await flush()
        cold=await asyncio.gather(*(ask(seq,2,'known-cold',slot,gold) for slot,seq,_,gold in prompts))
        assert [r['input_sha256'] for r in hot]==[r['input_sha256'] for r in cold]
        known=dict(restored_correct=sum(r['correct'] for r in hot),cold_correct=sum(r['correct'] for r in cold),
                   host_hit_requests=sum(bool(r['cache'].get('host')) for r in hot),
                   host_hit_tokens=sum(r['cache'].get('host') or 0 for r in hot),cold_cached=sum(r['cached'] or 0 for r in cold),
                   choices_agree=sum(x['choice']==y['choice'] for x,y in zip(hot,cold)),option_tokens=choice_ids)
        Path(a.output).write_text(json.dumps(dict(waves=waves,rows=rows,known=known)))
        await flush()
    print('Prefill only',waves,known)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prefill',required=True);p.add_argument('--decode');p.add_argument('--bootstrap',type=int);p.add_argument('--output',required=True);asyncio.run(main(p.parse_args()))
