"""Measure which streamed completions are represented in the real Decode ITL histogram."""
import argparse
import asyncio
import json
import secrets
from pathlib import Path
from urllib.parse import urlsplit
import aiohttp

METRICS={'itl_count':'sglang:inter_token_latency_seconds_count','itl_sum':'sglang:inter_token_latency_seconds_sum',
         'ttft_count':'sglang:time_to_first_token_seconds_count','generation':'sglang:generation_tokens_total',
         'requests':'sglang:num_requests_total'}

async def run(a):
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as client:
        async def metrics():
            async with client.get(a.decode+'/metrics',headers={'Accept-Encoding':'identity'}) as r:
                r.raise_for_status();lines=(await r.text()).splitlines()
            return {k:sum(float(line.rsplit(' ',1)[1]) for line in lines if line.startswith((v+'{',v+' '))) for k,v in METRICS.items()}
        samples=[]
        for length in (2,3,5,8,24,64):
            room=secrets.randbits(55)
            body=dict(input_ids=[1223,1544,2661,4781,1911,4332,5699,6311],session_id='counter-calibration-'+str(room),rid='counter-'+str(room),
                bootstrap_host=urlsplit(a.prefill).hostname,bootstrap_port=18784,bootstrap_room=room,
                sampling_params=dict(temperature=0,max_new_tokens=length,ignore_eos=True))
            async def prefill():
                async with client.post(a.prefill+'/generate',json=dict(body,stream=False)) as r:
                    r.raise_for_status();obj=await r.json();assert 'error' not in obj
            async def decode():
                counts=[]
                async with client.post(a.decode+'/generate',json=dict(body,stream=True)) as r:
                    r.raise_for_status()
                    async for line in r.content:
                        if not line.startswith(b'data:'):continue
                        event=line[5:].strip()
                        if event==b'[DONE]':break
                        if event:
                            obj=json.loads(event);assert 'error' not in obj
                            counts.append(int(obj['meta_info']['completion_tokens']))
                return counts
            before=await metrics()
            _,counts=await asyncio.gather(prefill(),decode())
            after=await metrics()
            for _ in range(12):
                if after['generation']-before['generation']>=length and after['requests']>before['requests']:
                    break
                await asyncio.sleep(.25)
                after=await metrics()
            assert counts and counts[-1]==length and counts==sorted(counts)
            samples.append(dict(output_limit=length,streamed_cumulative_counts=counts,
                                counters={k:after[k]-before[k] for k in before}))
    Path(a.output).write_text(json.dumps(dict(samples=samples),indent=2)+'\n')
    print([(s['output_limit'],s['streamed_cumulative_counts'][0],s['counters']['itl_count']) for s in samples],flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('prefill','decode','output'):parser.add_argument('--'+name,required=True)
    asyncio.run(run(parser.parse_args()))
