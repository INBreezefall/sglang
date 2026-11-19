"""Test whether a full generated 128-token page survives a real NEXTN decode completion."""

import asyncio
import hashlib
import json
import random
import secrets
import sys
import time
from urllib.parse import urlsplit
import aiohttp

prefill, decode, port = sys.argv[1:4]
nonce = secrets.randbelow(1 << 29) + 1


def digest(z):
    return hashlib.sha256(json.dumps(z, separators=(",", ":")).encode()).hexdigest()


def prompt(i):
    r = random.Random(948520 + i)
    return [r.randrange(512, 22000) for _ in range(1025)]


async def main():
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=255)
    ) as client:
        for ep in (prefill, decode):
            async with client.post(ep + "/flush_cache?timeout=20") as r:
                r.raise_for_status()

        async def ask(i, phase, ids, count):
            params = dict(temperature=0, ignore_eos=True, max_new_tokens=count)
            if phase == 0:
                params["logit_bias"] = {"1234": 100.0}
            body = dict(
                input_ids=ids,
                sampling_params=params,
                routed_dp_rank=0,
                stream=False,
                rid=f"retain-{nonce}-{phase}-{i}",
                bootstrap_host=urlsplit(prefill).hostname,
                bootstrap_port=int(port),
                bootstrap_room=(nonce << 23) + phase * 100 + i,
            )

            async def one(ep):
                async with client.post(ep + "/generate", json=body) as r:
                    r.raise_for_status()
                    value = await r.json()
                    if (value.get("meta_info", {}).get("finish_reason") or {}).get(
                        "type"
                    ) == "abort":
                        raise RuntimeError("request aborted")
                    return value

            start = time.monotonic()
            a, b = await asyncio.gather(*(one(ep) for ep in (prefill, decode)))
            pm = a["meta_info"]
            dm = b["meta_info"]
            out = b.get("output_ids")
            assert (
                isinstance(out, list)
                and len(out) == dm["completion_tokens"] == count
                and pm["dp_rank"] == dm["dp_rank"] == 0
            )
            return dict(
                case=("input-prefix" if i == 0 else "generated-tail"),
                phase=phase,
                input_sha256=digest(ids),
                generated_sha256=digest(out),
                input_tokens=len(ids),
                generated_tokens=len(out),
                fixed_token_count=out.count(1234),
                prefill_cached=pm.get("cached_tokens"),
                decode_cached=dm.get("cached_tokens"),
                elapsed_s=time.monotonic() - start,
            ), out

        seeds = await asyncio.gather(*(ask(i, 0, prompt(i), 128) for i in range(2)))
        follow_inputs = [prompt(0) + [345], prompt(1) + seeds[1][1] + [345]]
        nexts = await asyncio.gather(
            *(ask(i, 1, seq, 2) for i, seq in enumerate(follow_inputs))
        )
        print(
            json.dumps(
                dict(
                    scope="one generated page on two rank-pinned real sessions",
                    rows=[x[0] for x in [*seeds, *nexts]],
                )
            )
        )


asyncio.run(main())
