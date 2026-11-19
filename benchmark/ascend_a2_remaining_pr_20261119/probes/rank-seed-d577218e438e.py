"""Prime each NEXTN decode rank with one matched-length request before timing."""

import asyncio
import hashlib
import json
import random
import secrets
import sys
from urllib.parse import urlsplit
import aiohttp

prefill, decode, port = sys.argv[1:4]
nonce = secrets.randbelow(1 << 28) + 1


def digest(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


async def main():
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=105)
    ) as client:

        async def prime(rank):
            rng = random.Random(681500 + rank)
            ids = [rng.randrange(512, 22000) for _ in range(512)]
            body = dict(
                input_ids=ids,
                routed_dp_rank=rank,
                stream=False,
                sampling_params=dict(temperature=0, max_new_tokens=4, ignore_eos=True),
                rid=f"prime-{nonce}-{rank}",
                bootstrap_host=urlsplit(prefill).hostname,
                bootstrap_port=int(port),
                bootstrap_room=(nonce << 14) + rank,
            )

            async def send(ep):
                async with client.post(ep + "/generate", json=body) as response:
                    response.raise_for_status()
                    return await response.json()

            p, d = await asyncio.gather(send(prefill), send(decode))
            assert p["meta_info"]["dp_rank"] == d["meta_info"]["dp_rank"] == rank
            assert len(d["output_ids"]) == d["meta_info"]["completion_tokens"] == 4
            return dict(
                rank=rank,
                input_sha256=digest(ids),
                output_sha256=digest(d["output_ids"]),
            )

        rows = await asyncio.gather(*(prime(rank) for rank in range(16)))
        print(json.dumps(dict(rows=rows)))


asyncio.run(main())
