"""Drive device eviction and cold local-file restoration through two live PD services."""

import asyncio
import collections
import hashlib
import json
import os
import random
import re
import secrets
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit
import aiohttp

prefill, decode, bootstrap, folder = sys.argv[1:5]
folder = Path(folder)
nonce = secrets.randbelow(1 << 26) + 1
rows = []


def tokens(i):
    rng = random.Random(780012 + i)
    return [rng.randrange(512, 22000) for _ in range(2049)]


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def files():
    port = folder / "object-port"
    if port.exists():
        sys.path.insert(0, "/bench/object-runtime/python")
        import boto3
        from botocore.config import Config

        client = boto3.client(
            "s3",
            endpoint_url="http://127.0.0.1:" + port.read_text().strip(),
            region_name="us-east-1",
            aws_access_key_id="sglang-local-test",
            aws_secret_access_key=(folder / ".object-secret").read_text().strip(),
            config=Config(s3={"addressing_style": "path"}),
        )
        try:
            pages = client.get_paginator("list_objects_v2").paginate(
                Bucket="sglang-hicache"
            )
            return {
                obj["Key"]: (obj["Size"], obj["ETag"])
                for page in pages
                for obj in page.get("Contents", [])
                if obj["Key"] != "health/object"
            }
        finally:
            client.close()
    p = folder / "storage"
    return (
        {
            str(f.relative_to(p)): f.stat().st_mtime_ns
            for f in p.rglob("*")
            if f.is_file()
        }
        if p.exists()
        else {}
    )


async def main():
    phases = []
    endpoints = [prefill, decode]
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=100)
    ) as client:

        async def flush():
            async def one(ep):
                async with client.post(ep + "/flush_cache?timeout=20") as response:
                    response.raise_for_status()

            await asyncio.gather(*(one(ep) for ep in endpoints))

        async def ask(i, phase, order):
            seq = tokens(i)
            rid = f"cache-check-{nonce}-{order}-{i}"
            body = dict(
                input_ids=seq,
                routed_dp_rank=0,
                sampling_params=dict(max_new_tokens=2, ignore_eos=True, temperature=0),
                rid=rid,
                bootstrap_host=urlsplit(prefill).hostname,
                bootstrap_port=int(bootstrap),
                bootstrap_room=(nonce << 26) + (order << 10) + i,
                stream=False,
            )
            start = time.monotonic()

            async def one(ep):
                async with client.post(ep + "/generate", json=body) as r:
                    r.raise_for_status()
                    v = await r.json()
                    if (
                        v.get("error")
                        or (v.get("meta_info", {}).get("finish_reason") or {}).get(
                            "type"
                        )
                        == "abort"
                    ):
                        raise RuntimeError("generation aborted")
                    return v

            a, b = await asyncio.gather(*(one(ep) for ep in endpoints))
            pm = a["meta_info"]
            dm = b["meta_info"]
            assert (
                pm["dp_rank"] == 0
                and dm["dp_rank"] == 0
                and dm["completion_tokens"] == 2
            )
            return dict(
                phase=phase,
                prompt=i,
                input_hash=digest(seq),
                input_tokens=len(seq),
                prefill_dp_rank=pm["dp_rank"],
                decode_dp_rank=dm["dp_rank"],
                prefill_cached=pm.get("cached_tokens"),
                prefill_cache_details=pm.get("cached_tokens_details") or {},
                decode_cached=dm.get("cached_tokens"),
                output_hash=digest(b.get("output_ids"))
                if b.get("output_ids")
                else None,
                elapsed_s=time.monotonic() - start,
            )

        async def wave(indices, phase, order):
            rows.extend(await asyncio.gather(*(ask(i, phase, order) for i in indices)))

        await flush()
        initial = files()
        await wave([0, 1], "seed", 1)
        deadline = time.monotonic() + 14
        while time.monotonic() < deadline:
            now = files()
            changed = sum(initial.get(k) != v for k, v in now.items())
            if changed >= 16:
                break
            await asyncio.sleep(0.5)
        phases.append(dict(phase="seeded", files=len(now), new_or_modified=changed))
        for k in range(3):
            await wave([10 + k * 2, 11 + k * 2], "device-pressure", 2 + k)
        await wave([0, 1], "after-device-pressure", 5)
        await flush()
        await wave([0, 1], "after-cache-flush", 6)
        for k in range(3):
            await wave([32 + k * 2, 33 + k * 2], "second-device-pressure", 7 + k)
        await wave([0, 1], "after-refill-and-pressure", 10)
        phases.append(dict(phase="finished", files=len(files())))
    events = collections.Counter()
    for line in (folder / "server.log").open(errors="replace"):
        if (
            "HICACHE_PREFETCH_TIMING {" in line
            and "cache-check-" + str(nonce) + "-" in line
        ):
            try:
                data = json.loads(line.split("HICACHE_PREFETCH_TIMING ", 1)[1])
                events[data["event"]] += 1
            except (ValueError, KeyError):
                pass
    print(
        json.dumps(
            dict(
                method="rank-0 cache recovery after timed serving cohort",
                rows=rows,
                local_prefill_file_snapshots=phases,
                rank0_host_trace_events=events,
            )
        )
    )


asyncio.run(main())
