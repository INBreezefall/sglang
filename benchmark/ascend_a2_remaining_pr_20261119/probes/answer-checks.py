"""Four small, fixed answer checks sent to both PD roles after the timed replay."""

import concurrent.futures
import hashlib
import json
import re
import secrets
import sys
import time
import urllib.parse
import requests
from transformers import AutoTokenizer

prefill, decode, bootstrap = sys.argv[1:4]
tokenizer = AutoTokenizer.from_pretrained(
    "/model", trust_remote_code=True, local_files_only=True
)
cases = [
    ("addition", "Reply only with the number. What is 17 + 19?", "36"),
    ("multiplication", "Reply only with the number. What is 6 times 7?", "42"),
    (
        "copy",
        "Copy the following three digits exactly. Reply with just the digits. The digits are 815.",
        "815",
    ),
    ("subtraction", "Reply only with the number. What is 205 minus 17?", "188"),
]


def digest(obj):
    return hashlib.sha256(
        json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def send(ep, body):
    r = requests.post(ep + "/generate", json=body, timeout=(4, 65))
    r.raise_for_status()
    value = r.json()
    if (
        "error" in value
        or value.get("meta_info", {}).get("finish_reason", {}).get("type") == "abort"
    ):
        raise RuntimeError("generation failed")
    return value


def run(job):
    i, (name, question, expected) = job
    inputs = tokenizer.apply_chat_template(
        [dict(role="user", content=question)],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=False,
    )
    if isinstance(inputs, dict):
        inputs = inputs["input_ids"]
    if hasattr(inputs, "tolist"):
        inputs = inputs.tolist()
    assert isinstance(inputs, list) and isinstance(inputs[0], int)
    nonce = secrets.randbelow(1 << 43) + 1
    body = dict(
        input_ids=inputs,
        sampling_params=dict(temperature=0, max_new_tokens=24),
        rid="quality-" + str(nonce),
        bootstrap_host=urllib.parse.urlsplit(prefill).hostname,
        bootstrap_port=int(bootstrap),
        bootstrap_room=nonce,
        stream=False,
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        fs = [pool.submit(send, ep, body) for ep in (prefill, decode)]
        p, d = [f.result() for f in fs]
    raw = d.get("text", "")
    answer = raw.rsplit("</think>", 1)[-1].strip().strip(" `\n\t.")
    return dict(
        task=name,
        prompt=question,
        expected=expected,
        reply=raw,
        normalized=answer,
        exact=answer == expected,
        input_hash=digest(inputs),
        output_ids_sha256=digest(d.get("output_ids", [])),
        tokens=d["meta_info"]["completion_tokens"],
    )


with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    rows = list(pool.map(run, enumerate(cases)))
print(
    json.dumps(
        dict(
            scope="Four short deterministic-answer checks after formal measurement",
            rows=rows,
        )
    )
)
