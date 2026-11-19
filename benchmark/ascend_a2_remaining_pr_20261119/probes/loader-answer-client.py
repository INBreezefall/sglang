"""Tokenize four explicit chat prompts and check a live, single-node TP8 model."""

import concurrent.futures
import hashlib
import json
import time
import urllib.request
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(
    "/model", trust_remote_code=True, local_files_only=True
)
prompts = [
    ("addition", "Reply only with the number. What is 17 + 19?", "36"),
    ("multiplication", "Reply only with the number. What is 6 times 7?", "42"),
    (
        "copy",
        "Copy the following three digits exactly. Reply with just the digits. The digits are 815.",
        "815",
    ),
    ("subtraction", "Reply only with the number. What is 205 minus 17?", "188"),
]


def digest(z):
    return hashlib.sha256(json.dumps(z, separators=(",", ":")).encode()).hexdigest()


def ask(x):
    name, question, target = x
    seq = tok.apply_chat_template(
        [dict(role="user", content=question)],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=False,
    )
    if hasattr(seq, "tolist"):
        seq = seq.tolist()
    if isinstance(seq, dict):
        seq = seq["input_ids"]
    req = urllib.request.Request(
        "http://127.0.0.1:19272/generate",
        data=json.dumps(
            dict(input_ids=seq, sampling_params=dict(temperature=0, max_new_tokens=16))
        ).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=90) as response:
        result = json.load(response)
    answer = result["text"].rsplit("</think>", 1)[-1].strip().strip(" `\n\t.")
    return dict(
        task=name,
        expected=target,
        answer=answer,
        exact=answer == target,
        input_sha256=digest(seq),
        tokens=result["meta_info"]["completion_tokens"],
        token_ids=result.get("output_ids"),
    )


deadline = time.monotonic() + 720
while time.monotonic() < deadline:
    try:
        with urllib.request.urlopen(
            "http://127.0.0.1:19272/model_info", timeout=2
        ) as r:
            if r.status == 200:
                break
    except OSError:
        time.sleep(1)
else:
    raise TimeoutError("service readiness")
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    rows = list(pool.map(ask, prompts))
print(json.dumps(rows))
