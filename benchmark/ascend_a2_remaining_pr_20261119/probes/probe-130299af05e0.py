"""Build a network-isolated Docker context with each revision's actual ignore rules."""

import hashlib, json, os, shutil, subprocess, sys, tarfile, tempfile, time
from pathlib import Path

root = Path(os.environ["BENCH_ROOT"])
job = root / "jobs/docker-context"
job.mkdir(exist_ok=True)
assert (root / "owner").read_text().strip() == os.environ["BENCH_OWNER"]
rows = []
paths = {
    "application.py": 37,
    "pkg/__pycache__/a.cpython-312.pyc": 8 * 1024**2,
    "pkg/mod.pyc": 2 * 1024**2,
    "deep/tools/x.pyo": 1024**2,
    "pkg/resource.bin": 8192,
    "deep/notcache/keep.py": 125,
}


def docker(*args, **extra):
    return subprocess.run(
        ["docker", *map(str, args)],
        capture_output=True,
        check=True,
        env={**os.environ, "DOCKER_BUILDKIT": "0"},
        **extra,
    )


with tempfile.TemporaryDirectory(dir=job) as tmp:
    context = Path(tmp)
    for name, size in paths.items():
        f = context / name
        f.parent.mkdir(parents=True, exist_ok=True)
        block = hashlib.sha256(name.encode()).digest() * 32768
        with f.open("wb") as h:
            while size:
                n = min(size, len(block))
                h.write(block[:n])
                size -= n
    (context / "Dockerfile").write_text("FROM scratch\nCOPY . /fixture/\n")
    for i, arm in enumerate(("main", "49181", "49181", "main")):
        src = root / "src" / arm
        ignore = src / ".dockerignore"
        g = src / ".gitignore"
        shutil.copyfile(g, context / ".gitignore")
        f = context / ".dockerignore"
        if f.is_symlink() or f.exists():
            f.unlink()
        if ignore.is_symlink():
            f.symlink_to(os.readlink(ignore))
        else:
            shutil.copyfile(ignore, f)
        tag = (
            "sglang-context-check:"
            + hashlib.sha256(str(context).encode()).hexdigest()[:16]
            + "-"
            + str(i)
        )
        try:
            t = time.perf_counter_ns()
            docker(
                "build",
                "--network=none",
                "--no-cache",
                "--pull=false",
                "--label",
                "sglang-context-check=bytecode",
                "-t",
                tag,
                str(context),
            )
            ms = (time.perf_counter_ns() - t) / 1e6
            with tempfile.NamedTemporaryFile(dir=job) as export:
                docker("save", "-o", export.name, tag)
                with tarfile.open(export.name) as outer:
                    manifest = json.load(outer.extractfile("manifest.json"))
                    layer = manifest[0]["Layers"][-1]
                    with tarfile.open(
                        fileobj=outer.extractfile(layer), mode="r|*"
                    ) as inner:
                        included = {
                            m.name.removeprefix("fixture/"): m.size
                            for m in inner
                            if m.isfile() and m.name.startswith("fixture/")
                        }
            rows.append(
                dict(
                    flag=int(arm != "main"),
                    sha=(src / "SHA").read_text().strip(),
                    ms=ms,
                    image_bytes=int(
                        docker("image", "inspect", "-f", "{{.Size}}", tag).stdout
                    ),
                    included=included,
                    nested_bytecode_bytes=sum(
                        v for k, v in included.items() if k.endswith((".pyc", ".pyo"))
                    ),
                )
            )
        finally:
            state = subprocess.run(
                [
                    "docker",
                    "image",
                    "inspect",
                    "-f",
                    '{{index .Config.Labels "sglang-context-check"}}',
                    tag,
                ],
                capture_output=True,
                text=True,
            )
            if state.returncode == 0 and state.stdout.strip() == "bytecode":
                docker("image", "rm", tag)
assert all(
    r["included"].get("application.py") == 37
    and r["included"].get("pkg/resource.bin") == 8192
    for r in rows
)
print(
    json.dumps(
        dict(
            feature=49181,
            driver="legacy Docker builder; networking disabled",
            rows=rows,
        )
    )
)
