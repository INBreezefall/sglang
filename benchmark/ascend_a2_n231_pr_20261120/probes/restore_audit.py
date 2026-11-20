"""Compare samples of real restored device rows with the owning host pages after transfer."""
from collections import Counter
import functools
import hashlib
import json
import os
from pathlib import Path


def install(module, folder):
    cls = module.MLATokenToKVPoolHost
    transfer = cls.load_to_device_per_layer
    checked = Counter()
    marker = Path(folder) / "check-restores"
    output = Path(folder) / f"forward-rank-copy-{os.getpid()}.jsonl"

    @functools.wraps(transfer)
    def load(self, device_pool, host_indices, device_indices, layer_id, io_backend, *, is_draft=False):
        result = transfer(self, device_pool, host_indices, device_indices, layer_id, io_backend, is_draft=is_draft)
        if io_backend != "kernel_ascend" or self.layout != "page_first_kv_split" or not marker.is_file():
            return result
        binding = self._npu_layer_bindings.get(layer_id)
        if binding is None or not host_indices.numel() or not device_indices.numel():
            return result
        packed_draft = is_draft or layer_id >= self._device_layer_num
        key = (self.pool_label, packed_draft, binding.get("device_index_k") is not None)
        if checked[key] >= 4:
            return result
        checked[key] += 1
        import torch

        row = dict(event="restore-check", label=str(self.pool_label), packed_draft=bool(packed_draft),
                   target_layers=int(self.target_layer_num), host_tokens=int(self.size),
                   layer=int(layer_id), transfer_tokens=int(device_indices.numel()), buffers=[])
        try:
            torch.npu.current_stream().synchronize()
            positions = sorted({0, min(13, device_indices.numel()-1), device_indices.numel()-1})
            for param in ("k", "v", "index_k", "index_k_scale"):
                device, host = binding.get("device_"+param), binding.get("host_"+param)
                if device is None or host is None or not device.numel() or not host.numel():
                    continue
                layer_key = "index_layer_id" if param.startswith("index") else "layer_id"
                dl, hl = binding.get("device_"+layer_key), binding.get("host_"+layer_key)
                if dl is None or hl is None:
                    continue
                page = int(binding["page_size"])
                for pos in positions:
                    di, hi = int(device_indices[pos].item()), int(host_indices[pos].item())
                    a = device[dl, di//page, di%page].detach().cpu().contiguous().view(torch.uint8)
                    b = host[hi//page, hl, hi%page].detach().cpu().contiguous().view(torch.uint8)
                    row["buffers"].append(dict(name=param, position=pos, bytes=a.numel(),
                           device_sha256=hashlib.sha256(a.numpy().tobytes()).hexdigest(),
                           host_sha256=hashlib.sha256(b.numpy().tobytes()).hexdigest(),
                           equal=bool(torch.equal(a,b))))
        except (RuntimeError, ValueError, IndexError, TypeError) as exc:
            row["audit_error"] = type(exc).__name__ + ": " + str(exc).splitlines()[0][:200]
        with output.open("a") as fp:
            print(json.dumps(row),file=fp)
        return result

    cls.load_to_device_per_layer = load
