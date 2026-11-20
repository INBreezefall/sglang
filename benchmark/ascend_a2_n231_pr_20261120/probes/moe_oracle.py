"""Compare tiled and untiled experts on the same live GLM activations, routes and weights."""
from collections import Counter
import functools
import json
import os
from pathlib import Path

import base_observe

base_install = base_observe.install
seen = Counter()


def install(module):
    base_install(module)
    if module.__name__ != base_observe.tile_module or os.environ.get('GLM_PR_ROLE') != 'prefill':
        return
    import torch
    from sglang.srt.runtime_context import get_parallel

    original = module.run_prefill_experts
    folder = Path(os.environ['GLM_PR_POOL_PROBE'])

    def duplicate(routes):
        return routes._replace(**{name: value.detach().clone() for name, value in zip(routes._fields, routes)
                                  if isinstance(value, torch.Tensor)})

    def difference(actual, reference):
        actual, reference = actual.float(), reference.float()
        error = actual - reference
        return dict(equal_values=int(torch.eq(actual, reference).sum().item()), values=actual.numel(),
                    finite=bool(torch.isfinite(actual).all() and torch.isfinite(reference).all()),
                    max_abs=error.abs().max().item(),
                    relative_l2=(torch.linalg.vector_norm(error) / torch.linalg.vector_norm(reference).clamp_min(1e-12)).item())

    @functools.wraps(original)
    def measured(layer, inputs, routes):
        layer_id = layer.layer_id
        if (layer_id not in (3, 32, 64) or seen[layer_id] >= 2 or inputs.shape[0] <= 256
                or not (folder / 'check-moe-oracle').is_file()
                or not module.can_tile_prefill_moe(layer, inputs, routes, 256)):
            return original(layer, inputs, routes)
        seen[layer_id] += 1
        input_a, input_b = inputs.detach().clone(), inputs.detach().clone()
        route_a, route_b = duplicate(routes), duplicate(routes)
        actual = original(layer, inputs, routes)
        snapshot = actual.detach().clone()
        control = layer.experts(input_a, route_a).detach().clone()
        repeat = layer.experts(input_b, route_b)
        row = dict(event='moe-oracle', layer=layer_id, rank=get_parallel().tp_rank,
                   sample=seen[layer_id], rows=inputs.shape[0], hidden=inputs.shape[-1],
                   dtype=str(inputs.dtype), quantization=type(layer.experts.quant_method).__name__,
                   tiled_vs_untiled=difference(snapshot, control), untiled_repeat=difference(repeat, control))
        with (folder / f'forward-rank-oracle-{os.getpid()}.jsonl').open('a') as f:
            print(json.dumps(row), file=f)
        return actual
    module.run_prefill_experts = measured


base_observe.install = install
