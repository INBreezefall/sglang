"""Packed CSA and index cache behavior on an Ascend 910B."""

import unittest

import torch

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.dsv4 import dsv41_int8_indexer as indexer
from sglang.srt.hardware_backend.npu.dsv4.dsv4_kv_layout import NPUDSV4KVLayout
from sglang.srt.hardware_backend.npu.dsv4.dsv41_int8_kv import (
    decode_int8,
    encode_int8,
    pack_kv,
    unpack_kv,
)
from sglang.srt.hardware_backend.npu.utils import dsv41_a2_int8_cache_modes


class TestDSV41A2Int8Cache(unittest.TestCase):
    def test_independent_910b_formats(self):
        self.assertTrue(torch.npu.get_device_name().startswith("Ascend910B"))
        for main in (False, True):
            for index in (False, True):
                with (
                    envs.SGLANG_DSV41_NPU_INT8_MAIN_CACHE.override(main),
                    envs.SGLANG_DSV41_NPU_INT8_INDEX_CACHE.override(index),
                ):
                    self.assertEqual(dsv41_a2_int8_cache_modes(), (main, index))
        self.assertEqual(NPUDSV4KVLayout.V41_INT8.kv_info(448, 64), (896, torch.uint8))
        self.assertEqual(indexer.INDEX_METADATA_BYTES, 98)
        self.assertTrue(NPUDSV4KVLayout.V41_INT8.is_compressed)

    def test_graph_replay_observes_new_signed_bytes(self):
        gen = torch.Generator().manual_seed(61121)

        def inputs():
            return torch.randn((3, 1, 512), generator=gen).bfloat16().npu()

        live = inputs()
        for _ in range(2):
            unpack_kv(pack_kv(live))
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            packed = pack_kv(live)
            decoded = unpack_kv(packed)
        for _ in range(2):
            fresh = inputs()
            reference = pack_kv(fresh)
            live.copy_(fresh)
            graph.replay()
            torch.npu.synchronize()
            torch.testing.assert_close(packed, reference, rtol=0, atol=0)
            torch.testing.assert_close(decoded, unpack_kv(reference), rtol=0, atol=0)

    def test_pruning_and_reranking_use_the_same_logical_pages(self):
        gen = torch.Generator().manual_seed(226512)
        original = torch.randn((5, 64, 1, 128), generator=gen).bfloat16().npu()
        keys, metadata = encode_int8(original, 128)
        query = torch.randn((1, 32, 128), generator=gen).bfloat16().npu()
        weight = torch.ones((1, 32), device="npu")
        table = torch.tensor([[4, 1, 3]], dtype=torch.int32, device="npu")
        used = torch.tensor([180], dtype=torch.int32, device="npu")
        for ratio in (1, 2):
            for filtered in (False, True):
                kwargs = dict(block_table=table, seqused_k=used, cmp_ratio=ratio)
                q, bank, w, qs, ks = indexer.prepare_indexer_operands(
                    query,
                    keys,
                    weight,
                    metadata,
                    used * ratio,
                    (used * ratio).cpu(),
                    kwargs,
                    capture=False,
                    capacity=128,
                    filtered=filtered,
                )
                if filtered:
                    self.assertIsNone(qs)
                    self.assertIsNone(ks)
                    torch.testing.assert_close(q, query, atol=0, rtol=0)
                    page = kwargs["block_table"][0, 1]
                    torch.testing.assert_close(
                        bank[page],
                        decode_int8(keys[1], metadata[1], 128),
                        atol=0,
                        rtol=0,
                    )
                else:
                    self.assertEqual(
                        (q.dtype, bank.dtype, w.dtype),
                        (torch.int8, torch.int8, torch.float16),
                    )
                    self.assertEqual(
                        (qs.dtype, ks.dtype), (torch.float16, torch.float16)
                    )

    def test_public_candidate_source_matches_a2_results(self):
        try:
            from sgl_kernel_npu.operators.indexer.indexer_utils import (  # noqa: F401
                validate_quant_indexer_candidates,
            )
        except ImportError:
            self.skipTest(
                "The installed sgl-kernel-npu predates INT8 candidate modes (#1045)"
            )
        from sgl_kernel_npu.operators.indexer.quant_lightning_indexer import (
            quant_lightning_indexer,
        )

        gen = torch.Generator().manual_seed(8264)
        q, qs = indexer.quantize_rows(
            torch.randn((2, 32, 128), generator=gen).bfloat16().npu()
        )
        k, ks = indexer.quantize_rows(
            torch.randn((274, 64, 1, 128), generator=gen).bfloat16().npu()
        )
        w = torch.rand((2, 32), generator=gen).half().npu()
        table = (torch.randperm(272, generator=gen) + 1).int().npu()[None]
        kw = dict(
            cu_seqlens_q=torch.tensor([0, 2], dtype=torch.int32, device="npu"),
            seqused_k=torch.tensor([17405], dtype=torch.int32, device="npu"),
            cmp_residual_k=None,
            block_table=table,
            cmp_ratio=1,
            layout_q="TND",
            layout_k="PA_BBND",
            mask_mode=3,
        )
        ids, blocks = indexer.index_topk(
            q, k, w, qs, ks[..., None], 512, candidate_topk_blocks=2048, **kw
        )
        reference = quant_lightning_indexer(
            q,
            k,
            w,
            qs,
            ks,
            512,
            2,
            impl="torch_native",
            candidate_mode=1,
            candidate_topk_blocks=2048,
            candidate_block_size=8,
            **kw,
        )
        torch.testing.assert_close(blocks, reference[2], atol=0, rtol=0)
        for actual, expected in zip(
            ids.cpu().tolist(), reference[0].flatten(1).cpu().tolist()
        ):
            self.assertGreater(len(set(actual) & set(expected)) / 512, 0.99)


if __name__ == "__main__":
    unittest.main()
