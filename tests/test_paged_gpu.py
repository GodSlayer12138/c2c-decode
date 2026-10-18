"""Real CUDA copies and block tables without downloading any model weights."""
from types import SimpleNamespace
import pytest
import torch
from transformers import DynamicCache
from c2c_decode.reference import Handoff
from c2c_decode.paged import PagedDecoder, BlockManager
from nanovllm.utils.context import get_context

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


def decoder(mode="scatter"):
    d = PagedDecoder.__new__(PagedDecoder)
    d.config = SimpleNamespace(max_model_len=2048, max_num_seqs=8)
    d.block_size, d.num_blocks = 256, 12
    d.kv_cache = torch.full((2, 3, 12, 256, 2, 4), -99., device="cuda", dtype=torch.bfloat16)
    d.blocks = BlockManager(d.num_blocks, d.block_size)
    d._live = {}
    d.import_mode = mode
    return d


def handoff(n, offset=0):
    cache = DynamicCache()
    for layer in range(3):
        # Contiguous sequence-major backing with non-contiguous HF layout.
        x = (torch.arange(n*8, device="cuda").reshape(1, n, 2, 4) % 31).to(torch.bfloat16)
        x = (x + layer + offset).transpose(1, 2)
        cache.update(x, -x, layer)
    logits = torch.zeros(1, 16, device="cuda")
    logits[0, 7] = 1
    return Handoff([2]*n, cache, logits, {})


@pytest.mark.parametrize("n", [31, 255, 256, 257, 1024])
@pytest.mark.parametrize("mode", ["loop", "scatter"])
@pytest.mark.parametrize("fast", [False, True])
def test_lossless_import_positions_and_growth(n, mode, fast):
    d, h = decoder(mode), handoff(n)
    d.fast_inputs = fast
    seq = d.import_cache(h)
    assert seq.last_token == 7 and len(seq) == n+1 and seq.num_cached_tokens == n
    for old, new in zip(h.cache, d.export_cache(seq, n)):
        assert all(torch.equal(a, b) for a, b in zip(old, new))
    if n % d.block_size:
        assert (d.kv_cache[:, :, seq.block_table[-1], n % d.block_size:] == -99).all()

    def run(ids, positions, is_prefill):
        ctx = get_context()
        assert ids.tolist() == [7] and positions.tolist() == [n]
        assert ctx.context_lens.tolist() == [n+1]
        assert ctx.slot_mapping.tolist() == [seq.block_table[n//256]*256+n%256]
        assert not is_prefill
        return h.logits

    d.run_model = run
    d.step([seq])
    assert seq.num_cached_tokens == n+1
    assert len(seq.block_table) == (n+256)//256
    d.release(seq)
    assert len(d.blocks.free_block_ids) == d.num_blocks


@pytest.mark.parametrize("mode", ["loop", "scatter"])
def test_same_tokens_different_fused_state_never_share_blocks(mode):
    d = decoder(mode)
    a, b = handoff(257), handoff(257, 8)
    s1, s2 = d.import_cache(a), d.import_cache(b)
    assert set(s1.block_table).isdisjoint(s2.block_table)
    d.release(s1)
    # Fragmentation/reuse must not overwrite the still-live request.
    s3 = d.import_cache(handoff(513, 12))
    for old, new in zip(b.cache, d.export_cache(s2, 257)):
        assert all(torch.equal(x, y) for x, y in zip(old, new))
    d.release(s2)
    d.release(s3)
    assert len(d.blocks.free_block_ids) == d.num_blocks
    assert not d.blocks.hash_to_block_id


def test_bad_shape_and_capacity_leave_pool_unchanged():
    d = decoder()
    bad = handoff(256)
    bad.cache.key_cache[0] = bad.cache.key_cache[0].float()
    with pytest.raises(ValueError, match="BF16"):
        d.import_cache(bad)
    seq = d.import_cache(handoff(1800))
    before = set(d.blocks.free_block_ids)
    with pytest.raises(MemoryError):
        d.import_cache(handoff(1800))
    assert set(d.blocks.free_block_ids) == before
    d.release(seq)
    with pytest.raises(ValueError):
        d.release(seq)
