"""Bounded KV storage and decode execution using nano-vLLM's model/kernels."""
import math
import tempfile
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.distributed as dist
from transformers import AutoConfig
from .paths import enable_upstreams, model_paths

enable_upstreams()
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.utils.loader import load_model
from nanovllm.utils.context import reset_context


class PagedDecoder(ModelRunner):
    """Single GPU only. No automatic whole-GPU memory sizing or prefix hashing.

    A token-only prefix hash is invalid for externally fused KV: identical tokens
    can represent different sender/Fuser/context states. Blocks remain private.
    """
    def __init__(self, lock_path="models.lock.json", kv_pool_mib=1024,
                 max_model_len=4096, max_batch=8, graph=False, reserve_mib=2048,
                 import_mode="scatter", fast_inputs=False):
        if import_mode not in ("loop", "scatter"):
            raise ValueError("import_mode must be loop or scatter")
        self.import_mode = import_mode
        self.fast_inputs = fast_inputs
        if kv_pool_mib <= 0 or max_batch not in (1, 2, 4, 8):
            raise ValueError("Positive pool size and batch in 1/2/4/8 required")
        if not 2 <= max_model_len <= 4096:
            raise ValueError("Initial implementation limits total length to 4096")
        if dist.is_initialized():
            raise RuntimeError("Use one PagedDecoder per process")
        model_path = model_paths(lock_path)["receiver"]
        cfg = AutoConfig.from_pretrained(model_path, local_files_only=True)
        self.block_size = Sequence.block_size  # FlashAttention paged block: 256
        self.rank, self.world_size = 0, 1
        self.enforce_eager = not graph
        self.config = SimpleNamespace(max_num_seqs=max_batch, max_model_len=max_model_len, hf_config=cfg)
        self._rendezvous = tempfile.TemporaryDirectory(prefix="c2c-dist-")
        dist.init_process_group("nccl", init_method=Path(self._rendezvous.name, "init").as_uri(),
                                rank=0, world_size=1)
        torch.cuda.set_device(0)
        previous_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16)
            with torch.device("cuda"):
                self.model = Qwen3ForCausalLM(cfg).eval()
                load_model(self.model, model_path)
                heads, dim = cfg.num_key_value_heads, cfg.head_dim
                block_bytes = 2 * cfg.num_hidden_layers * self.block_size * heads * dim * 2
                self.num_blocks = int(kv_pool_mib * 2**20) // block_bytes
                if self.num_blocks < math.ceil(max_model_len / self.block_size):
                    raise ValueError("KV pool cannot hold one maximum-length request")
                needed = self.num_blocks * block_bytes
                if needed + reserve_mib * 2**20 > torch.cuda.mem_get_info()[0]:
                    raise MemoryError("KV pool would violate the explicit free-memory reserve")
                self.kv_cache = torch.empty(2, cfg.num_hidden_layers, self.num_blocks,
                                            self.block_size, heads, dim, dtype=torch.bfloat16)
                self.pool_bytes = self.kv_cache.numel() * self.kv_cache.element_size()
                for layer, module in enumerate(self.model.model.layers):
                    module.self_attn.attn.k_cache = self.kv_cache[0, layer]
                    module.self_attn.attn.v_cache = self.kv_cache[1, layer]
                if graph:
                    self.capture_cudagraph()
        except BaseException:
            dist.destroy_process_group()
            self._rendezvous.cleanup()
            raise
        finally:
            torch.set_default_dtype(previous_dtype)
        self.blocks = BlockManager(self.num_blocks, self.block_size)
        self._live = {}

    def new_sequence(self, prompt_ids):
        if not 2 <= len(prompt_ids) < self.config.max_model_len:
            raise ValueError("Invalid prompt length")
        seq = Sequence(prompt_ids)
        seq.is_prefill = False
        return seq

    def prepare_decode(self, seqs):
        if not getattr(self, "fast_inputs", False):
            return super().prepare_decode(seqs)
        if not hasattr(self, "input_buffers"):
            from .optimized import DecodeInputs
            self.input_buffers = DecodeInputs(self.config.max_num_seqs,
                math.ceil(self.config.max_model_len/self.block_size))
        return self.input_buffers.prepare(seqs, self.block_size)

    def _validate_cache(self, handoff):
        cache, n = handoff.cache, len(handoff.prompt_ids)
        expected = (1, self.kv_cache.shape[-2], n, self.kv_cache.shape[-1])
        if len(cache) != self.kv_cache.shape[1] or cache.get_seq_length() != n:
            raise ValueError("Wrong cache layers or sequence length")
        for k, v in cache:
            for value in (k, v):
                if tuple(value.shape) != expected or value.dtype != torch.bfloat16:
                    raise ValueError(f"Cache must be BF16 with shape {expected}")
                if value.device != self.kv_cache.device:
                    raise ValueError("Cache and pool must be on the same CUDA device")

    @torch.inference_mode()
    def import_cache(self, handoff):
        self._validate_cache(handoff)
        seq = self.new_sequence(handoff.prompt_ids)
        if len(self.blocks.free_block_ids) < seq.num_blocks:
            raise MemoryError("Not enough free KV blocks")
        self.blocks.allocate(seq, num_cached_blocks=0)
        self._live[seq.seq_id] = seq
        try:
            # HF [1,H,N,D] -> nano [physical_block, offset,H,D]. Already
            # post-RoPE: apply no positional transform during import.
            if getattr(self, "import_mode", "scatter") == "scatter":
                positions = torch.arange(len(seq), device=self.kv_cache.device)
                table = torch.tensor(seq.block_table, device=self.kv_cache.device)
                physical, offsets = table[positions // self.block_size], positions % self.block_size
                for layer, pair in enumerate(handoff.cache):
                    for kind, source in enumerate(pair):
                        self.kv_cache[kind, layer, physical, offsets] = source[0].transpose(0, 1)
            else:
                # Retained as the measured pre-optimization baseline.
                for layer, (key, value) in enumerate(handoff.cache):
                    for logical, physical in enumerate(seq.block_table):
                        start = logical * self.block_size
                        end = min(start + self.block_size, len(seq))
                        for kind, source in enumerate((key, value)):
                            self.kv_cache[kind, layer, physical, :end-start].copy_(
                                source[0, :, start:end].transpose(0, 1))
            seq.num_cached_tokens = len(seq)
            seq.append_token(handoff.first_token)
            return seq
        except BaseException:
            self.release(seq)
            raise

    @torch.inference_mode()
    def export_cache(self, seq, length):
        """Debug-only round trip; never called by the serving loop."""
        result = []
        for layer in range(self.kv_cache.shape[1]):
            pair = []
            for kind in range(2):
                flat = torch.cat([self.kv_cache[kind, layer, block]
                                  for block in seq.block_table], dim=0)[:length]
                pair.append(flat.transpose(0, 1).unsqueeze(0).contiguous())
            result.append(tuple(pair))
        return result

    @torch.inference_mode()
    def step(self, seqs):
        if not seqs or len(seqs) > self.config.max_num_seqs:
            raise ValueError("Empty batch or batch exceeds configured maximum")
        if len({s.seq_id for s in seqs}) != len(seqs):
            raise ValueError("Duplicate sequence in decode batch")
        needed = 0
        for seq in seqs:
            if self._live.get(seq.seq_id) is not seq:
                raise ValueError("Sequence is not owned by this decoder")
            if len(seq) > self.config.max_model_len or seq.num_cached_tokens != len(seq)-1:
                raise ValueError("Decode needs exactly one uncached token at a valid position")
            needed += int(len(seq) % self.block_size == 1)
        if len(self.blocks.free_block_ids) < needed:
            raise MemoryError("Decode growth needs more blocks; admission reservation violated")
        for seq in seqs:
            self.blocks.may_append(seq)
        try:
            ids, positions = self.prepare_decode(seqs)
            logits = self.run_model(ids, positions, False)
            for seq in seqs:
                seq.num_cached_tokens = len(seq)
            return logits
        finally:
            reset_context()

    def release(self, seq):
        if self._live.pop(seq.seq_id, None) is not seq:
            raise ValueError("Sequence already released or foreign")
        self.blocks.deallocate(seq)

    def close(self):
        for seq in list(self._live.values()):
            self.release(seq)
        torch.cuda.synchronize()
        if dist.is_initialized():
            dist.destroy_process_group()
        self._rendezvous.cleanup()
        for name in ("graphs", "graph_vars", "graph_pool", "model", "kv_cache", "input_buffers"):
            if hasattr(self, name):
                delattr(self, name)
