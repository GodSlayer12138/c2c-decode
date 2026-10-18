"""Opt-in producer and input preparation experiments; official math stays upstream."""
from contextlib import contextmanager
import time
import torch
from .reference import OfficialC2C


class OptimizedC2C(OfficialC2C):
    @contextmanager
    def last_logits(self, cached=None):
        originals = []
        for index, model in enumerate(self.model.model_list):
            original = model.forward
            originals.append(original)
            def forward(*args, _fn=original, _index=index, **kwargs):
                if cached is not None and _index in cached:
                    return cached.pop(_index)
                kwargs["logits_to_keep"] = 1
                return _fn(*args, **kwargs)
            model.forward = forward
        try:
            yield
        finally:
            for model, original in zip(self.model.model_list, originals):
                model.forward = original
            self.model.kv_cache_dict.clear()

    @torch.inference_mode()
    def prefill(self, ids):
        with self.last_logits():
            return super().prefill(ids)

    @torch.inference_mode()
    def prefill_steps(self, ids, chunk_tokens):
        """Yield between clean-prefix chunks; replay their outputs into official fusion.

        Neither model ever sees fused KV while building its clean prefix. The
        original Rosetta forward still owns cloning, Fuser math and final token.
        Only one producer job may be advanced at a time on this model instance.
        """
        if chunk_tokens < 1 or len(ids) < 2:
            raise ValueError("Positive chunk size and at least two tokens required")
        cached, phases = {}, {}
        for index, model in enumerate(self.model.model_list):
            cache, elapsed = None, 0.
            for start in range(0, len(ids)-1, chunk_tokens):
                end = min(start+chunk_tokens, len(ids)-1)
                before = time.perf_counter()
                positions = torch.arange(start, end, device="cuda")
                output = model(input_ids=torch.tensor([ids[start:end]], device="cuda"),
                    attention_mask=torch.ones((1, end), dtype=torch.long, device="cuda"),
                    position_ids=positions.unsqueeze(0), cache_position=positions,
                    past_key_values=cache, use_cache=True, logits_to_keep=1)
                cache = output.past_key_values
                torch.cuda.synchronize()
                elapsed += (time.perf_counter()-before)*1000
                yield None
            cached[index] = output
            phases[f"clean_model_{index}_ms"] = elapsed
        before = time.perf_counter()
        with self.last_logits(cached):
            handoff = super().prefill(ids)
        handoff.timings.update(phases, final_fusion_and_token_ms=(time.perf_counter()-before)*1000,
            chunk_tokens=chunk_tokens)
        yield handoff


class DecodeInputs:
    """Two persistent pinned staging buffers; event protects asynchronous H2D reuse."""
    def __init__(self, batch, blocks):
        self.batch, self.blocks = batch, blocks
        self.long = torch.empty((2, batch), dtype=torch.int64, pin_memory=True)
        self.int = torch.empty(batch*(blocks+2), dtype=torch.int32, pin_memory=True)
        self.long_gpu, self.int_gpu = self.long.to("cuda"), self.int.to("cuda")
        self.long_np, self.int_np = self.long.numpy(), self.int.numpy()
        self.copied = torch.cuda.Event()
        self.pending = False

    def prepare(self, seqs, block_size):
        from nanovllm.utils.context import set_context
        if self.pending:
            self.copied.synchronize()
        self.int_np.fill(0)
        for row, seq in enumerate(seqs):
            n = len(seq)
            self.long_np[:, row] = (seq.last_token, n-1)
            self.int_np[row] = seq.block_table[(n-1)//block_size]*block_size+(n-1)%block_size
            self.int_np[self.batch+row] = n
            self.int_np[2*self.batch:].reshape(self.batch, self.blocks)[row, :len(seq.block_table)] = seq.block_table
        self.long_gpu.copy_(self.long, non_blocking=True)
        self.int_gpu.copy_(self.int, non_blocking=True)
        self.copied.record()
        self.pending = True
        n = len(seqs)
        set_context(False, slot_mapping=self.int_gpu[:n], context_lens=self.int_gpu[self.batch:self.batch+n],
                    block_tables=self.int_gpu[2*self.batch:].view(self.batch, self.blocks)[:n])
        return self.long_gpu[0, :n], self.long_gpu[1, :n]
