"""Unmodified RosettaModel.forward/generate are the correctness authority."""
from dataclasses import dataclass
from pathlib import Path
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .paths import enable_upstreams, model_paths

enable_upstreams()
from rosetta.model.wrapper import RosettaModel
from rosetta.model.projector import load_projector


@dataclass
class Handoff:
    prompt_ids: list[int]
    cache: object
    logits: torch.Tensor
    timings: dict

    @property
    def first_token(self):
        return int(self.logits.argmax(-1).item())


class OfficialC2C:
    def __init__(self, lock_path="models.lock.json"):
        paths = model_paths(lock_path)
        self.tokenizer = AutoTokenizer.from_pretrained(paths["receiver"], local_files_only=True)
        models = [AutoModelForCausalLM.from_pretrained(
            paths[role], torch_dtype=torch.bfloat16, device_map={"": "cuda"},
            attn_implementation="sdpa", local_files_only=True).eval()
            for role in ("receiver", "sender")]
        directory = Path(paths["fuser"])
        files = sorted(directory.glob("projector_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))
        if not files:
            raise ValueError("No official Fuser weights found")
        projectors = []
        for i, file in enumerate(files):
            if file.stem != f"projector_{i}":
                raise ValueError("Non-contiguous projector files")
            proj = load_projector(str(file.with_suffix(".json")))
            proj.load_state_dict(torch.load(file, map_location="cpu", weights_only=True), strict=True)
            projectors.append(proj)
        self.model = RosettaModel(models, base_model_idx=0, projector_list=projectors).eval()
        self.model.load_projector_config(str(directory / "projector_config.json"))
        self.receiver = models[0]
        eos = self.receiver.generation_config.eos_token_id
        self.eos_ids = set(eos if isinstance(eos, list) else [eos])
        self.pad_token_id = self.receiver.generation_config.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = eos[0] if isinstance(eos, list) else eos

    def encode(self, prompt):
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=True,
            add_generation_prompt=True, enable_thinking=False)

    @staticmethod
    def inputs(ids):
        if len(ids) < 2:
            raise ValueError("C2C fusion needs at least two prompt tokens")
        tokens = torch.tensor([ids], dtype=torch.long, device="cuda")
        sections = [torch.tensor([1, 0], device="cuda").repeat(1, len(ids)-1, 1),
                    torch.tensor([[[-1, 0]]], device="cuda")]
        return dict(input_ids=tokens, attention_mask=torch.ones_like(tokens),
                    kv_cache_index=sections, use_cache=True)

    @torch.inference_mode()
    def prefill(self, ids):
        """N-1 tokens fuse, last prompt token runs through the fused cache.

        Model calls are instrumented without replacing any fusion arithmetic.
        Fusion/overhead is the remainder, including clones and Python dispatch.
        """
        events = []
        originals = []
        for i, model in enumerate(self.model.model_list):
            original = model.forward
            originals.append(original)

            def timed(*args, _fn=original, _idx=i, **kwargs):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                result = _fn(*args, **kwargs)
                end.record()
                events.append((_idx, start, end))
                return result

            model.forward = timed
        torch.cuda.synchronize()
        start = time.perf_counter()
        try:
            output = self.model(**self.inputs(ids))
            logits = output.logits[:, -1, :].clone()
            cache = output.past_key_values
            # Rosetta retains clean sender/receiver caches; no later decode needs them.
            self.model.kv_cache_dict.clear()
            torch.cuda.synchronize()
        finally:
            for model, original in zip(self.model.model_list, originals):
                model.forward = original
        total = (time.perf_counter() - start) * 1000
        base = [s.elapsed_time(e) for i, s, e in events if i == 0]
        sender = sum(s.elapsed_time(e) for i, s, e in events if i == 1)
        # The last sender forward ends before the official cache cloning/fusion
        # region; the second receiver forward begins immediately after it.
        sender_end = [e for i, s, e in events if i == 1][-1]
        handoff_start = [s for i, s, e in events if i == 0][-1]
        fusion_ms = sender_end.elapsed_time(handoff_start)
        timings = {"receiver_prefill_ms": base[0], "sender_prefill_ms": sender,
                   "last_prompt_token_ms": base[-1],
                   "prefill_ms": base[0] + sender,
                   "fusion_ms": fusion_ms,
                   "fusion_and_overhead_ms": max(0., total - sum(base) - sender),
                   "official_prefill_total_ms": total}
        if cache.get_seq_length() != len(ids):
            raise AssertionError("Official handoff cache length does not match prompt")
        return Handoff(list(ids), cache, logits, timings)

    @torch.inference_mode()
    def generate(self, ids, max_new_tokens, ignore_eos=False):
        # Disable EOS in the fixed-token correctness benchmark. The upstream
        # generate replaces EOS by pad_token_id, so compare its normal path separately.
        kwargs = {"eos_token_id": []} if ignore_eos else {}
        result = self.model.generate(**self.inputs(ids), do_sample=False,
                                     max_new_tokens=max_new_tokens, **kwargs)
        self.model.kv_cache_dict.clear()
        return result[0, len(ids):].tolist()

    @torch.inference_mode()
    def decode(self, token, cache):
        n = cache.get_seq_length()
        output = self.model(
            input_ids=torch.tensor([[token]], device="cuda"),
            attention_mask=torch.ones((1, n+1), dtype=torch.long, device="cuda"),
            position_ids=torch.tensor([[n]], device="cuda"),
            kv_cache_index=[torch.tensor([[[-1, 0]]], device="cuda")],
            past_key_values=cache, use_cache=True)
        return output.logits[:, -1, :], output.past_key_values
