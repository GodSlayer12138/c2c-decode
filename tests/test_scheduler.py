"""CPU tests for admission, failure cleanup, EOS and mixed-length completion."""
from types import SimpleNamespace
from unittest.mock import patch
import pytest
import torch
from c2c_decode.scheduler import Request, Scheduler


class Seq:
    def __init__(self, sid, ids, blocks):
        self.seq_id, self.ids, self.block_table = sid, list(ids), blocks
        self.completion_token_ids = [7]

    @property
    def last_token(self): return self.completion_token_ids[-1]

    @property
    def num_completion_tokens(self): return len(self.completion_token_ids)

    def append_token(self, t): self.completion_token_ids.append(t)


class FakeDecoder:
    block_size, num_blocks, pool_bytes = 4, 4, 1024

    def __init__(self):
        self.config = SimpleNamespace(max_num_seqs=2, max_model_len=16)
        self.blocks = SimpleNamespace(free_block_ids=set(range(4)), used_block_ids=set())
        self._live = {}
        self.sid = 0
        self.batches = []

    def import_cache(self, handoff):
        n = (len(handoff.prompt_ids)+3)//4
        blocks = [self.blocks.free_block_ids.pop() for _ in range(n)]
        self.blocks.used_block_ids.update(blocks)
        seq = Seq(self.sid, handoff.prompt_ids, blocks)
        self.sid += 1
        self._live[seq.seq_id] = seq
        return seq

    def step(self, seqs):
        self.batches.append(len(seqs))
        for seq in seqs:
            n = (len(seq.ids)+len(seq.completion_token_ids)+3)//4
            while len(seq.block_table) < n:
                b = self.blocks.free_block_ids.pop()
                self.blocks.used_block_ids.add(b)
                seq.block_table.append(b)
        return torch.tensor([[0., 1.]] * len(seqs))

    def release(self, seq):
        del self._live[seq.seq_id]
        for b in seq.block_table:
            self.blocks.used_block_ids.remove(b)
            self.blocks.free_block_ids.add(b)


class FakeReference:
    eos_ids = {7}
    pad_token_id = 0

    def prefill(self, ids):
        return SimpleNamespace(prompt_ids=ids, timings={})


@pytest.fixture(autouse=True)
def cpu_only():
    with patch("torch.cuda.synchronize"):
        yield


def test_reservation_backpressure_and_reuse():
    d = FakeDecoder()
    scheduler = Scheduler(FakeReference(), d)
    requests = [Request(str(i), [2]*7, 5, True) for i in range(5)]
    result = scheduler.run(requests)
    assert all(r.token_ids == [7, 1, 1, 1, 1] for r in result)
    assert max(d.batches) == 1  # Three reserved blocks per request, pool has four.
    assert len(d.blocks.free_block_ids) == 4 and not d._live
    assert scheduler.run(requests)[0].token_ids == result[0].token_ids


def test_mixed_lengths_and_first_token_eos():
    d = FakeDecoder()
    scheduler = Scheduler(FakeReference(), d)
    result = scheduler.run([Request("a", [2]*3, 4, True), Request("b", [2]*3, 2, True),
                            Request("c", [2]*2, 4, False), Request("d", [2]*2, 1, True)])
    assert [len(r.token_ids) for r in result] == [4, 2, 1, 1]
    assert result[2].finish_reason == "eos"
    assert result[2].token_ids == [0]  # Exact official terminal-token convention.
    assert 2 in d.batches and len(d.blocks.free_block_ids) == 4


def test_failure_reclaims_all_blocks():
    d = FakeDecoder()
    scheduler = Scheduler(FakeReference(), d)
    with patch.object(d, "step", side_effect=RuntimeError("GPU failure")):
        with pytest.raises(RuntimeError, match="GPU failure"):
            scheduler.run([Request("a", [2]*3, 4, True)])
    assert len(d.blocks.free_block_ids) == 4 and not d._live


def test_rejects_oversized_or_duplicate_requests():
    scheduler = Scheduler(FakeReference(), FakeDecoder())
    with pytest.raises(ValueError):
        scheduler.run([Request("a", [2]*15, 2)])
    with pytest.raises(ValueError):
        scheduler.run([Request("a", [2]*3), Request("a", [2]*3)])


def test_future_arrivals_and_token_timestamps(monkeypatch):
    clock = [100.]
    monkeypatch.setattr("c2c_decode.scheduler.time.perf_counter", lambda: clock[0])
    monkeypatch.setattr("c2c_decode.scheduler.time.sleep", lambda seconds: clock.__setitem__(0, clock[0]+seconds))
    scheduler = Scheduler(FakeReference(), FakeDecoder(), prefill_limit_per_round=1)
    result = scheduler.run([Request("late", [2]*3, 2, True, 20), Request("early", [2]*3, 2, True)])
    assert [r.request_id for r in result] == ["late", "early"]
    assert result[0].timings["scheduled_arrival_ms"] == 20
    assert all(len(r.timings["token_times_ms"]) == len(r.token_ids) for r in result)
    assert clock[0] == 100.02


def test_prefill_quantum_handles_immediate_completion():
    d = FakeDecoder()
    result = Scheduler(FakeReference(), d, prefill_limit_per_round=1).run(
        [Request(str(i), [2]*3, 1, True) for i in range(4)])
    assert len(result) == 4 and len(d.blocks.free_block_ids) == 4


def test_prefill_quantum_interleaves_decode():
    d = FakeDecoder()
    result = Scheduler(FakeReference(), d, prefill_limit_per_round=1).run(
        [Request("a", [2]*3, 3, True), Request("b", [2]*3, 3, True)])
    assert d.batches == [1, 2, 1]
    assert all(len(r.token_ids) == 3 for r in result)
    with pytest.raises(ValueError, match="arrival_ms"):
        Scheduler(FakeReference(), d).run([Request("bad", [2]*3, 2, True, float("nan"))])
