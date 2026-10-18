"""Deterministic service lifecycle tests with the allocator fake, no model weights."""
import time
from threading import Thread
from unittest.mock import patch
import pytest
from c2c_decode.scheduler import Request
from c2c_decode.serving import ServingEngine
from test_scheduler import FakeDecoder, FakeReference


class ChunkReference(FakeReference):
    def prefill_steps(self, ids, chunk_tokens):
        yield None
        yield None
        yield self.prefill(ids)


@pytest.fixture(autouse=True)
def no_cuda_sync():
    with patch("torch.cuda.synchronize"):
        yield


def test_bounded_queue_timeout_cancel_and_reuse():
    d = FakeDecoder()
    engine = ServingEngine(ChunkReference(), d, chunk_tokens=2, queue_limit=2)
    a = engine.submit(Request("a", [2]*3, 4, True))
    b = engine.submit(Request("b", [2]*3, 4, True))
    with pytest.raises(OverflowError):
        engine.submit(Request("c", [2]*3, 4, True))
    engine.tick()  # a's first chunk: reservation exists, no page yet.
    assert a.reservation and a.sequence is None
    assert engine.cancel("a")
    b.deadline = time.perf_counter()-1
    engine.tick()
    assert a.reason == "cancelled" and b.reason == "timeout"
    assert not engine.jobs and engine.reserved == 0 and len(d.blocks.free_block_ids) == 4
    c = engine.submit(Request("c", [2]*3, 3, True))
    for _ in range(8):
        engine.tick()
    assert c.tokens == [7, 1, 1] and c.reason == "length"
    assert len(c.times_ms) == 3 and len(d.blocks.free_block_ids) == 4


def test_active_cancellation_and_worker_failure_cleanup():
    d = FakeDecoder()
    engine = ServingEngine(FakeReference(), d)
    a = engine.submit(Request("a", [2]*3, 8, True))
    engine.tick()
    engine.cancel("a")
    engine.tick()
    assert a.reason == "cancelled" and not d._live
    b = engine.submit(Request("b", [2]*3, 8, True))
    with patch.object(d, "step", side_effect=RuntimeError("injected failure")):
        engine.run()
    assert b.reason == "error" and engine.failed and engine.reserved == 0 and not d._live
    with pytest.raises(RuntimeError):
        engine.submit(Request("c", [2]*3, 2))


def test_shutdown_reclaims_preparing_active_and_pending():
    d = FakeDecoder()
    engine = ServingEngine(ChunkReference(), d, chunk_tokens=2)
    jobs = [engine.submit(Request(str(i), [2]*3, 8, True)) for i in range(3)]
    for _ in range(4):
        engine.tick()
    engine.stop.set()
    engine.run()
    assert all(j.done.is_set() and j.reason == "shutdown" for j in jobs)
    assert engine.reserved == 0 and len(d.blocks.free_block_ids) == 4


def test_future_deadline_and_duplicates_rejected():
    engine = ServingEngine(FakeReference(), FakeDecoder())
    for timeout in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            engine.submit(Request("bad", [2]*3, 2), timeout)
    engine.submit(Request("same", [2]*3, 2))
    with pytest.raises(ValueError):
        engine.submit(Request("same", [2]*3, 2))
