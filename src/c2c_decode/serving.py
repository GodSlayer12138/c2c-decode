"""Single GPU owner, bounded admission, streaming events and cooperative cancellation."""
from collections import deque
from dataclasses import dataclass, field
import math
from queue import Queue
from threading import Event, Lock
import time
import torch


@dataclass
class Job:
    request: object
    submitted: float
    deadline: float
    events: Queue = field(default_factory=Queue)
    cancel: Event = field(default_factory=Event)
    done: Event = field(default_factory=Event)
    tokens: list = field(default_factory=list)
    times_ms: list = field(default_factory=list)
    sequence: object = None
    iterator: object = None
    reservation: int = 0
    queue_ms: float = 0.
    reason: str = ""
    elapsed_ms: float = 0.


class ServingEngine:
    def __init__(self, reference, decoder, chunk_tokens=0, queue_limit=32):
        if chunk_tokens < 0 or queue_limit < 1:
            raise ValueError("Invalid chunk size or queue limit")
        self.reference, self.decoder = reference, decoder
        self.chunk_tokens, self.queue_limit = chunk_tokens, queue_limit
        self.pending, self.jobs, self.active = deque(), {}, []
        self.preparing, self.reserved = None, 0
        self.lock, self.wake, self.stop = Lock(), Event(), Event()
        self.failed = None
        self.peak_blocks = 0

    def submit(self, request, timeout_s=60):
        d = self.decoder
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be finite and positive")
        if not isinstance(request.max_new_tokens, int) or request.max_new_tokens < 1 or len(request.prompt_ids) < 2:
            raise ValueError("Invalid prompt or output length")
        if len(request.prompt_ids)+request.max_new_tokens > d.config.max_model_len:
            raise ValueError("Total length exceeds model limit")
        if math.ceil((len(request.prompt_ids)+request.max_new_tokens-1)/d.block_size) > d.num_blocks:
            raise ValueError("Request exceeds KV pool")
        with self.lock:
            if self.failed or self.stop.is_set():
                raise RuntimeError(self.failed or "Engine stopped")
            if len(self.pending) >= self.queue_limit:
                raise OverflowError("Admission queue full")
            if request.request_id in self.jobs:
                raise ValueError("Duplicate live request ID")
            now = time.perf_counter()
            job = Job(request, now, now+timeout_s)
            self.jobs[request.request_id] = job
            self.pending.append(job)
        self.wake.set()
        return job

    def cancel(self, request_id):
        with self.lock:
            job = self.jobs.get(request_id)
            if job:
                job.cancel.set()
        self.wake.set()
        return job is not None

    @staticmethod
    def interrupted(job):
        return "cancelled" if job.cancel.is_set() else "timeout" if time.perf_counter() >= job.deadline else None

    def finish(self, job, reason):
        if job.iterator is not None:
            job.iterator.close()
            job.iterator = None
        if job.sequence is not None:
            self.decoder.release(job.sequence)
            job.sequence = None
        self.reserved -= job.reservation
        job.reservation = 0
        job.reason, job.elapsed_ms = reason, (time.perf_counter()-job.submitted)*1000
        with self.lock:
            self.jobs.pop(job.request.request_id, None)
        job.events.put(dict(event="done", reason=reason, elapsed_ms=job.elapsed_ms))
        job.done.set()

    def emit(self, job, token):
        job.tokens.append(token)
        job.times_ms.append((time.perf_counter()-job.submitted)*1000)
        job.events.put(dict(event="token", token_id=token, index=len(job.tokens), elapsed_ms=job.times_ms[-1]))
        reason = ("eos" if not job.request.ignore_eos and token in self.reference.eos_ids else
                  "length" if len(job.tokens) >= job.request.max_new_tokens else None)
        if reason:
            self.finish(job, reason)
        return not reason

    @torch.inference_mode()
    def tick(self):
        d = self.decoder
        with self.lock:
            checked = [(j, self.interrupted(j)) for j in self.pending]
            removed = [(j, reason) for j, reason in checked if reason]
            self.pending = deque(j for j, reason in checked if not reason)
        for job, reason in removed:
            self.finish(job, reason)
        for job in list(self.active):
            if reason := self.interrupted(job):
                self.active.remove(job)
                self.finish(job, reason)
        if self.preparing and (reason := self.interrupted(self.preparing)):
            self.finish(self.preparing, reason)
            self.preparing = None
        while len(self.active) < d.config.max_num_seqs:
            if self.preparing is None:
                with self.lock:
                    if not self.pending:
                        break
                    job = self.pending[0]
                    demand = math.ceil((len(job.request.prompt_ids)+job.request.max_new_tokens-1)/d.block_size)
                    if self.reserved+demand > d.num_blocks:
                        break
                    self.preparing = job = self.pending.popleft()
                self.reserved += demand
                job.reservation = demand
                job.queue_ms = (time.perf_counter()-job.submitted)*1000
                if self.chunk_tokens:
                    job.iterator = self.reference.prefill_steps(job.request.prompt_ids, self.chunk_tokens)
            job = self.preparing
            handoff = next(job.iterator) if job.iterator is not None else self.reference.prefill(job.request.prompt_ids)
            if handoff is not None:
                job.sequence = d.import_cache(handoff)
                del handoff
                self.peak_blocks = max(self.peak_blocks, len(d.blocks.used_block_ids))
                self.preparing = None
                if reason := self.interrupted(job):
                    self.finish(job, reason)
                elif self.emit(job, job.sequence.last_token):
                    if job.iterator is not None:
                        job.iterator.close()
                        job.iterator = None
                    self.active.append(job)
            if self.chunk_tokens:
                break
        if self.active:
            # tolist waits for the sampled tokens; no redundant device-wide sync.
            tokens = d.step([j.sequence for j in self.active]).argmax(-1).tolist()
            self.peak_blocks = max(self.peak_blocks, len(d.blocks.used_block_ids))
            continuing = []
            for job, token in zip(self.active, tokens):
                job.sequence.append_token(token)
                if reason := self.interrupted(job):
                    self.finish(job, reason)
                elif self.emit(job, token):
                    continuing.append(job)
            self.active = continuing

    def run(self):
        try:
            while not self.stop.is_set():
                self.tick()
                with self.lock:
                    idle = not self.jobs
                if idle:
                    self.wake.wait(.05)
                    self.wake.clear()
        except Exception as error:
            self.failed = repr(error)
        finally:
            with self.lock:
                jobs = list(self.jobs.values())
            for job in jobs:
                self.finish(job, "error" if self.failed else "shutdown")
            self.active.clear()
            self.pending.clear()
            self.preparing = None
