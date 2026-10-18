"""Matched arrival replay, real loopback SSE lifecycle, and bounded duration soak."""
import gc
import json
import math
from pathlib import Path
import random
from threading import Thread
import time
from types import SimpleNamespace
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor
import torch
from c2c_decode.optimized import OptimizedC2C
from c2c_decode.reference import OfficialC2C
from c2c_decode.paged import PagedDecoder
from c2c_decode.scheduler import Request
from c2c_decode.serving import ServingEngine
from c2c_decode.server import make_server
from c2c_decode.evaluate import token_prompt
from profiling_utils import Hooks

OUT = Path("results/next")


def save(name, data):
    (OUT/name).write_text(json.dumps(data, indent=2)+"\n")


def percentiles(values):
    values = sorted(values)
    def p(q):
        x = (len(values)-1)*q
        lo, hi = math.floor(x), math.ceil(x)
        return values[lo] if lo == hi else values[lo]*(hi-x)+values[hi]*(x-lo)
    return {f"p{int(q*100)}": p(q) for q in (.5, .95, .99)} if values else {}


def job_record(job):
    return dict(id=job.request.request_id, input_tokens=len(job.request.prompt_ids), tokens=job.tokens,
        queue_ms=job.queue_ms, token_times_ms=job.times_ms, reason=job.reason, e2e_ms=job.elapsed_ms)


def summarize(jobs, wall):
    complete = [j for j in jobs if j.reason == "length"]
    intervals = [b-a for j in complete for a, b in zip(j.times_ms, j.times_ms[1:])]
    # A declared, fixed illustrative SLO, not chosen after looking at the results.
    good = [j for j in complete if j.times_ms[0] <= 1000 and
        max((b-a for a, b in zip(j.times_ms, j.times_ms[1:])), default=0) <= 100]
    return dict(requests=len(jobs), completed=len(complete), wall_s=wall,
        tokens_per_s=sum(len(j.tokens) for j in complete)/wall,
        slo=dict(ttft_ms=1000, max_inter_token_ms=100, good_requests=len(good), good_requests_per_s=len(good)/wall),
        ttft_ms=percentiles([j.times_ms[0] for j in complete]),
        queue_ms=percentiles([j.queue_ms for j in complete]),
        tpot_ms=percentiles([(j.times_ms[-1]-j.times_ms[0])/(len(j.times_ms)-1) for j in complete]),
        interval_ms=percentiles(intervals), max_interval_ms=max(intervals, default=0))


def replay(reference, decoder):
    ids = {n: token_prompt(reference, n) for n in (128, 512, 1024, 2048)}
    original = SimpleNamespace(prefill=lambda ids: OfficialC2C.prefill(reference, ids), eos_ids=reference.eos_ids)
    configs = [("official", original, 0, False), ("lean", reference, 0, False),
        ("lean-fast", reference, 0, True), ("chunk-256", reference, 256, True),
        ("chunk-512", reference, 512, True), ("chunk-1024", reference, 1024, True)]
    rows = []
    for gap in (0, 100, 350):
        for repeat in range(3):
            rng = random.Random(20261018+repeat)
            arrivals, at = [], 0.
            for i in range(24):
                at += rng.expovariate(1000/gap) if gap else 0.
                arrivals.append((at, rng.choice([128, 512, 1024, 2048]), rng.choice([16, 32, 64, 128])))
            for name, producer, chunk, fast in rng.sample(configs, len(configs)):
                decoder.fast_inputs = fast
                engine = ServingEngine(producer, decoder, chunk, queue_limit=32)
                start, index, jobs = time.perf_counter(), 0, []
                while index < len(arrivals) or engine.jobs:
                    now = time.perf_counter()-start
                    while index < len(arrivals) and arrivals[index][0] <= now:
                        due, n, count = arrivals[index]
                        job = engine.submit(Request(str(index), ids[n], count, True), 120)
                        job.submitted = start+due  # Include delay from scheduled arrival, avoiding coordinated omission.
                        jobs.append(job)
                        index += 1
                    engine.tick()
                    if not engine.jobs and index < len(arrivals):
                        time.sleep(min(max(0, arrivals[index][0]-(time.perf_counter()-start)), .02))
                wall = time.perf_counter()-start
                assert len(decoder.blocks.free_block_ids) == decoder.num_blocks and engine.reserved == 0
                rows.append(dict(mode=name, gap_ms=gap, repeat=repeat, arrivals=arrivals,
                    summary=summarize(jobs, wall), jobs=[job_record(j) for j in jobs],
                    free_blocks_after=len(decoder.blocks.free_block_ids), peak_blocks=engine.peak_blocks))
                save("service-load.json", dict(runs=rows))
                print("LOAD", name, gap, repeat, flush=True)


def http_checks(reference, decoder):
    engine = ServingEngine(reference, decoder, queue_limit=2)
    server = make_server(engine, 0)
    server_thread = Thread(target=server.serve_forever)
    server_thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    def call(path, data=None, method=None):
        request = urllib.request.Request(base+path, data=json.dumps(data).encode() if data else None,
            headers={"Content-Type": "application/json"}, method=method)
        return urllib.request.urlopen(request, timeout=30)
    def events(response):
        for line in response:
            if line.startswith(b"data: "):
                yield json.loads(line[6:])
    report = {}
    a = engine.submit(Request("queue-a", reference.encode("Hello"), 4))
    b = engine.submit(Request("queue-b", reference.encode("Hello"), 4))
    try:
        call("/generate", dict(prompt="Hi"))
    except urllib.error.HTTPError as error:
        report["queue_full_status"] = error.code
    engine.cancel("queue-a")
    engine.cancel("queue-b")
    worker = Thread(target=engine.run)
    worker.start()
    try:
        a.done.wait(10)
        b.done.wait(10)
        report["health"] = json.load(call("/health"))
        response = call("/generate", dict(prompt="Count from 1 to 1000.", max_new_tokens=256))
        stream = events(response)
        accepted = next(stream)
        report["cancel_api"] = json.load(call("/requests/"+accepted["request_id"], method="DELETE"))
        report["cancel_stream"] = list(stream)[-1]
        response.close()
        with call("/generate", dict(prompt="Count from 1 to 1000.", max_new_tokens=64, timeout_s=.001)) as response:
            report["timeout_stream"] = list(events(response))[-1]
        def complete(i):
            try:
                with call("/generate", dict(prompt=f"What is {i}+2? Return only a number.", max_new_tokens=8)) as response:
                    return list(events(response))
            except urllib.error.HTTPError as error:
                return [dict(event="rejected", status=error.code)]
        with ThreadPoolExecutor(max_workers=4) as pool:
            report["concurrent"] = list(pool.map(complete, range(8)))
        # Disconnect during generation, without sending DELETE.
        response = call("/generate", dict(prompt="Count from 1 to 1000.", max_new_tokens=256))
        abandoned_id = next(events(response))["request_id"]
        response.close()
        deadline = time.perf_counter()+10
        while abandoned_id in engine.jobs and time.perf_counter() < deadline:
            time.sleep(.02)
        report["disconnect_retired"] = abandoned_id not in engine.jobs
    finally:
        engine.stop.set()
        engine.wake.set()
        worker.join(30)
        server.shutdown()
        server.server_close()
        server_thread.join()
    report["engine_error"] = engine.failed
    report["free_blocks_after"] = len(decoder.blocks.free_block_ids)
    save("http.json", report)
    assert report["queue_full_status"] == 429 and report["cancel_stream"]["reason"] == "cancelled"
    assert report["timeout_stream"]["reason"] == "timeout" and report["disconnect_retired"]
    assert not engine.failed and not worker.is_alive() and len(decoder.blocks.free_block_ids) == decoder.num_blocks
    print("HTTP_COMPLETE", flush=True)


def soak(reference, decoder, duration=120):
    engine = ServingEngine(reference, decoder, queue_limit=16)
    ids = {n: token_prompt(reference, n) for n in (128, 512, 1024, 2048)}
    rng = random.Random(20261018)
    jobs, samples, rejected, cancelled = [], [], 0, 0
    start, next_due, counter = time.perf_counter(), 0., 0
    gc.collect()
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated()
    while time.perf_counter()-start < duration or engine.jobs:
        elapsed = time.perf_counter()-start
        while next_due < duration and next_due <= elapsed:
            try:
                job = engine.submit(Request(str(counter), ids[rng.choice(list(ids))], rng.choice([16,32,64,128]), True),
                    timeout_s=.05 if counter % 11 == 0 else 30)
                if counter % 7 == 0:
                    job.cancel.set()
                    cancelled += 1
                jobs.append(job)
            except OverflowError:
                rejected += 1
            counter += 1
            next_due += rng.expovariate(5)
        engine.tick()
        if not samples or time.perf_counter()-start-samples[-1]["s"] >= 5:
            samples.append(dict(s=time.perf_counter()-start, allocated=torch.cuda.memory_allocated(),
                reserved=torch.cuda.memory_reserved(), pending=len(engine.pending), active=len(engine.active)))
        if not engine.jobs:
            time.sleep(.005)
    gc.collect()
    report = dict(duration_s=time.perf_counter()-start, offered=counter, rejected=rejected, cancel_requested=cancelled,
        reasons={reason: sum(j.reason == reason for j in jobs) for reason in ("length", "cancelled", "timeout", "error")},
        samples=samples, baseline_allocated=baseline, final_allocated=torch.cuda.memory_allocated(),
        free_blocks_after=len(decoder.blocks.free_block_ids), summary=summarize(jobs, time.perf_counter()-start),
        jobs=[job_record(j) for j in jobs])
    save("soak.json", report)
    assert engine.reserved == 0 and len(decoder.blocks.free_block_ids) == decoder.num_blocks
    assert report["reasons"]["error"] == 0
    print("SOAK_COMPLETE", counter, flush=True)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--section", choices=["all", "load", "http", "soak"], default="all")
    args = p.parse_args()
    reference = OptimizedC2C()
    decoder = PagedDecoder(kv_pool_mib=2048, graph=True)
    try:
        warm = reference.prefill(token_prompt(reference, 512))
        seq = decoder.import_cache(warm)
        decoder.step([seq]).argmax(-1).tolist()
        decoder.release(seq)
        del warm, seq
        if args.section in ("all", "load"):
            replay(reference, decoder)
        decoder.fast_inputs = False
        if args.section in ("all", "http"):
            http_checks(reference, decoder)
        if args.section in ("all", "soak"):
            soak(reference, decoder)
    finally:
        decoder.close()


if __name__ == "__main__":
    main()
