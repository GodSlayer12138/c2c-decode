"""Loopback HTTP/SSE demo. One model owner; no public deployment defaults."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from threading import Thread
import uuid
from .optimized import OptimizedC2C
from .paged import PagedDecoder
from .scheduler import Request
from .serving import ServingEngine


def make_server(engine, port=8000):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, status, data):
            raw = json.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if self.path != "/health":
                return self.reply(404, {})
            self.reply(503 if engine.failed or engine.stop.is_set() else 200,
                       dict(ready=not engine.failed and not engine.stop.is_set()))

        def do_DELETE(self):
            if not self.path.startswith("/requests/"):
                return self.reply(404, {})
            self.reply(200, dict(cancelled=engine.cancel(self.path.split("/")[-1])))

        def do_POST(self):
            if self.path != "/generate":
                return self.reply(404, {})
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 65536:
                    return self.reply(413, dict(error="Body must be 1..65536 bytes"))
                data = json.loads(self.rfile.read(size))
                prompt = data["prompt"]
                if not isinstance(prompt, str):
                    raise ValueError("prompt must be text")
                request_id = uuid.uuid4().hex
                job = engine.submit(Request(request_id, engine.reference.encode(prompt),
                    data.get("max_new_tokens", 64)), float(data.get("timeout_s", 60)))
            except OverflowError as error:
                return self.reply(429, dict(error=str(error)))
            except RuntimeError as error:
                return self.reply(503, dict(error=str(error)))
            except (ValueError, TypeError, KeyError) as error:
                return self.reply(400, dict(error=str(error)))
            try:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.end_headers()
                self.wfile.write(("data: "+json.dumps(dict(event="accepted", request_id=request_id))+"\n\n").encode())
                self.wfile.flush()
                while True:
                    event = job.events.get()
                    if event["event"] == "token":
                        event["text"] = engine.reference.tokenizer.decode(job.tokens[:event["index"]], skip_special_tokens=True)
                    self.wfile.write(("data: "+json.dumps(event, ensure_ascii=False)+"\n\n").encode())
                    self.wfile.flush()
                    if event["event"] == "done":
                        break
            except (BrokenPipeError, ConnectionResetError):
                job.cancel.set()
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    return server


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--chunk-tokens", type=int, default=0)
    args = p.parse_args()
    reference = OptimizedC2C()
    decoder = PagedDecoder(kv_pool_mib=2048, graph=True)
    engine = ServingEngine(reference, decoder, args.chunk_tokens)
    worker = Thread(target=engine.run)
    worker.start()
    server = make_server(engine, args.port)
    try:
        server.serve_forever()
    finally:
        engine.stop.set()
        engine.wake.set()
        worker.join()
        server.server_close()
        decoder.close()


if __name__ == "__main__":
    main()
