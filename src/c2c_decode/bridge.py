"""Local CUDA IPC bridge; keeps official C2C in its pinned Transformers env."""
import argparse
import gc
import os
from pathlib import Path
import pickle
from multiprocessing.connection import Client, Listener
from multiprocessing.reduction import ForkingPickler
import subprocess
import tempfile
import time
from types import SimpleNamespace
import torch
import torch.multiprocessing  # Register CUDA tensor reducers.


class CacheView(list):
    def get_seq_length(self):
        return self[0][0].shape[2]


def send(connection, value):
    connection.send_bytes(ForkingPickler.dumps(value))


def receive(connection):
    return pickle.loads(connection.recv_bytes())


class OfficialBridge:
    def __init__(self, python, models="models.lock.json"):
        Path(".cache").mkdir(exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(prefix="c2c-ipc-", dir=".cache")
        address = str(Path(self.directory.name, "socket").resolve())
        self.log = open(Path(self.directory.name, "producer.log"), "w+")
        self.process = subprocess.Popen([python, "-u", "-m", "c2c_decode.bridge",
            "--serve", address, "--models", models], stdout=self.log, stderr=subprocess.STDOUT)
        deadline = time.monotonic()+180
        while True:
            if self.process.poll() is not None or time.monotonic() > deadline:
                self.log.seek(0)
                raise RuntimeError("Official producer failed: " + self.log.read())
            try:
                self.connection = Client(address, family="AF_UNIX")
                break
            except (FileNotFoundError, ConnectionRefusedError):
                time.sleep(.1)
        self.info = receive(self.connection)

    def call(self, command, **kwargs):
        send(self.connection, dict(command=command, **kwargs))
        result = receive(self.connection)
        if isinstance(result, dict) and "error" in result:
            raise RuntimeError(result["error"])
        return result

    def prefill(self, length=None, prompt=None, mode="official"):
        start = time.perf_counter()
        value = self.call("prefill", length=length, prompt=prompt, mode=mode)
        value["cache"] = CacheView(value["cache"])
        value["bridge_prefill_ms"] = (time.perf_counter()-start)*1000
        return SimpleNamespace(**value)

    def release(self, key):
        # Caller must drop all imported tensor references before this handshake.
        torch.cuda.synchronize()
        gc.collect()
        self.call("release", key=key)

    def close(self):
        self.call("close")
        self.connection.close()
        self.process.wait(timeout=30)
        self.log.close()
        self.directory.cleanup()


@torch.inference_mode()
def serve(address, models):
    from .reference import OfficialC2C
    from .optimized import OptimizedC2C
    from .evaluate import token_prompt, metadata
    from transformers import DynamicCache
    reference = OptimizedC2C(models)
    owned = {}
    counter = 0
    with Listener(address, family="AF_UNIX") as listener:
        os.chmod(address, 0o600)
        with listener.accept() as connection:
            send(connection, metadata(SimpleNamespace(models=models)))
            while True:
                request = receive(connection)
                command = request["command"]
                try:
                    if command == "close":
                        send(connection, True)
                        break
                    if command == "release":
                        del owned[request["key"]]
                        gc.collect()
                        torch.cuda.ipc_collect()
                        send(connection, True)
                        continue
                    if command == "prefill":
                        ids = (reference.encode(request["prompt"]) if request["prompt"] is not None
                               else token_prompt(reference, request["length"]))
                        mode = request.get("mode", "official")
                        if mode == "official":
                            handoff = OfficialC2C.prefill(reference, ids)
                        elif mode == "lean":
                            handoff = reference.prefill(ids)
                        elif mode.startswith("chunk-"):
                            for handoff in reference.prefill_steps(ids, int(mode.split("-")[1])):
                                if handoff is not None:
                                    break
                        else:
                            raise ValueError("Unknown producer mode")
                        key = str(counter)
                        counter += 1
                        owned[key] = {"handoff": handoff}
                        value = dict(key=key, prompt_ids=ids, first_token=handoff.first_token,
                            cache=list(handoff.cache), timings=handoff.timings)
                        send(connection, value)
                        del value, handoff
                    elif command == "history":
                        handoff = owned[request["key"]]["handoff"]
                        cache = DynamicCache.from_legacy_cache(tuple(
                            (k.clone(), v.clone()) for k, v in handoff.cache))
                        history = [handoff.first_token]
                        logits_list = []
                        for i in range(request["count"]-1):
                            logits, cache = reference.decode(history[-1], cache)
                            history.append(int(logits.argmax(-1).item()))
                            if i < request.get("check_steps", 0):
                                logits_list.append(logits.clone())
                        owned[request["key"]]["logits"] = logits_list
                        torch.cuda.synchronize()
                        send(connection, dict(history=history, logits=logits_list))
                        del handoff, cache, logits_list, logits
                    else:
                        raise ValueError(command)
                except Exception:
                    import traceback
                    send(connection, {"error": traceback.format_exc()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", required=True)
    parser.add_argument("--models", default="models.lock.json")
    args = parser.parse_args()
    serve(args.serve, args.models)
