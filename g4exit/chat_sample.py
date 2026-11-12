#!/usr/bin/env python3
"""E1: sample chat replies (EN + RU) from a GGUF with llama-server, for reading. One shard of chat_prompts.jsonl."""
import argparse
import json
import os
import subprocess
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--server", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--chat", default="g4exit/chat_prompts.jsonl")
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--out", required=True)
ap.add_argument("--temp", type=float, default=0.7)
ap.add_argument("--top-p", type=float, default=0.8)
ap.add_argument("--top-k", type=int, default=20)
ap.add_argument("--presence", type=float, default=0.0)
a = ap.parse_args()

rows = [json.loads(l) for l in open(a.chat, encoding="utf-8") if l.strip()]
rows = [r for i, r in enumerate(rows) if i % a.nshards == a.shard]
bindir = os.path.dirname(os.path.abspath(a.server))
env = dict(os.environ, LD_LIBRARY_PATH=bindir + ":" + os.environ.get("LD_LIBRARY_PATH", ""))
log = open("server_chat_%d.log" % a.shard, "w")
srv = subprocess.Popen([a.server, "-m", a.model, "--jinja", "-c", "4096", "-t", str(os.cpu_count()), "--port", "8081",
                        "-np", "1", "-ngl", "0"], env=env, stdout=log, stderr=subprocess.STDOUT)


def call(path, body=None, timeout=1200):
    req = urllib.request.Request("http://127.0.0.1:8081" + path,
                                 data=None if body is None else json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


t0 = time.time()
while True:
    try:
        if call("/health", timeout=5).get("status") == "ok":
            break
    except Exception:  # noqa: BLE001
        pass
    if srv.poll() is not None or time.time() - t0 > 900:
        raise SystemExit("server did not come up")
    time.sleep(3)

with open(a.out, "w", encoding="utf-8") as out:
    for r in rows:
        body = {"messages": [{"role": "user", "content": r["prompt"]}], "temperature": a.temp, "top_p": a.top_p,
                "top_k": a.top_k, "presence_penalty": a.presence, "seed": 1234, "max_tokens": 220,
                "chat_template_kwargs": {"enable_thinking": False}}
        try:
            resp = call("/v1/chat/completions", body)
            msg = resp["choices"][0]["message"]
            reply = msg.get("content") or ""
            if msg.get("reasoning_content"):
                reply += " [reasoning] " + msg["reasoning_content"]
            tm = resp.get("timings", {})
        except Exception as e:  # noqa: BLE001
            reply, tm = "[ERR %r]" % (e,), {}
        out.write(json.dumps(dict(id=r["id"], lang=r["lang"], prompt=r["prompt"], reply=reply,
                                  gen_tps=tm.get("predicted_per_second"), pp_tps=tm.get("prompt_per_second"),
                                  n=tm.get("predicted_n")), ensure_ascii=False) + "\n")
        out.flush()
        print(r["id"], tm.get("predicted_per_second"), flush=True)
srv.terminate()
