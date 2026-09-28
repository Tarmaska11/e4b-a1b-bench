#!/usr/bin/env python3
"""Answer one shard of the MC prompts with a llama.cpp server (greedy, no thinking, 24 tokens max)."""
import argparse
import json
import os
import subprocess
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--server", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--out", required=True)
a = ap.parse_args()

rows = [json.loads(l) for l in open(a.data, encoding="utf-8") if l.strip()]
rows = [r for i, r in enumerate(rows) if i % a.nshards == a.shard]
bindir = os.path.dirname(os.path.abspath(a.server))
env = dict(os.environ, LD_LIBRARY_PATH=bindir + ":" + os.environ.get("LD_LIBRARY_PATH", ""))
log = open("server_%d.log" % a.shard, "w")
srv = subprocess.Popen([a.server, "-m", a.model, "--jinja", "-c", "4096", "-t", str(os.cpu_count()),
                        "--port", "8080", "-np", "1", "-ngl", "0"], env=env, stdout=log, stderr=subprocess.STDOUT)


def call(path, body=None, timeout=600):
    req = urllib.request.Request("http://127.0.0.1:8080" + path,
                                 data=None if body is None else json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


t0 = time.time()
while True:
    try:
        if call("/health", timeout=5).get("status") == "ok":
            break
    except Exception:
        pass
    if srv.poll() is not None or time.time() - t0 > 900:
        raise SystemExit("server did not come up; see server_%d.log" % a.shard)
    time.sleep(3)
print("server up in %.0fs; %d prompts" % (time.time() - t0, len(rows)), flush=True)

with open(a.out, "w", encoding="utf-8") as out:
    for i, r in enumerate(rows):
        body = {"messages": [{"role": "user", "content": r["prompt"]}], "temperature": 0, "top_k": 1,
                "max_tokens": 24, "chat_template_kwargs": {"enable_thinking": False}}
        t = time.time()
        try:
            resp = call("/v1/chat/completions", body)
            msg = resp["choices"][0]["message"]
            reply = (msg.get("content") or "") + ("" if not msg.get("reasoning_content") else
                                                  " [reasoning] " + msg["reasoning_content"])
            usage = resp.get("usage", {})
        except Exception as e:
            reply, usage = "[ERR %r]" % e, {}
        out.write(json.dumps(dict(key=r["id"] + "|" + r["lang"], id=r["id"], lang=r["lang"], gold=r["gold"],
                                  reply=reply, ms=int((time.time() - t) * 1000),
                                  prompt_tokens=usage.get("prompt_tokens")), ensure_ascii=False) + "\n")
        out.flush()
        if i % 20 == 0:
            print("%d/%d %.0fs" % (i, len(rows), time.time() - t0), flush=True)
srv.terminate()
print("done in %.0fs" % (time.time() - t0))
