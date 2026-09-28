"""Answer one shard of the MC prompts with Google's LiteRT-LM runtime (the official `litert-lm-api`
Python package, CPU backend) and an unmodified .litertlm model: a fresh conversation per question,
no system prompt, greedy (top_k 1), at most 24 new tokens -- the same budget as the llama.cpp shards
(bench/llamacpp_mc.py) and the on-device harness."""
import argparse
import json
import os
import time

import litert_lm as lm

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--threads", type=int, default=0)
ap.add_argument("--out", required=True)
a = ap.parse_args()

rows = [json.loads(l) for l in open(a.data, encoding="utf-8") if l.strip()]
rows = [r for i, r in enumerate(rows) if i % a.nshards == a.shard]
if a.limit:
    rows = rows[:a.limit]

t0 = time.time()
backend = lm.Backend.CPU(thread_count=a.threads) if a.threads else lm.Backend.CPU()
eng = lm.Engine(a.model, backend=backend, max_num_tokens=2048, cache_dir=os.path.abspath(".lrtcache"))
print("engine ready in %.0fs; %d prompts in shard %d/%d" % (time.time() - t0, len(rows), a.shard, a.nshards), flush=True)
greedy = lm.SamplerConfig(top_k=1, top_p=1.0, temperature=0.0)


def text_of(resp):
    c = resp.get("content", resp) if isinstance(resp, dict) else resp
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "".join(x.get("text", "") for x in c if isinstance(x, dict))
    return json.dumps(resp, ensure_ascii=False)


with open(a.out, "w", encoding="utf-8") as out:
    for n, r in enumerate(rows):
        ts = time.time()
        conv = eng.create_conversation(sampler_config=greedy, max_output_tokens=24)
        try:
            reply = text_of(conv.send_message(r["prompt"]))
        except Exception as e:  # keep going; the scorer counts it as unparsed
            reply = "[ERR %s]" % e
        finally:
            conv.close()
        out.write(json.dumps(dict(key=r["id"] + "|" + r["lang"], id=r["id"], lang=r["lang"], gold=r["gold"],
                                  reply=reply, ms=int(1000 * (time.time() - ts))), ensure_ascii=False) + "\n")
        out.flush()
        if n < 3 or n % 20 == 0:
            print("%d/%d %.1fs %r" % (n + 1, len(rows), time.time() - ts, reply[:60]), flush=True)
print("done in %.0fs" % (time.time() - t0), flush=True)
