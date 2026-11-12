#!/usr/bin/env python3
"""E0 collector, one shard: sample chat replies with the full model, then dump residuals for every arm.

mode e4b: chat (every token) + benchmark (last token) for the full pass and each G4_SKIP arm.
mode e2b: benchmark only, last layer only (the control).
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g4bin  # noqa: E402

SEP = "\n<<<G4SEP>>>\n"

ap = argparse.ArgumentParser()
ap.add_argument("--server", required=True)
ap.add_argument("--g4dump", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--mode", default="e4b")
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--bench", default="data/bench_gmmlu.jsonl")
ap.add_argument("--chat", default="g4exit/chat_prompts.jsonl")
ap.add_argument("--out", required=True)
ap.add_argument("--limit", type=int, default=0, help="smoke: keep only this many records of each kind")
a = ap.parse_args()

os.makedirs(a.out, exist_ok=True)
NCPU = os.cpu_count() or 4
T0 = time.time()


def log(*x):
    print("[%6.0fs]" % (time.time() - T0), *x, flush=True)


def call(path, body=None, timeout=900):
    req = urllib.request.Request("http://127.0.0.1:8080" + path,
                                 data=None if body is None else json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def load(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    rows = [r for i, r in enumerate(rows) if i % a.nshards == a.shard]
    return rows[:a.limit] if a.limit else rows


bench = load(a.bench)
chat = load(a.chat) if a.mode == "e4b" else []
log("shard %d/%d: %d bench, %d chat" % (a.shard, a.nshards, len(bench), len(chat)))

# ---------------------------------------------------------------- server: template, letters, replies
bindir = os.path.dirname(os.path.abspath(a.server))
env = dict(os.environ, LD_LIBRARY_PATH=bindir + ":" + os.environ.get("LD_LIBRARY_PATH", ""))
slog = open(os.path.join(a.out, "server.log"), "w")
srv = subprocess.Popen([a.server, "-m", a.model, "--jinja", "-c", "4096", "-t", str(NCPU), "--port", "8080",
                        "-np", "1", "-ngl", "0"], env=env, stdout=slog, stderr=subprocess.STDOUT)
t = time.time()
while True:
    try:
        if call("/health", timeout=5).get("status") == "ok":
            break
    except Exception:
        pass
    if srv.poll() is not None or time.time() - t > 900:
        raise SystemExit("server did not come up")
    time.sleep(3)
log("server up")

meta = dict(mode=a.mode, shard=a.shard, nshards=a.nshards, model=os.path.basename(a.model))
MARK = "@@USER@@"
try:
    tpl = call("/apply-template", {"messages": [{"role": "user", "content": MARK}],
                                   "chat_template_kwargs": {"enable_thinking": False}})["prompt"]
    prefix, suffix = tpl.split(MARK)
    meta["template_source"] = "server"
except Exception as e:  # noqa: BLE001
    log("apply-template failed (%r); using the documented template" % (e,))
    prefix, suffix = "<bos><|turn>user\n", "<turn|>\n<|turn>model\n"
    meta["template_source"] = "manual"
meta["template_raw"] = [prefix, suffix]
if prefix.startswith("<bos>"):
    prefix = prefix[len("<bos>"):]  # g4dump adds BOS itself
meta["prefix"], meta["suffix"] = prefix, suffix
log("template:", json.dumps([prefix, suffix]))


def tok(text, special=False):
    try:
        return call("/tokenize", {"content": text, "add_special": special, "parse_special": True})["tokens"]
    except Exception as e:  # noqa: BLE001
        log("tokenize failed", repr(e))
        return None


meta["letters"] = {c: tok(c) for c in "ABCD"}
meta["suffix_tokens"] = tok(suffix)
log("letters:", meta["letters"])

for r in chat:
    body = {"messages": [{"role": "user", "content": r["prompt"]}], "temperature": 0.8, "top_k": 64, "top_p": 0.95,
            "seed": 1234, "max_tokens": 160, "chat_template_kwargs": {"enable_thinking": False}}
    t = time.time()
    resp = call("/v1/chat/completions", body)
    r["reply"] = resp["choices"][0]["message"].get("content") or ""
    r["gen_tokens"] = resp.get("usage", {}).get("completion_tokens")
    r["gen_s"] = round(time.time() - t, 1)
    pt = tok(prefix + r["prompt"] + suffix, special=True)
    r["n_prompt"] = len(pt) if pt is not None else None
    log("chat %s: %s tokens in %.0fs" % (r["id"], r["gen_tokens"], r["gen_s"]))
srv.terminate()
try:
    srv.wait(timeout=60)
except Exception:  # noqa: BLE001
    srv.kill()

meta["chat"] = chat
meta["bench"] = [dict(id=r["id"], lang=r["lang"], gold=r["gold"]) for r in bench]

chat_recs = [prefix + r["prompt"] + suffix + r["reply"] for r in chat]
bench_recs = [prefix + r["prompt"] + suffix for r in bench]


# ---------------------------------------------------------------- dumps
def dump(name, recs, layers, last, skip=None, ffn=None, desc=""):
    if not recs:
        return None
    for x in recs:
        assert SEP not in x
    rf = os.path.join(a.out, name + ".txt")
    with open(rf, "w", encoding="utf-8", newline="\n") as f:
        f.write(SEP.join(recs) + SEP + "<<<END>>>")
    ob = os.path.join(a.out, name + ".bin")
    e = dict(os.environ, G4_LAYERS=",".join(str(l) for l in layers), G4_OUT=ob, G4_LAST="1" if last else "0")
    if skip:
        e["G4_SKIP"] = skip
    if ffn:
        e["G4_FFN"] = ffn
    t = time.time()
    with open(os.path.join(a.out, name + ".log"), "w") as lg:
        rc = subprocess.call([a.g4dump, "-m", a.model, "-f", rf, "--no-escape", "--no-repack", "-c", "2048", "-b", "2048",
                              "-t", str(NCPU), "-ngl", "0"], env=e, stdout=lg, stderr=subprocess.STDOUT)
    dt = time.time() - t
    if rc != 0:
        log("DUMP FAILED %s rc=%d" % (name, rc))
        os.system("tail -5 %s" % os.path.join(a.out, name + ".log"))
        return dict(name=name, rc=rc, s=round(dt), desc=desc)
    d = g4bin.parse_bin(ob)
    g4bin.save_npz(d, os.path.join(a.out, name + ".npz"))
    os.remove(ob)
    os.remove(rf)
    ntok = int(sum(len(x) for x in d["tokens"]))
    log("dump %-22s %4d recs %7d tok %5.0fs %6.1f tok/s skipped=%d" % (name, len(d["rec_idx"]), ntok, dt, ntok / max(dt, 1e-9),
                                                                    len(d["skipped"])))
    import re
    kept = {int(m.group(1)): [int(m.group(2)), int(m.group(3))] for m in
            re.finditer(r"G4FFN_STAT il=(\d+) kept=(\d+) total=(\d+)", open(os.path.join(a.out, name + ".log"), errors="replace").read())}
    if kept:
        log("   FFN kept %.2f %% over %d layers" % (100.0 * sum(v[0] for v in kept.values()) / max(1, sum(v[1] for v in kept.values())), len(kept)))
    return dict(name=name, rc=0, s=round(dt), recs=len(d["rec_idx"]), tokens=ntok, skipped=d["skipped"], layers=layers,
                skip=skip, ffn=ffn, ffn_kept=kept, desc=desc, last=last)


arms = []
if a.mode == "e2b":
    arms.append(dump("bench_full", bench_recs, [34], True))
else:
    FULL = [23, 26, 29, 32, 35, 38, 40, 41]
    arms.append(dump("chat_full", chat_recs, FULL, False))
    arms.append(dump("bench_full", bench_recs, FULL, True))
    # (name, G4_SKIP, G4_FFN, description). Every arm reads the final residual (layer 41).
    ARMS = [("s36_40", "36:40", None, "38 layers (drop 36-39)"),
            ("s30_40", "30:40", None, "32 layers (drop 30-39)"),
            ("s24_40", "24:40", None, "26 layers (drop 24-39)"),
            ("f_k0_25", None, "k:0:42:0.25", "INSTRUMENT CHECK: oracle 25 %, all 42 layers (old result: about -4 points)"),
            ("f_k0_50", None, "k:0:42:0.5", "oracle 50 % of FFN neurons, all layers"),
            ("f_k24_125", None, "k:24:42:0.125", "oracle 12.5 % in the stateless layers 24-41"),
            ("f_k24_25", None, "k:24:42:0.25", "oracle 25 % in the stateless layers 24-41"),
            ("f_t0_02", None, "t:0:42:0.02", "oracle, bounded: drop <= 2 % of contribution mass, all layers"),
            ("f_t0_005", None, "t:0:42:0.005", "oracle, bounded: drop <= 0.5 % of contribution mass, all layers"),
            ("f_g0_50", None, "g:0:42:0.5", "GATE-ONLY selector, keep 50 %, all layers"),
            ("f_g0_25", None, "g:0:42:0.25", "GATE-ONLY selector, keep 25 %, all layers"),
            ("f_G0_02", None, "G:0:42:0.02", "GATE-ONLY selector, drop <= 2 % of gate-score mass, all layers"),
            ("f_g_mix", None, "g:0:24:0.5,g:24:42:0.25", "GATE-ONLY selector, 50 % in layers 0-23, 25 % in 24-41")]
    if a.limit:
        ARMS.append(("f_ctrl", None, "k:0:42:1.0", "control: sparse code path with every neuron kept (must equal full)"))
    for name, spec, ffn, desc in ARMS:
        arms.append(dump("chat_" + name, chat_recs, [41], False, spec, ffn, desc))
        arms.append(dump("bench_" + name, bench_recs, [41], True, spec, ffn, desc))
meta["arms"] = arms
json.dump(meta, open(os.path.join(a.out, "meta.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
log("done")
