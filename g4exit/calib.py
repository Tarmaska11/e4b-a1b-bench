#!/usr/bin/env python3
"""Calibration for the gate-first sparse FFN, a chunk of layers at a time (closed form, cached activations).

For each layer it writes cal_LNN.bin (read by the G4_FFN mode 'c' in patch_gemma4.py):
  c[n]   = rms(u_j) * ||W_down[:, j]||       the per-neuron weight of the gate-only selector
  per kept fraction f: a rank-r linear bypass  y += A2^T (A1^T x)  fitted (ridge) on what the neurons dropped at
  that fraction would have added.
Format: int32 'G4C1', n, d, r, nf; float32 fr[nf]; float32 c[n]; per fraction: f16 A1[d][r], f16 A2[r][d].
"""
import argparse
import json
import os
import random
import struct
import subprocess
import time

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--g4dump", default="")
ap.add_argument("--model", default="")
ap.add_argument("--server", default="", help="llama-server: the full model writes replies to the calibration chats")
ap.add_argument("--bench", default="g4exit/cal_mmlu.jsonl", help="questions that are NOT in the benchmark")
ap.add_argument("--chat", default="g4exit/cal_chat_prompts.jsonl", help="chats that are NOT the evaluation chats")
ap.add_argument("--layers", default="0,1,2,3,4,5,6")
ap.add_argument("--nrec", type=int, default=260)
ap.add_argument("--rank", type=int, default=256)
ap.add_argument("--out", default="cal_out")
ap.add_argument("--selftest", action="store_true")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
T0 = time.time()
LAYERS = [int(x) for x in a.layers.split(",")]
FR = [0.125, 0.25, 0.5]
REPORT = []


def say(*x):
    print("[%5.0fs]" % (time.time() - T0), *x, flush=True)


def rep(s=""):
    REPORT.append(s)
    print(s, flush=True)
    open(os.path.join(a.out, "report_cal_%s.md" % a.layers.replace(",", "_")), "w", encoding="utf-8").write("\n".join(REPORT) + "\n")


def gelu(g):
    return 0.5 * g * (1.0 + np.tanh(0.7978845608028654 * (g + 0.044715 * g * g * g)))


def relerr(Yh, Y):
    return float(((Yh - Y) ** 2).sum() / (Y ** 2).sum())


def ridge(Ftr, Ytr, Fva, Yva):
    A = (Ftr.T @ Ftr).astype(np.float64)
    B = (Ftr.T @ Ytr).astype(np.float64)
    nf = A.shape[0]
    best = None
    for alpha in (1e-4, 1e-3, 1e-2, 1e-1):
        W = np.linalg.solve(A + alpha * np.trace(A) / nf * np.eye(nf), B).astype(np.float32)
        e = relerr(Fva @ W, Yva)
        if best is None or e < best[0]:
            best = (e, W)
    return best[1]


if a.selftest:
    rs_ = np.random.default_rng(0)
    d, n_, N, nL = 128, 512, 3000, len(LAYERS)
    a.rank = 32
    _W = {}

    def weight(name, cols):
        if name not in _W:
            shape = (d, n_) if "down" in name else (n_, d)
            _W[name] = (rs_.standard_normal(shape) / np.sqrt(shape[1])).astype(np.float32)
        return _W[name]

    D = rs_.standard_normal((nL, N, d)).astype(np.float32)
    rec = np.repeat(np.arange(100), N // 100).astype(np.int32)
    first = np.zeros(N, bool)
    first[::N // 100] = True
    nrec = 100
else:
    SEP = "\n<<<G4SEP>>>\n"
    PRE, SUF = "<|turn>user\n", "<turn|>\n<|turn>model\n"
    rows = [json.loads(l) for l in open(a.bench, encoding="utf-8") if l.strip()]
    rows = random.Random(0).sample(rows, min(a.nrec, len(rows)))
    chats = [json.loads(l) for l in open(a.chat, encoding="utf-8") if l.strip()]
    if a.server:
        import urllib.request
        bindir = os.path.dirname(os.path.abspath(a.server))
        senv = dict(os.environ, LD_LIBRARY_PATH=bindir + ":" + os.environ.get("LD_LIBRARY_PATH", ""))
        senv.pop("G4_FFN", None)
        srv = subprocess.Popen([a.server, "-m", a.model, "--jinja", "-c", "4096", "-t", str(os.cpu_count() or 4), "--port", "8082",
                                "-np", "1", "-ngl", "0"], env=senv, stdout=open(os.path.join(a.out, "server.log"), "w"), stderr=subprocess.STDOUT)

        def call(path, body=None, timeout=900):
            req = urllib.request.Request("http://127.0.0.1:8082" + path, data=None if body is None else json.dumps(body).encode("utf-8"),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r_:
                return json.loads(r_.read().decode("utf-8"))

        t_ = time.time()
        while True:
            try:
                if call("/health", timeout=5).get("status") == "ok":
                    break
            except Exception:  # noqa: BLE001
                pass
            assert srv.poll() is None and time.time() - t_ < 900, "server did not come up"
            time.sleep(3)
        for c_ in chats:
            resp = call("/v1/chat/completions", {"messages": [{"role": "user", "content": c_["prompt"]}], "temperature": 0.8, "top_k": 64,
                                                 "top_p": 0.95, "seed": 4321, "max_tokens": 200, "chat_template_kwargs": {"enable_thinking": False}})
            c_["reply"] = resp["choices"][0]["message"].get("content") or ""
        srv.terminate()
        try:
            srv.wait(timeout=60)
        except Exception:  # noqa: BLE001
            srv.kill()
        say("replies written for", len(chats), "calibration chats")
        json.dump(chats, open(os.path.join(a.out, "cal_chats_%s.json" % a.layers.replace(",", "_")), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    texts = [PRE + r["prompt"] + SUF for r in rows] + [PRE + c_["prompt"] + SUF + c_.get("reply", "") for c_ in chats]
    random.Random(1).shuffle(texts)
    rf, ob = os.path.join(a.out, "recs.txt"), os.path.join(a.out, "dump.bin")
    open(rf, "w", encoding="utf-8", newline="\n").write(SEP.join(texts) + SEP + "<<<END>>>")
    env = dict(os.environ, G4_LAYERS=a.layers, G4_NAMES="ffn_norm", G4_OUT=ob, G4_LAST="0")
    env.pop("G4_FFN", None)
    with open(os.path.join(a.out, "dump.log"), "w") as lg:
        rc = subprocess.call([a.g4dump, "-m", a.model, "-f", rf, "--no-escape", "--no-repack", "-c", "2048", "-b", "2048",
                              "-t", str(os.cpu_count() or 4), "-ngl", "0"], env=env, stdout=lg, stderr=subprocess.STDOUT)
    say("dump rc", rc, "size %.2f GB" % (os.path.getsize(ob) / 1e9))
    assert rc == 0
    with open(ob, "rb") as f:
        magic, d, last_only, n_slots = struct.unpack("<4i", f.read(16))
        slot_layers = list(struct.unpack("<%di" % n_slots, f.read(4 * n_slots)))
        assert slot_layers == LAYERS, slot_layers
        hdr0 = f.tell()
        sizes = []
        while True:
            h = f.read(12)
            if len(h) < 12:
                break
            r_, n_tok, n_vec = struct.unpack("<3i", h)
            if n_tok < 0:
                continue
            sizes.append(n_vec)
            f.seek(4 * n_tok + 4 * n_vec * d * n_slots, 1)
        N = sum(sizes)
        say("records", len(sizes), "tokens", N)
        D = np.zeros((n_slots, N, d), np.float32)
        rec = np.zeros(N, np.int32)
        first = np.zeros(N, bool)
        f.seek(hdr0)
        o = 0
        for i, n_vec in enumerate(sizes):
            while True:
                r_, n_tok, nv = struct.unpack("<3i", f.read(12))
                if n_tok >= 0:
                    break
            f.seek(4 * n_tok, 1)
            D[:, o:o + nv, :] = np.frombuffer(f.read(4 * nv * d * n_slots), dtype="<f4").reshape(n_slots, nv, d)
            rec[o:o + nv] = i
            first[o] = True
            o += nv
    os.remove(ob)
    nrec = len(sizes)
    import gguf
    from gguf.quants import dequantize

    rd = gguf.GGUFReader(a.model)
    TENS = {t.name: t for t in rd.tensors}

    def weight(name, cols):
        t = TENS[name]
        return np.array(dequantize(t.data, t.tensor_type), dtype=np.float32).reshape(-1, cols)

keep = ~first
te = keep & (rec >= int(nrec * 0.9))
va = keep & (rec >= int(nrec * 0.8)) & ~te
tr = keep & ~te & ~va
rep("# calibration, layers %s — train %d, validation %d, held-out %d tokens" % (a.layers, int(tr.sum()), int(va.sum()), int(te.sum())))
rep("")
rep("| layer | f | gate-cal, no bypass | with the rank-%d bypass |" % a.rank)
rep("|---:|---:|---:|---:|")

for li, L in enumerate(LAYERS):
    X = D[li]
    Wg, Wu = weight("blk.%d.ffn_gate.weight" % L, d), weight("blk.%d.ffn_up.weight" % L, d)
    Wd = weight("blk.%d.ffn_down.weight" % L, Wg.shape[0])
    n = Wg.shape[0]
    dn = np.linalg.norm(Wd, axis=0)
    GG = np.zeros((N, n), np.float32)
    H = np.zeros((N, n), np.float32)
    usq = np.zeros(n, np.float64)
    for s in range(0, N, 4096):
        e = min(N, s + 4096)
        GG[s:e] = gelu(X[s:e] @ Wg.T)
        U = X[s:e] @ Wu.T
        H[s:e] = GG[s:e] * U
        usq += (U[tr[s:e]].astype(np.float64) ** 2).sum(0)
    c = (np.sqrt(usq / max(1, int(tr.sum()))) * dn).astype(np.float32)
    Y = H @ Wd.T
    Xtr, Xva, Xte = X[tr], X[va], X[te]
    r = min(a.rank, d)
    with open(os.path.join(a.out, "cal_L%02d.bin" % L), "wb") as f:
        f.write(struct.pack("<5i", 0x31433447, n, d, r, len(FR)))
        f.write(np.array(FR, dtype="<f4").tobytes())
        f.write(c.astype("<f4").tobytes())
        for fr in FR:
            k = int(np.ceil(fr * n))
            Yk = np.zeros_like(Y)
            for s in range(0, N, 2048):
                e = min(N, s + 2048)
                sc = np.abs(GG[s:e]) * c
                thr = -np.partition(-sc, k - 1, axis=1)[:, k - 1:k]
                Yk[s:e] = (H[s:e] * (sc >= thr)) @ Wd.T
            E = Y - Yk
            W = ridge(Xtr, E[tr], Xva, E[va])                      # [d, d]
            _, _, Vt = np.linalg.svd(Xtr[:4000] @ W, full_matrices=False)
            V = Vt[:r].T                                            # [d, r]
            A1, A2 = (W @ V).astype(np.float16), V.T.astype(np.float16)
            e0 = relerr(Yk[te], Y[te])
            e1 = relerr(Yk[te] + (Xte @ A1.astype(np.float32)) @ A2.astype(np.float32), Y[te])
            rep("| %d | %.3f | %.4f | %.4f |" % (L, fr, e0, e1))
            f.write(A1.astype("<f2").tobytes())
            f.write(A2.astype("<f2").tobytes())
    say("layer", L, "written")
    del GG, H, Y
say("done")
