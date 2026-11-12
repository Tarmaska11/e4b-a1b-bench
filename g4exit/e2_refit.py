#!/usr/bin/env python3
"""E2: can E4B's FFN be RE-BUILT (closed form, from cached activations) into something sparse that selection
alone cannot reach?  Per layer, on held-out tokens, relative error of the FFN output for:

  oracle            per-token top-k neurons by contribution norm, original weights (the old campaign's best arm)
  oracle+bypass     the same, plus ONE linear map x -> (what the dropped neurons would have added)
  gate-sel+bypass   neurons chosen from the gate projection only (a selector an engine can afford), plus bypass
  linear            a linear map alone (zero neurons)
  static+refit      one fixed neuron subset, output matrix re-solved by least squares, plus a linear term
  8 regions+refit   k-means regions of the input; each region has its own subset and re-solved output matrix

rel. error = sum ||y_hat - y||^2 / sum ||y||^2 over held-out tokens (y = the dense FFN output).
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
ap.add_argument("--g4dump", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--bench", default="data/bench_gmmlu.jsonl")
ap.add_argument("--chat", default="g4exit/chat_prompts.jsonl")
ap.add_argument("--layers", default="6,16,34")
ap.add_argument("--nrec", type=int, default=200)
ap.add_argument("--out", default="e2_out")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
T0 = time.time()
LAYERS = [int(x) for x in a.layers.split(",")]
REPORT, RES = [], {}


def say(*x):
    print("[%5.0fs]" % (time.time() - T0), *x, flush=True)


def rep(s=""):
    REPORT.append(s)
    print(s, flush=True)
    open(os.path.join(a.out, "report_e2.md"), "w", encoding="utf-8").write("\n".join(REPORT) + "\n")
    json.dump(RES, open(os.path.join(a.out, "results_e2.json"), "w"), indent=1)


# ------------------------------------------------------------------ dump
SEP = "\n<<<G4SEP>>>\n"
PRE, SUF = "<|turn>user\n", "<turn|>\n<|turn>model\n"
rows = [json.loads(l) for l in open(a.bench, encoding="utf-8") if l.strip()]
rows = random.Random(0).sample(rows, min(a.nrec, len(rows)))
texts = [PRE + r["prompt"] + SUF for r in rows] + [PRE + json.loads(l)["prompt"] + SUF for l in open(a.chat, encoding="utf-8") if l.strip()]
random.Random(1).shuffle(texts)
rf, ob = os.path.join(a.out, "recs.txt"), os.path.join(a.out, "dump.bin")
open(rf, "w", encoding="utf-8", newline="\n").write(SEP.join(texts) + SEP + "<<<END>>>")
env = dict(os.environ, G4_LAYERS=a.layers, G4_NAMES="ffn_norm,ffn_out", G4_OUT=ob, G4_LAST="0")
with open(os.path.join(a.out, "dump.log"), "w") as lg:
    rc = subprocess.call([a.g4dump, "-m", a.model, "-f", rf, "--no-escape", "-c", "2048", "-b", "2048",
                          "-t", str(os.cpu_count() or 4), "-ngl", "0"], env=env, stdout=lg, stderr=subprocess.STDOUT)
say("dump rc", rc, "size %.2f GB" % (os.path.getsize(ob) / 1e9))
assert rc == 0

# ------------------------------------------------------------------ load (two passes, preallocated)
with open(ob, "rb") as f:
    magic, d, last_only, n_slots = struct.unpack("<4i", f.read(16))
    slot_layers = list(struct.unpack("<%di" % n_slots, f.read(4 * n_slots)))
    hdr = f.tell()
    sizes = []
    while True:
        h = f.read(12)
        if len(h) < 12:
            break
        r, n_tok, n_vec = struct.unpack("<3i", h)
        if n_tok < 0:
            continue
        sizes.append(n_vec)
        f.seek(4 * n_tok + 4 * n_vec * d * n_slots, 1)
    N = sum(sizes)
    say("records", len(sizes), "tokens", N, "slots", slot_layers)
    D = np.zeros((n_slots, N, d), np.float32)
    rec = np.zeros(N, np.int32)
    first = np.zeros(N, bool)
    f.seek(hdr)
    o = 0
    for i, n_vec in enumerate(sizes):
        while True:
            r, n_tok, nv = struct.unpack("<3i", f.read(12))
            if n_tok >= 0:
                break
        f.seek(4 * n_tok, 1)
        D[:, o:o + nv, :] = np.frombuffer(f.read(4 * nv * d * n_slots), dtype="<f4").reshape(n_slots, nv, d)
        rec[o:o + nv] = i
        first[o] = True
        o += nv
os.remove(ob)
nL = len(LAYERS)
nrec = len(sizes)
keep = ~first  # the BOS position is an outlier
te = keep & (rec >= int(nrec * 0.85))
va = keep & (rec >= int(nrec * 0.75)) & ~te
tr = keep & ~te & ~va
say("train/val/test tokens", int(tr.sum()), int(va.sum()), int(te.sum()))

import gguf  # noqa: E402
from gguf.quants import dequantize  # noqa: E402

rd = gguf.GGUFReader(a.model)
TENS = {t.name: t for t in rd.tensors}


def weight(name, cols):
    t = TENS[name]
    return np.array(dequantize(t.data, t.tensor_type), dtype=np.float32).reshape(-1, cols)


def gelu(g):
    return 0.5 * g * (1.0 + np.tanh(0.7978845608028654 * (g + 0.044715 * g * g * g)))


def relerr(Yh, Y):
    return float(((Yh - Y) ** 2).sum() / (Y ** 2).sum())


def ridge_eval(Ftr, Ytr, Fva, Yva, Fte, Yte):
    """least squares F -> Y with ridge chosen on the validation tokens; returns the prediction on test"""
    A = (Ftr.T @ Ftr).astype(np.float64)
    B = (Ftr.T @ Ytr).astype(np.float64)
    nf = A.shape[0]
    best = None
    for alpha in (1e-4, 1e-3, 1e-2, 1e-1):
        W = np.linalg.solve(A + alpha * np.trace(A) / nf * np.eye(nf), B).astype(np.float32)
        e = relerr(Fva @ W, Yva)
        if best is None or e < best[0]:
            best = (e, W)
    return Fte @ best[1]


rep("# E2 — re-building E4B's FFN in closed form (Q4_K_M weights, cached activations, no gradient training)")
rep("")
rep("tokens: train %d, validation %d, held-out test %d (split by prompt). Lower is better; 0 = identical to the dense FFN."
    % (int(tr.sum()), int(va.sum()), int(te.sum())))
BUD = [0.0625, 0.125, 0.25, 0.5]

for li, L in enumerate(LAYERS):
    try:
        X, Yd = D[li], D[nL + li]
        Wg, Wu = weight("blk.%d.ffn_gate.weight" % L, d), weight("blk.%d.ffn_up.weight" % L, d)
        Wd = weight("blk.%d.ffn_down.weight" % L, Wg.shape[0])
        n = Wg.shape[0]
        H = np.zeros((N, n), np.float32)
        Gs = np.zeros((N, n), np.float32)  # gate-only score
        un = np.linalg.norm(Wu, axis=1)
        dn = np.linalg.norm(Wd, axis=0)
        for s in range(0, N, 4096):
            g = gelu(X[s:s + 4096] @ Wg.T)
            H[s:s + 4096] = g * (X[s:s + 4096] @ Wu.T)
            Gs[s:s + 4096] = np.abs(g) * (un * dn)
        Y = H @ Wd.T
        chk = relerr(Y[keep], Yd[keep])
        rep("")
        rep("## layer %d  (%d neurons)" % (L, n))
        rep("")
        rep("instrument check — numpy FFN vs the engine's own FFN output: rel. error %.2e" % chk)
        S = np.abs(H) * dn
        srt = -np.sort(-S[te][:2000], axis=1)
        cs = np.cumsum(srt, axis=1) / srt.sum(1, keepdims=True)
        rep("share of total contribution mass carried by the top 6.25 / 12.5 / 25 / 50 %% of neurons: %s"
            % " / ".join("%.1f %%" % (100 * cs[:, int(b * n) - 1].mean()) for b in BUD))
        RES["L%d" % L] = R = dict(check=chk)
        Xtr, Xva, Xte, Ytr, Yva, Yte = X[tr], X[va], X[te], Y[tr], Y[va], Y[te]
        R["linear"] = relerr(ridge_eval(Xtr, Ytr, Xva, Yva, Xte, Yte), Yte)
        rep("")
        rep("| neurons kept per token | oracle | oracle + bypass | gate-selected + bypass | static + refit | 8 regions + refit |")
        rep("|---:|---:|---:|---:|---:|---:|")
        rep("| 0 (a linear map alone) | | %.4f | | | |" % R["linear"])
        # regions
        Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
        rng = np.random.default_rng(0)
        C = Xn[tr][rng.choice(int(tr.sum()), 8, replace=False)]
        for _ in range(15):
            asg = (Xn[tr] @ C.T).argmax(1)
            for c in range(8):
                if (asg == c).sum() > 0:
                    v = Xn[tr][asg == c].mean(0)
                    C[c] = v / np.linalg.norm(v)
        reg = (Xn @ C.T).argmax(1)
        del Xn
        for b in BUD:
            k = int(b * n)
            row = {}

            def masked(score):
                Yk = np.zeros_like(Y)
                for s in range(0, N, 4096):
                    sc = score[s:s + 4096]
                    thr = -np.partition(-sc, k - 1, axis=1)[:, k - 1:k]
                    Yk[s:s + 4096] = (H[s:s + 4096] * (sc >= thr)) @ Wd.T
                return Yk

            Yo = masked(S)
            row["oracle"] = relerr(Yo[te], Yte)
            row["oracle_bypass"] = relerr(Yo[te] + ridge_eval(Xtr, Ytr - Yo[tr], Xva, Yva - Yo[va], Xte, Yte - Yo[te]), Yte)
            Yg = masked(Gs)
            row["gate_bypass"] = relerr(Yg[te] + ridge_eval(Xtr, Ytr - Yg[tr], Xva, Yva - Yg[va], Xte, Yte - Yg[te]), Yte)
            del Yo, Yg
            if b <= 0.25:
                idx = np.argsort(-S[tr].mean(0))[:k]
                F = np.concatenate([H[:, idx], X], axis=1)
                row["static_refit"] = relerr(ridge_eval(F[tr], Ytr, F[va], Yva, F[te], Yte), Yte)
                del F
            if b <= 0.125:
                Yh = np.zeros_like(Yte)
                for c in range(8):
                    mtr, mva, mte = tr & (reg == c), va & (reg == c), te & (reg == c)
                    if mte.sum() == 0:
                        continue
                    if mtr.sum() < 50 or mva.sum() < 5:
                        Yh[reg[te] == c] = 0
                        continue
                    idx = np.argsort(-S[mtr].mean(0))[:k]
                    F = np.concatenate([H[:, idx], X], axis=1)
                    Yh[reg[te] == c] = ridge_eval(F[mtr], Y[mtr], F[mva], Y[mva], F[mte], Y[mte])
                    del F
                row["regions_refit"] = relerr(Yh, Yte)
            R["b%g" % b] = row
            rep("| %.2f %% | %.4f | %.4f | %.4f | %s | %s |" % (100 * b, row["oracle"], row["oracle_bypass"], row["gate_bypass"],
                                                             "%.4f" % row["static_refit"] if "static_refit" in row else "",
                                                             "%.4f" % row["regions_refit"] if "regions_refit" in row else ""))
            say("layer", L, "budget", b, row)
        del H, Gs, S, Y
    except Exception:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        rep("layer %d FAILED: see log" % L)
say("done")
