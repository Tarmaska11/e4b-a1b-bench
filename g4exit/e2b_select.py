#!/usr/bin/env python3
"""E2b: selector and bypass design for a sparse E4B FFN, per layer, on cached activations (no gradient training).

Metric everywhere: sum ||y_hat - y||^2 / sum ||y||^2 on held-out tokens, y = the dense Q4 FFN output.
Yardstick: the same metric between the bf16 ORIGINAL weights and the Q4_K_M weights (the quantisation noise that
the benchmark already shows to be free).

Columns (b = fraction of neurons computed exactly):
  oracle            top-b by true contribution |h_j| ||d_j||, rest dropped
  +mean             ... rest replaced by a constant vector
  +lin256           ... rest replaced by a rank-256 linear map of x
  +lin              ... rest replaced by a full linear map of x
  gate              selection from the gate only: |gelu(g_j)| ||up_j|| ||d_j||, + full linear bypass
  gate-cal          the same with ||up_j|| replaced by the measured RMS of u_j, + bypass
  gate+u128/u512    gate exact, u estimated from a rank-128 / rank-512 factorisation of W_up, + bypass
  predict-correct   every neuron gets a LINEAR prediction h^_j(x); the top-b by |h_j - h^_j| ||d_j|| are corrected
                    exactly (oracle choice). full = full-rank predictor, r256 = rank-256 predictor
  pc-real128/512    predict-correct with an affordable selector: gate exact, u from the rank-r factorisation,
                    predictor rank 256
"""
import argparse
import json
import os
import random
import struct
import subprocess
import time
import urllib.request

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--g4dump", default="")
ap.add_argument("--model", default="")
ap.add_argument("--bench", default="data/bench_gmmlu.jsonl")
ap.add_argument("--chat", default="g4exit/chat_prompts.jsonl")
ap.add_argument("--layers", default="6,16,34")
ap.add_argument("--nrec", type=int, default=200)
ap.add_argument("--out", default="e2b_out")
ap.add_argument("--hf", default="google/gemma-4-E4B-it")
ap.add_argument("--selftest", action="store_true", help="tiny random model, no dump: checks the analysis code")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
T0 = time.time()
LAYERS = [int(x) for x in a.layers.split(",")]
TAG = a.layers.replace(",", "_")
REPORT, RES = [], {}


def say(*x):
    print("[%5.0fs]" % (time.time() - T0), *x, flush=True)


def rep(s=""):
    REPORT.append(s)
    print(s, flush=True)
    open(os.path.join(a.out, "report_e2b_%s.md" % TAG), "w", encoding="utf-8").write("\n".join(REPORT) + "\n")
    json.dump(RES, open(os.path.join(a.out, "results_e2b_%s.json" % TAG), "w"), indent=1)


# ------------------------------------------------------------------ bf16 originals by HTTP range
def hf_open(repo):
    url = "https://huggingface.co/%s/resolve/main/model.safetensors" % repo

    def rng(lo, hi):
        for attempt in range(4):
            try:
                req = urllib.request.Request(url, headers={"Range": "bytes=%d-%d" % (lo, hi - 1)})
                with urllib.request.urlopen(req, timeout=300) as r:
                    return r.read()
            except Exception as e:  # noqa: BLE001
                say("range retry", attempt, repr(e))
                time.sleep(5)
        raise RuntimeError("range read failed")

    n = struct.unpack("<Q", rng(0, 8))[0]
    hdr = json.loads(rng(8, 8 + n))
    return rng, hdr, 8 + n


def hf_tensor(rng, hdr, base, suffix):
    names = [k for k in hdr if k.endswith(suffix)]
    assert len(names) == 1, (suffix, names[:4])
    m = hdr[names[0]]
    raw = rng(base + m["data_offsets"][0], base + m["data_offsets"][1])
    if m["dtype"] == "BF16":
        v = (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32)
    elif m["dtype"] == "F16":
        v = np.frombuffer(raw, dtype="<f2").astype(np.float32)
    else:
        v = np.frombuffer(raw, dtype="<f4").astype(np.float32)
    return v.reshape(m["shape"])


if a.selftest:
    rs_ = np.random.default_rng(0)
    d, n_, N, nL = 128, 512, 3000, len(LAYERS)
    _W = {}

    def weight(name, cols):
        if name not in _W:
            shape = (d, n_) if "down" in name else (n_, d)
            _W[name] = (rs_.standard_normal(shape) / np.sqrt(shape[1])).astype(np.float32)
        return _W[name]

    D = np.zeros((2 * nL, N, d), np.float32)
    for li_, L_ in enumerate(LAYERS):
        D[li_] = rs_.standard_normal((N, d)).astype(np.float32)
        g_ = D[li_] @ weight("blk.%d.ffn_gate.weight" % L_, d).T
        g_ = 0.5 * g_ * (1.0 + np.tanh(0.7978845608028654 * (g_ + 0.044715 * g_ ** 3)))
        D[nL + li_] = (g_ * (D[li_] @ weight("blk.%d.ffn_up.weight" % L_, d).T)) @ weight("blk.%d.ffn_down.weight" % L_, n_).T
    rec = np.repeat(np.arange(100), N // 100).astype(np.int32)
    first = np.zeros(N, bool)
    first[::N // 100] = True
    nrec = 100
    keep = ~first
    te = keep & (rec >= 85)
    va = keep & (rec >= 75) & ~te
    tr = keep & ~te & ~va
else:
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
        rc = subprocess.call([a.g4dump, "-m", a.model, "-f", rf, "--no-escape", "--no-repack", "-c", "2048", "-b", "2048",
                              "-t", str(os.cpu_count() or 4), "-ngl", "0"], env=env, stdout=lg, stderr=subprocess.STDOUT)
    say("dump rc", rc, "size %.2f GB" % (os.path.getsize(ob) / 1e9))
    assert rc == 0

    with open(ob, "rb") as f:
        magic, d, last_only, n_slots = struct.unpack("<4i", f.read(16))
        slot_layers = list(struct.unpack("<%di" % n_slots, f.read(4 * n_slots)))
        hdr0 = f.tell()
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
        f.seek(hdr0)
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
    keep = ~first
    te = keep & (rec >= int(nrec * 0.85))
    va = keep & (rec >= int(nrec * 0.75)) & ~te
    tr = keep & ~te & ~va

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


def ridge(Ftr, Ytr, Fva, Yva):
    """least squares F -> Y, ridge strength chosen on validation; returns W"""
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


def low_rank(W, Xtr, r):
    """reduced-rank version of the linear map x -> x W (rank r, in the metric of the training predictions)"""
    P = Xtr[:4000] @ W
    _, _, Vt = np.linalg.svd(P, full_matrices=False)
    V = Vt[:r].T
    return (W @ V) @ V.T


try:
    HF = None if a.selftest else hf_open(a.hf)
    say("safetensors header: %s tensors" % (len(HF[1]) if HF else "-"))
except Exception as e:  # noqa: BLE001
    HF = None
    say("no bf16 originals:", repr(e))

rep("# E2b — selector and bypass design, layers %s" % a.layers)
rep("")
rep("tokens: train %d, validation %d, held-out test %d (split by prompt)" % (int(tr.sum()), int(va.sum()), int(te.sum())))
BUD = [0.0625, 0.125, 0.25, 0.5]
COLS = ["oracle", "+mean", "+lin256", "+lin", "gate", "gate-cal", "gate+u128", "gate+u512", "pc-full", "pc-r256", "pc-real128", "pc-real512"]

for li, L in enumerate(LAYERS):
    try:
        X, Yd = D[li], D[nL + li]
        Wg, Wu = weight("blk.%d.ffn_gate.weight" % L, d), weight("blk.%d.ffn_up.weight" % L, d)
        Wd = weight("blk.%d.ffn_down.weight" % L, Wg.shape[0])
        n = Wg.shape[0]
        dn = np.linalg.norm(Wd, axis=0)
        un = np.linalg.norm(Wu, axis=1)
        GG = np.zeros((N, n), np.float32)
        U = np.zeros((N, n), np.float32)
        for s in range(0, N, 4096):
            GG[s:s + 4096] = gelu(X[s:s + 4096] @ Wg.T)
            U[s:s + 4096] = X[s:s + 4096] @ Wu.T
        H = GG * U
        Y = H @ Wd.T
        R = RES["L%d" % L] = dict(check=relerr(Y[keep], Yd[keep]))
        rep("")
        rep("## layer %d" % L)
        rep("")
        rep("instrument check (numpy vs engine FFN output): %.2e" % R["check"])
        # ---- quantisation noise floor
        if HF is not None:
            try:
                rng, hdr, base = HF
                Bg = hf_tensor(rng, hdr, base, "language_model.layers.%d.mlp.gate_proj.weight" % L)
                Bu = hf_tensor(rng, hdr, base, "language_model.layers.%d.mlp.up_proj.weight" % L)
                Bd = hf_tensor(rng, hdr, base, "language_model.layers.%d.mlp.down_proj.weight" % L)
                Xt = X[te]
                Yb = (gelu(Xt @ Bg.T) * (Xt @ Bu.T)) @ Bd.T
                R["q4_noise"] = relerr(Y[te], Yb)
                R["q4_weight_err"] = [relerr(Wg, Bg), relerr(Wu, Bu), relerr(Wd, Bd)]
                # which matrix carries the noise: replace one at a time
                R["q4_noise_gate"] = relerr((gelu(Xt @ Wg.T) * (Xt @ Bu.T)) @ Bd.T, Yb)
                R["q4_noise_up"] = relerr((gelu(Xt @ Bg.T) * (Xt @ Wu.T)) @ Bd.T, Yb)
                R["q4_noise_down"] = relerr((gelu(Xt @ Bg.T) * (Xt @ Bu.T)) @ Wd.T, Yb)
                rep("**Q4 noise floor: %.4f** (bf16 original vs this Q4_K_M file, same inputs). By matrix: gate %.4f, up %.4f, down %.4f."
                    " Weight-space error: gate %.4f, up %.4f, down %.4f."
                    % (R["q4_noise"], R["q4_noise_gate"], R["q4_noise_up"], R["q4_noise_down"], *R["q4_weight_err"]))
                del Bg, Bu, Bd, Yb
            except Exception as e:  # noqa: BLE001
                rep("noise floor not available: %r" % (e,))
        St = np.abs(H[te][:1500]) * dn
        srt = -np.sort(-St, axis=1)
        cs = np.cumsum(srt, axis=1) / srt.sum(1, keepdims=True)
        en = np.cumsum(srt ** 2, axis=1) / (srt ** 2).sum(1, keepdims=True)
        rep("top 6.25 / 12.5 / 25 / 50 %% of neurons carry: contribution mass %s; energy %s. Share of neurons with |gelu(g)| < 0.02: %.1f %%."
            % (" / ".join("%.0f %%" % (100 * cs[:, int(b * n) - 1].mean()) for b in BUD),
               " / ".join("%.1f %%" % (100 * en[:, int(b * n) - 1].mean()) for b in BUD), 100 * float((np.abs(GG[te][:1500]) < 0.02).mean())))
        del srt, cs, en, St
        Xtr, Xva, Xte, Ytr, Yva, Yte = X[tr], X[va], X[te], Y[tr], Y[va], Y[te]
        R["linear"] = relerr(Xte @ ridge(Xtr, Ytr, Xva, Yva), Yte)
        # selector ingredients
        urms = np.sqrt((U[tr] ** 2).mean(0))
        del U
        Uw, Sw, Vw = np.linalg.svd(Wu, full_matrices=False)
        R["up_spectrum"] = [float((Sw[:128] ** 2).sum() / (Sw ** 2).sum()), float((Sw[:512] ** 2).sum() / (Sw ** 2).sum())]
        rep("W_up spectrum: the top 128 / 512 of 2560 singular directions hold %.1f / %.1f %% of its energy"
            % (100 * R["up_spectrum"][0], 100 * R["up_spectrum"][1]))
        UH = {r: (Vw[:min(r, len(Sw))].T.copy(), (Uw[:, :r] * Sw[:r]).T.copy()) for r in (128, 512)}
        del Uw, Vw

        def u_hat(r, s, e):
            return (X[s:e] @ UH[r][0]) @ UH[r][1]

        # per-neuron linear predictor of h (full rank), and its rank-256 version
        Htr = H[tr]
        P = ridge(Xtr, Htr, Xva, H[va])            # [d, n]
        P256 = low_rank(P, Xtr, 256)
        del Htr

        def masked(score_fn, k, base_fn=None):
            """y with exact neurons where the score is in the row's top k; base_fn gives h for the others"""
            Yk = np.zeros_like(Y)
            for s in range(0, N, 2048):
                e = min(N, s + 2048)
                sc = score_fn(s, e)
                thr = -np.partition(-sc, k - 1, axis=1)[:, k - 1:k]
                m = sc >= thr
                hh = H[s:e] * m
                if base_fn is not None:
                    hh += base_fn(s, e) * (~m)
                Yk[s:e] = hh @ Wd.T
            return Yk

        def with_bypass(Yk, kind="lin"):
            E_tr, E_va = Ytr - Yk[tr], Yva - Yk[va]
            if kind == "mean":
                return relerr(Yk[te] + E_tr.mean(0), Yte)
            W = ridge(Xtr, E_tr, Xva, E_va)
            if kind == "lin256":
                W = low_rank(W, Xtr, 256)
            return relerr(Yk[te] + Xte @ W, Yte)

        SC = {"contrib": lambda s, e: np.abs(H[s:e]) * dn,
              "gate": lambda s, e: np.abs(GG[s:e]) * (un * dn),
              "gate-cal": lambda s, e: np.abs(GG[s:e]) * (urms * dn),
              "gate+u128": lambda s, e: np.abs(GG[s:e] * u_hat(128, s, e)) * dn,
              "gate+u512": lambda s, e: np.abs(GG[s:e] * u_hat(512, s, e)) * dn}
        rep("")
        rep("| b | " + " | ".join(COLS) + " |")
        rep("|---:|" + "---:|" * len(COLS))
        rep("| 0 | | | | %.4f | | | | | | | | |" % R["linear"])
        for b in BUD:
            k = int(b * n)
            row = {}
            Yo = masked(SC["contrib"], k)
            row["oracle"] = relerr(Yo[te], Yte)
            row["+mean"] = with_bypass(Yo, "mean")
            row["+lin256"] = with_bypass(Yo, "lin256")
            row["+lin"] = with_bypass(Yo, "lin")
            del Yo
            for nm in ("gate", "gate-cal", "gate+u128", "gate+u512"):
                row[nm] = with_bypass(masked(SC[nm], k), "lin")
            for nm, PP in (("pc-full", P), ("pc-r256", P256)):
                base = lambda s, e, PP=PP: X[s:e] @ PP  # noqa: E731
                row[nm] = relerr(masked(lambda s, e: np.abs(H[s:e] - base(s, e)) * dn, k, base)[te], Yte)
                if nm == "pc-r256":
                    for r in (128, 512):
                        row["pc-real%d" % r] = relerr(masked(lambda s, e, r=r: np.abs(GG[s:e] * u_hat(r, s, e) - base(s, e)) * dn, k, base)[te], Yte)
            R["b%g" % b] = row
            rep("| %.2f %% | " % (100 * b) + " | ".join("%.4f" % row[c] for c in COLS) + " |")
            say("layer", L, "budget", b, "done")
        del H, GG, Y, P, P256, UH
    except Exception:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        rep("layer %d FAILED: see log" % L)
say("done")
