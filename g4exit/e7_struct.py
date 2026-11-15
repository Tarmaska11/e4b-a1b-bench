#!/usr/bin/env python3
"""E7 -- closed-form structure tests for rebuilding one E4B FFN as "linear bypass + shared neurons + routed experts".

One layer per run. No gradient training: k-means, least squares, eigen-decompositions only, so nothing can overfit
the way a per-region refit does. Metric: sum ||y_hat - y||^2 / sum ||y||^2 on held-out positions (y = the dense Q4
FFN output), and the same on the RMS-normalised output the model really adds to the residual ("dir").

Tests
  I   rank of the linear bypass
  A1  share of variance explained by the token type (and by the previous token)
  A2  region structure: dynamic oracle / one global static set / one static set per region (regions = k-means on
      the FFN input, or clusters of token TYPES) / budgeted block emulation (shared set + the K nearest regions'
      expert sets, pool <= 2.2x the FFN, every weight copied from E4B)
  B   gate and up pre-activations through a rank-r bottleneck (optimal reduced rank in the data metric)
"""
import argparse
import json
import os
import struct
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e7_corpus as e7c  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--g4dump", default="")
ap.add_argument("--model", default="")
ap.add_argument("--bench", default="data/bench_gmmlu.jsonl")
ap.add_argument("--cal", default="g4exit/cal_mmlu.jsonl")
ap.add_argument("--layer", type=int, default=16)
ap.add_argument("--nmmlu", type=int, default=400)
ap.add_argument("--nwiki", type=int, default=80)
ap.add_argument("--out", default="e7_out")
ap.add_argument("--selftest", action="store_true", help="tiny synthetic model and dump: checks the code end to end")
ap.add_argument("--pop", default="e7", help="e7 = MMLU + Wikipedia prose, de-duplicated; e2b = the exact E2b population")
ap.add_argument("--chat", default="g4exit/chat_prompts.jsonl")
ap.add_argument("--quick", action="store_true", help="stop after the oracle arms and the by-kind breakdown")
ap.add_argument("--sketch", action="store_true", help="section C: cheap selectors (1-bit / 2-bit scans) vs the oracle")
ap.add_argument("--cascade", action="store_true",
                help="section E: a 1-bit scan picks a shortlist m x the final share; exact scores of the shortlist pick the neurons")
ap.add_argument("--blocks", action="store_true",
                help="section D: contiguous blocks of co-active neurons chosen per token by exact score; variable k per token")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
T0 = time.time()
L = a.layer
REPORT, RES = [], {"layer": L}
SFX = "" if a.pop == "e7" else "_" + a.pop

def say(*x):
    print("[%5.0fs]" % (time.time() - T0), *x, flush=True)


def rep(s=""):
    REPORT.append(s)
    print(s, flush=True)
    open(os.path.join(a.out, "report_e7_L%d%s.md" % (L, SFX)), "w", encoding="utf-8").write("\n".join(REPORT) + "\n")
    json.dump(RES, open(os.path.join(a.out, "results_e7_L%d%s.json" % (L, SFX)), "w"), indent=1)


def gelu(g):
    return 0.5 * g * (1.0 + np.tanh(0.7978845608028654 * (g + 0.044715 * g * g * g)))


def fvu(Yh, Y):
    return float(((Yh - Y) ** 2).sum() / (Y ** 2).sum())


# ---------------------------------------------------------------------------------------------- data
if a.selftest:
    rs = np.random.default_rng(0)
    d, n, nrec_, rl = 96, 384, 150, 80
    Wg = (rs.standard_normal((n, d)) / np.sqrt(d)).astype(np.float32)
    Wu = (rs.standard_normal((n, d)) / np.sqrt(d)).astype(np.float32)
    Wd = (rs.standard_normal((d, n)) / np.sqrt(n)).astype(np.float32)
    gam = (1 + 0.1 * rs.standard_normal(d)).astype(np.float32)
    temb = rs.standard_normal((300, d)).astype(np.float32)
    path = os.path.join(a.out, "dump_selftest.bin")
    with open(path, "wb") as f:
        f.write(struct.pack("<4i", 0x31443447, d, 0, 2))
        f.write(struct.pack("<2i", L, L))
        for r in range(nrec_):
            t = np.minimum((rs.pareto(1.1, rl) * 3).astype(np.int32), 299)
            t[:4] = [2, 105, 7, 9]
            ctx = rs.standard_normal(d).astype(np.float32)
            x = temb[t] + 0.8 * ctx + 0.6 * rs.standard_normal((rl, d)).astype(np.float32) + 1.5
            x = x / np.sqrt((x ** 2).mean(1, keepdims=True))
            y = (gelu(x @ Wg.T) * (x @ Wu.T)) @ Wd.T
            f.write(struct.pack("<3i", r, rl, rl))
            f.write(t.astype("<i4").tobytes())
            f.write(x.astype("<f4").tobytes())
            f.write(y.astype("<f4").tobytes())
    texts = [dict(kind="self", lang="en", text="") for _ in range(nrec_)]
    CS = (4, 8, 16)
    NSUB, SVDN = 3000, 1500
else:
    texts = e7c.build_corpus_e2b(a.bench, a.chat) if a.pop == "e2b" else e7c.build_corpus(a.bench, a.cal, a.nmmlu, a.nwiki, say)
    say("records", len(texts))
    path = e7c.run_dump(a.g4dump, a.model, texts, str(L), "ffn_norm,ffn_out", a.out, "L%d" % L, say)
    import gguf  # noqa: E402
    from gguf.quants import dequantize  # noqa: E402

    rd = gguf.GGUFReader(a.model)
    TENS = {t.name: t for t in rd.tensors}

    def weight(name, cols):
        t = TENS[name]
        return np.array(dequantize(t.data, t.tensor_type), dtype=np.float32).reshape(-1, cols)

    CS = (16, 64, 256)
    NSUB, SVDN = 24000, 6000

d, D, tok, rec, pos = e7c.read_dump(path)
X, Yd = D
del D
if not a.selftest:
    os.remove(path)
    Wg, Wu = weight("blk.%d.ffn_gate.weight" % L, d), weight("blk.%d.ffn_up.weight" % L, d)
    n = Wg.shape[0]
    Wd = weight("blk.%d.ffn_down.weight" % L, n)
    gam = np.array(TENS["blk.%d.post_ffw_norm.weight" % L].data, dtype=np.float32).reshape(-1)
WdT = np.ascontiguousarray(Wd.T)       # [n, d]
dn = np.linalg.norm(Wd, axis=0)        # [n]
dn2 = (dn * dn).astype(np.float32)

keep = (pos > 0) if a.pop == "e2b" else e7c.dedup_keep(tok, rec, pos)   # E2b dropped only the first token
X, Yd, tok, rec, pos = X[keep], Yd[keep], tok[keep], rec[keep], pos[keep]
prev = np.zeros_like(tok)
N = len(tok)
nrec = len(texts)
te = rec >= int(nrec * 0.85)
va = (rec >= int(nrec * 0.75)) & ~te
tr = ~te & ~va
say("positions kept", N, "train", int(tr.sum()), "val", int(va.sum()), "test", int(te.sum()))


def nrm(Y):
    return Y / np.sqrt((Y ** 2).mean(1, keepdims=True) + 1e-6) * gam


def dirv(Yh, Y):
    b = nrm(Y)
    return float(((nrm(Yh) - b) ** 2).sum() / (b ** 2).sum())


def Hof(Xc):
    return gelu(Xc @ Wg.T) * (Xc @ Wu.T)


CH = 4096
# instrument check on a slice
chk = fvu(Hof(X[:3000]) @ WdT, Yd[:3000])
RES["check"] = chk
rep("# E7 -- structure of the FFN of layer %d (Q4_K_M weights, closed form, no gradient training)" % L)
rep("")
rep("positions (distinct token prefixes): train %d, validation %d, held-out test %d; split by record"
    % (int(tr.sum()), int(va.sum()), int(te.sum())))
rep("instrument check (numpy FFN vs the engine's ffn_out): %.2e" % chk)
Y = np.empty_like(Yd)
for s in range(0, N, CH):
    Y[s:s + CH] = Hof(X[s:s + CH]) @ WdT
del Yd

# evaluation set: a train subsample (fits the bypass), validation (picks the ridge strength), test (reported)
rs_ = np.random.default_rng(7)
itr = np.flatnonzero(tr)
isub = np.sort(rs_.choice(itr, min(NSUB, len(itr)), replace=False))
iva, ite = np.flatnonzero(va), np.flatnonzero(te)
iev = np.concatenate([isub, iva, ite])
e_tr = np.arange(len(isub))
e_va = len(isub) + np.arange(len(iva))
e_te = len(isub) + len(iva) + np.arange(len(ite))
Xev, Yev = X[iev], Y[iev]
Nev = len(iev)
Hev = np.empty((Nev, n), np.float32)
for s in range(0, Nev, CH):
    Hev[s:s + CH] = Hof(Xev[s:s + CH])
A_tr = (Xev[e_tr].T @ Xev[e_tr]).astype(np.float64)
TRA = np.trace(A_tr) / d


def bypass(E):
    """least-squares linear map x -> E fitted on the train subsample, strength picked on validation"""
    B = (Xev[e_tr].T @ E[e_tr]).astype(np.float64)
    best = None
    for alpha in (1e-4, 1e-3, 1e-2):
        W = np.linalg.solve(A_tr + alpha * TRA * np.eye(d), B).astype(np.float32)
        e = fvu(Xev[e_va] @ W, E[e_va])
        if best is None or e < best[0]:
            best = (e, W)
    return best[1]


LAST = {}


def score(Yk):
    """Yk = the block's output without bypass on the evaluation set -> (with bypass, dir with bypass, no bypass)"""
    W = bypass(Yev - Yk)
    Yh = Yk[e_te] + Xev[e_te] @ W
    LAST["Yh"] = Yh
    return fvu(Yh, Yev[e_te]), dirv(Yh, Yev[e_te]), fvu(Yk[e_te], Yev[e_te])


def masked(maskfn):
    Yk = np.empty((Nev, d), np.float32)
    act = 0.0
    for s in range(0, Nev, CH):
        e = min(Nev, s + CH)
        m = maskfn(s, e)
        act += float(m.sum())
        Yk[s:e] = (Hev[s:e] * m) @ WdT
    return Yk, act / (Nev * n)


# ---------------------------------------------------------------------------------------------- I: linear part
rep("")
rep("## I. the linear part")
Wl = bypass(Yev)
lin = fvu(Xev[e_te] @ Wl, Yev[e_te])
lin_dir = dirv(Xev[e_te] @ Wl, Yev[e_te])
RES["linear"] = [lin, lin_dir]
Pl = Xev[e_tr] @ Wl
ev_, Vl = np.linalg.eigh((Pl.T @ Pl).astype(np.float64))
Vl = Vl[:, ::-1].astype(np.float32)
row = []
for r in (64, 256, 512, 1024):
    r = min(r, d)
    Wr = (Wl @ Vl[:, :r]) @ Vl[:, :r].T
    row.append((r, fvu(Xev[e_te] @ Wr, Yev[e_te])))
RES["linear_rank"] = row
rep("one linear map of x: error %.4f (dir %.4f). Reduced rank: %s"
    % (lin, lin_dir, ", ".join("r=%d %.4f" % x for x in row)))
del Pl

# ---------------------------------------------------------------------------------------------- A1: token share
rep("")
rep("## A1. how much the token type explains (held-out positions of types seen >= 8 times in train)")
for i in range(1, N):
    prev[i] = tok[i - 1] if rec[i] == rec[i - 1] else -1
prev[0] = -1
Rall = Y - X @ Wl


def token_share(V, key):
    ktr = key[tr]
    u, inv, cnt = np.unique(ktr, return_inverse=True, return_counts=True)
    order = np.argsort(inv, kind="stable")
    starts = np.concatenate([[0], np.cumsum(cnt)[:-1]])
    sums = np.add.reduceat(V[tr][order], starts, axis=0)
    means = sums / cnt[:, None]
    gmean = V[tr].mean(0)
    good = {int(k): i for i, k in enumerate(u) if cnt[i] >= 8 and k >= 0}
    kte = key[te]
    idx = np.array([good.get(int(k), -1) for k in kte])
    cov = idx >= 0
    if cov.sum() < 50:
        return 0.0, float(cov.mean())
    Vt = V[te][cov]
    num = ((Vt - means[idx[cov]]) ** 2).sum()
    den = ((Vt - gmean) ** 2).sum()
    return float(1 - num / den), float(cov.mean())


RES["token_share"] = {}
for nm, V in (("FFN input x", X), ("FFN output y", Y), ("nonlinear residual y - Lx", Rall)):
    s1, c1 = token_share(V, tok)
    s2, c2 = token_share(V, prev)
    RES["token_share"][nm] = [s1, c1, s2, c2]
    rep("- %s: current token explains %.1f %% of the variance (coverage %.0f %%), previous token %.1f %% (coverage %.0f %%)"
        % (nm, 100 * s1, 100 * c1, 100 * s2, 100 * c2))
Rte_energy = float((Rall[te] ** 2).sum())
# output-side rank of the nonlinear residual
Rtr = Rall[isub]
ev_, Vo = np.linalg.eigh((Rtr.T @ Rtr).astype(np.float64))
Vo = Vo[:, ::-1].astype(np.float32)
row = []
for r in (256, 512, 1024):
    r = min(r, d)
    Pr = Rall[te] @ Vo[:, :r]
    row.append((r, float((Pr ** 2).sum() / Rte_energy)))
RES["resid_out_rank"] = row
rep("- output side: the top %s directions hold %s %% of the nonlinear residual's energy"
    % (" / ".join(str(x[0]) for x in row), " / ".join("%.1f" % (100 * x[1]) for x in row)))
del Rall, Rtr

# ---------------------------------------------------------------------------------------------- regions
mu = X[tr].mean(0)


def feat(Xc):
    F = Xc - mu
    return F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-8)


def kmeans(F, C, iters=12, seed=0):
    rg = np.random.default_rng(seed)
    cent = F[rg.choice(len(F), C, replace=False)].copy()
    for _ in range(iters):
        asg = np.concatenate([np.argmax(F[s:s + 16384] @ cent.T, 1) for s in range(0, len(F), 16384)])
        for c in range(C):
            m = asg == c
            if m.any():
                v = F[m].mean(0)
                cent[c] = v / (np.linalg.norm(v) + 1e-8)
            else:
                cent[c] = F[rg.integers(len(F))]
    return cent


Ftr = feat(X[tr])
fit = Ftr[rs_.choice(len(Ftr), min(60000, len(Ftr)), replace=False)]
CENT = {C: kmeans(fit, C, seed=C) for C in CS}
del fit
# token types: mean feature per type, clustered
utok, inv, cnt = np.unique(tok[tr], return_inverse=True, return_counts=True)
order = np.argsort(inv, kind="stable")
starts = np.concatenate([[0], np.cumsum(cnt)[:-1]])
tmean = np.add.reduceat(Ftr[order], starts, axis=0) / cnt[:, None]
tmean /= (np.linalg.norm(tmean, axis=1, keepdims=True) + 1e-8)
CT = CS[2]
big = cnt >= 3
tcent = kmeans(tmean[big] if big.sum() > CT * 2 else tmean, min(CT, max(2, int(big.sum()) // 2)), seed=99)
CT = len(tcent)
t2c = dict(zip(utok.tolist(), np.argmax(tmean @ tcent.T, 1).tolist()))
t2row = dict(zip(utok.tolist(), range(len(utok))))


def tok_region(tk):
    return np.array([t2c.get(int(t), -1) for t in tk])


SCH = ["x%d" % C for C in CS] + ["tok%d" % CT]
NC = {("x%d" % C): C for C in CS}
NC["tok%d" % CT] = CT


def region1(name, Xc, tk):
    if name.startswith("x"):
        return np.argmax(feat(Xc) @ CENT[NC[name]].T, 1)
    return tok_region(tk)


# importance of every neuron per region, from ALL train positions
imp_g = np.zeros(n, np.float64)
IMP = {s: np.zeros((NC[s], n), np.float64) for s in SCH}
CNT = {s: np.zeros(NC[s], np.float64) for s in SCH}
Xtr_, toktr_ = X[tr], tok[tr]
for s in range(0, len(Xtr_), CH):
    Xc = Xtr_[s:s + CH]
    H2 = Hof(Xc)
    H2 = H2 * H2 * dn2
    imp_g += H2.sum(0)
    for nm in SCH:
        rg_ = region1(nm, Xc, toktr_[s:s + CH])
        oh = np.zeros((len(Xc), NC[nm]), np.float32)
        ok = rg_ >= 0
        oh[np.flatnonzero(ok), rg_[ok]] = 1.0
        IMP[nm] += oh.T @ H2
        CNT[nm] += oh.sum(0)
del Xtr_, H2
say("importance accumulated")
for nm in SCH:
    IMP[nm] = (IMP[nm] / np.maximum(CNT[nm], 1)[:, None]).astype(np.float32)
imp_g = (imp_g / tr.sum()).astype(np.float32)
RES["region_sizes"] = {nm: [int(CNT[nm].min()), int(np.median(CNT[nm])), int(CNT[nm].max())] for nm in SCH}
gorder = np.argsort(-imp_g)

# routing of the evaluation positions
Fev = feat(Xev)
tokev = tok[iev]
ROUTE = {}
for C in CS:
    ROUTE["x%d" % C] = np.argsort(-(Fev @ CENT[C].T), axis=1)[:, :16].astype(np.int32)
trg = tok_region(tokev)
ROUTE["tok%d" % CT] = trg[:, None]
tok_cov = float((trg[e_te] >= 0).mean())
# a token-table router INTO the mid-size x-regions: the type's mean feature picks the regions
CM = CS[1]
trow = np.array([t2row.get(int(t), -1) for t in tokev])
tsim = tmean @ CENT[CM].T
ttop = np.argsort(-tsim, axis=1)[:, :16].astype(np.int32)
ROUTE["tok>x%d" % CM] = np.where(trow[:, None] >= 0, ttop[np.maximum(trow, 0)], -1)


def static_mask(idx):
    v = np.zeros(n, bool)
    v[idx] = True
    return v


def region_sets(nm, k, exclude=None):
    """[C, n] bool: the top-k neurons of every region (optionally outside `exclude`)"""
    I = IMP[nm].copy()
    if exclude is not None:
        I[:, exclude] = -1.0
    M = np.zeros(I.shape, bool)
    top = np.argpartition(-I, k - 1, axis=1)[:, :k]
    M[np.arange(len(I))[:, None], top] = True
    return M


# ---------------------------------------------------------------------------------------------- A2
rep("")
rep("## A2. region structure")
rep("")
rep("regions: %s. Sizes (train positions per region, min / median / max): %s. Token-table coverage of test: %.0f %%"
    % (", ".join(SCH), "; ".join("%s %d/%d/%d" % (k, *v) for k, v in RES["region_sizes"].items()), 100 * tok_cov))
rep("")
rep("| arm | active neurons | error with bypass | dir | error without bypass |")
rep("|---|---:|---:|---:|---:|")
RES["arms"] = []


def arm(name, maskfn):
    Yk, act = masked(maskfn)
    e, dv, e0 = score(Yk)
    RES["arms"].append(dict(arm=name, active=act, err=e, dir=dv, err_nobypass=e0))
    rep("| %s | %.1f %% | %.4f | %.4f | %.4f |" % (name, 100 * act, e, dv, e0))
    return e


rep("| linear map alone | 0 | %.4f | %.4f | |" % (lin, lin_dir))
ORA, BYK = {}, {}
kind_te = np.array(["%s_%s" % (texts[r]["kind"], texts[r]["lang"]) for r in rec[iev][e_te]])


def by_kind(Yh):
    out = {}
    for k in sorted(set(kind_te)):
        m = kind_te == k
        if m.sum() >= 200:
            out[k] = (fvu(Yh[m], Yev[e_te][m]), int(m.sum()))
    return out


for b in (0.0625, 0.125, 0.25, 0.5):
    k = int(b * n)

    def mf(s, e, k=k):
        sc = np.abs(Hev[s:e]) * dn
        thr = -np.partition(-sc, k - 1, axis=1)[:, k - 1:k]
        return sc >= thr
    ORA[b] = arm("dynamic oracle %.4g %%" % (100 * b), mf)
    BYK[b] = by_kind(LAST["Yh"])
rep("")
rep("the same arms by kind of position (held-out; bypass fitted on all kinds):")
rep("")
bl = by_kind(Xev[e_te] @ Wl)
ks = sorted(bl)
rep("| arm | " + " | ".join("%s (n=%d)" % (k, bl[k][1]) for k in ks) + " |")
rep("|---|" + "---:|" * len(ks))
rep("| linear map alone | " + " | ".join("%.4f" % bl[k][0] for k in ks) + " |")
for b in sorted(BYK):
    rep("| dynamic oracle %.4g %% + bypass | " % (100 * b) + " | ".join("%.4f" % BYK[b][k][0] for k in ks) + " |")
RES["by_kind"] = dict(linear={k: bl[k][0] for k in ks}, oracle={str(b): {k: BYK[b][k][0] for k in ks} for b in BYK})
# ---------------------------------------------------------------------------------------------- C: cheap selectors
if a.sketch:
    rep("")
    rep("## C. cheap selectors: which neurons to compute exactly, chosen from a low-precision scan (exact values for the chosen, bypass for the rest)")
    rep("")
    rep("| selector (bytes scanned, share of the dense FFN) | computed exactly | error with bypass | dir | overlap with the oracle's set |")
    rep("|---|---:|---:|---:|---:|")

    def sk1(W):
        return np.sign(W).astype(np.float32) * np.abs(W).mean(1, keepdims=True)

    def sk2(W):
        st = 0.9957 * np.sqrt((W ** 2).mean(1, keepdims=True))
        return ((np.clip(np.floor(W / st), -2, 1) + 0.5) * st).astype(np.float32)

    urms = np.sqrt(((Xev[e_tr][:8000] @ Wu.T) ** 2).mean(0))
    SEL = [("exact gate x calibrated |u| (33 %)", Wg, None),
           ("1-bit gate x calibrated |u| (7 %)", sk1(Wg), None),
           ("1-bit gate and 1-bit up (15 %)", sk1(Wg), sk1(Wu)),
           ("2-bit gate and 2-bit up (30 %)", sk2(Wg), sk2(Wu))]
    RES["sketch"] = []
    orc = np.abs(Hev) * dn
    for nm, Sg, Su in SEL:
        SC = np.empty((Nev, n), np.float32)
        for s0 in range(0, Nev, CH):
            g = gelu(Xev[s0:s0 + CH] @ Sg.T)
            SC[s0:s0 + CH] = np.abs(g) * (urms if Su is None else np.abs(Xev[s0:s0 + CH] @ Su.T)) * dn
        for b in (0.125, 0.25, 0.5):
            k = int(b * n)
            ov = 0.0

            def mf(s0, e0, k=k):
                sc = SC[s0:e0]
                thr = -np.partition(-sc, k - 1, axis=1)[:, k - 1:k]
                return sc >= thr
            Yk, act = masked(mf)
            e, dv, e0_ = score(Yk)
            for s0 in range(0, Nev, CH):                      # overlap with the oracle's top-k set
                o = orc[s0:s0 + CH]
                to = -np.partition(-o, k - 1, axis=1)[:, k - 1:k]
                ov += float(((o >= to) & mf(s0, min(Nev, s0 + CH))).sum())
            ov /= (Nev * k)
            RES["sketch"].append(dict(selector=nm, frac=b, err=e, dir=dv, overlap=ov, oracle=ORA[b]))
            rep("| %s | %.4g %% | %.4f | %.4f | %.0f %% |" % (nm, 100 * b, e, dv, 100 * ov))
        del SC
    del orc
    rep("")
    rep("(oracle with bypass on the same positions: " + ", ".join("%.4g %%: %.4f" % (100 * b, ORA[b]) for b in sorted(ORA)) + ")")
# ---------------------------------------------------------------------------------------------- D: blocks, variable k
if a.blocks:
    rep("")
    rep("## D. blocks of co-active neurons (a file layout could read them as contiguous runs), and a variable number of neurons per token")
    rep("")
    orc = np.abs(Hev) * dn
    JS = (10, 8, 6) if n % 1024 == 0 else (6, 4, 2)          # tree depths -> block sizes n / 2^j
    TC = min(16384, len(e_tr))
    k25 = int(0.25 * n)
    F = np.empty((n, TC), np.float32)                         # neuron x train position: is it in the token's top 25 %?
    for s0 in range(0, TC, CH):
        o = orc[s0:min(TC, s0 + CH)]
        thr = -np.partition(-o, k25 - 1, axis=1)[:, k25 - 1:k25]
        F[:, s0:s0 + len(o)] = (o >= thr).T
    rate = F.mean(1)
    F -= rate[:, None]
    F /= np.linalg.norm(F, axis=1, keepdims=True) + 1e-6      # rows are now correlation features

    def cluster_order(depth, seed=0):
        """balanced recursive bisection along the main co-activation direction; leaf order = the new neuron order"""
        rs2 = np.random.default_rng(seed)
        order = np.arange(n)
        segs = [(0, n, 0)]
        while segs:
            s0, e0, lv = segs.pop()
            if lv >= depth:
                continue
            idx = order[s0:e0]
            Fs = F[idx]
            mu = Fs.mean(0)
            v = rs2.standard_normal(TC).astype(np.float32)
            for _ in range(10):
                u = Fs @ v - float(mu @ v)
                v = Fs.T @ u - mu * float(u.sum())
                v /= np.linalg.norm(v) + 1e-12
            u = Fs @ v - float(mu @ v)
            order[s0:e0] = idx[np.argsort(u, kind="stable")]
            h = (e0 - s0) // 2
            segs += [(s0, s0 + h, lv + 1), (s0 + h, e0, lv + 1)]
        return order

    def within_corr(perm, B):
        """mean pairwise correlation of the top-25 % indicator inside a block (train positions)"""
        S = F[perm].reshape(n // B, B, TC).sum(1)
        return float(((S * S).sum() - (F * F).sum()) / (n * (B - 1)))

    def block_fn(perm, B, b):
        nba = n // B
        nb = max(1, int(round(b * nba)))
        inv = np.empty(n, np.int64)
        inv[perm] = np.arange(n)

        def mf(s0, e0):
            o = orc[s0:e0]
            en = (o * o)[:, perm].reshape(-1, nba, B).sum(2)
            thr = -np.partition(-en, nb - 1, axis=1)[:, nb - 1:nb]
            return np.repeat(en >= thr, B, axis=1)[:, inv]
        return mf

    t1 = time.time()
    perm_c = cluster_order(max(JS))
    perm_r = np.random.default_rng(11).permutation(n)
    say("neuron order by co-activation: %.0f s" % (time.time() - t1))
    RES["blocks"], RES["vark"] = [], []
    Yk, act = masked(block_fn(np.arange(n), 1, 0.25))
    chk1 = score(Yk)[0]
    RES["blocks_check"] = [chk1, ORA[0.25]]
    rep("instrument check: blocks of ONE neuron at 25 %% give %.4f; the neuron-level oracle above gave %.4f" % (chk1, ORA[0.25]))
    rep("")
    rep("| neuron order | block size | mean correlation inside a block | computed exactly | error with bypass | dir | x the neuron-level oracle |")
    rep("|---|---:|---:|---:|---:|---:|---:|")
    ARMS_D = [("co-activation clusters", perm_c, n >> j) for j in JS] + [("random order (control)", perm_r, n >> JS[1])]
    for nm, perm, B in ARMS_D:
        wc = within_corr(perm, B)
        for b in (0.125, 0.25, 0.5):
            Yk, act = masked(block_fn(perm, B, b))
            e, dv, _ = score(Yk)
            RES["blocks"].append(dict(order=nm, B=B, frac=b, active=act, corr=wc, err=e, dir=dv, oracle=ORA[b]))
            rep("| %s | %d | %.3f | %.4g %% | %.4f | %.4f | %.2f |" % (nm, B, wc, 100 * act, e, dv, e / ORA[b]))
    del F
    rep("")
    rep("variable k: one threshold for every token instead of a fixed count (same MEAN share; hard tokens get more neurons)")
    rep("")
    rep("| rule | mean share computed | neurons per token, 10th / 50th / 90th percentile | error with bypass | dir | x fixed top-k at that share |")
    rep("|---|---:|---:|---:|---:|---:|")
    sub = orc[e_tr[:: max(1, len(e_tr) // 2000)]]
    srm = np.sqrt((sub ** 2).mean(1, keepdims=True))
    te_s = e_te[:: max(1, len(e_te) // 3000)]
    for b in (0.125, 0.25, 0.5):
        ta = float(np.quantile(sub, 1 - b))
        tr_ = float(np.quantile(sub / srm, 1 - b))
        for nm, fn in (("absolute threshold on the score", lambda s0, e0, t=ta: orc[s0:e0] >= t),
                       ("threshold relative to the token's own rms score",
                        lambda s0, e0, t=tr_: orc[s0:e0] >= t * np.sqrt((orc[s0:e0] ** 2).mean(1, keepdims=True)))):
            Yk, act = masked(fn)
            e, dv, _ = score(Yk)
            kk = np.concatenate([fn(int(i), int(i) + 1).sum(1) for i in te_s])
            q = np.percentile(kk, (10, 50, 90))
            RES["vark"].append(dict(rule=nm, frac=b, active=act, k_p10=float(q[0]), k_p50=float(q[1]), k_p90=float(q[2]),
                                    err=e, dir=dv, oracle=ORA[b]))
            rep("| %s | %.2f %% | %d / %d / %d | %.4f | %.4f | %.2f |" % (nm, 100 * act, q[0], q[1], q[2], e, dv, e / ORA[b]))
    del orc
# ---------------------------------------------------------------------------------------------- E: cascade selector
if a.cascade:
    rep("")
    rep("## E. cascade: a 1-bit scan of all neurons picks a shortlist, exact scores of the shortlist pick the neurons to compute")
    rep("")
    rep("bytes are a share of the dense three-matrix FFN at Q4_K: a 1-bit matrix = 1/13.5; exact rows of the shortlist; down rows of the final set")
    rep("")

    def sk1c(W):
        return np.sign(W).astype(np.float32) * np.abs(W).mean(1, keepdims=True)

    urms_c = np.sqrt(((Xev[e_tr][:8000] @ Wu.T) ** 2).mean(0))
    orc = np.abs(Hev) * dn
    GC = np.empty((Nev, n), np.float32)                               # the exact-gate selector's score
    for s0 in range(0, Nev, CH):
        GC[s0:s0 + CH] = np.abs(gelu(Xev[s0:s0 + CH] @ Wg.T)) * urms_c * dn
    BASE = {}
    for b in (0.125, 0.25):
        k = int(b * n)

        def mfg(s0, e0, k=k):
            sc = GC[s0:e0]
            thr = -np.partition(-sc, k - 1, axis=1)[:, k - 1:k]
            return sc >= thr
        Yk, _ = masked(mfg)
        BASE[b] = score(Yk)[0]
    rep("reference on the same positions: " + "; ".join("%.4g %%: oracle %.4f, exact-gate selector %.4f (bytes %.0f %%)" % (
        100 * b, ORA[b], BASE[b], 100 * (1 / 3 + b * 2 / 3)) for b in sorted(BASE)))
    rep("")
    rep("| 1-bit scan | final choice by | computed | shortlist | bytes read | oracle's set inside the shortlist | error with bypass | x oracle | x exact-gate selector |")
    rep("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    RES["cascade"] = []
    PRE = [("gate only, x calibrated |u|", sk1c(Wg), None, 1 / 13.5),
           ("gate and up", sk1c(Wg), sk1c(Wu), 2 / 13.5)]
    for nm, Sg, Su, pb in PRE:
        SC = np.empty((Nev, n), np.float32)
        for s0 in range(0, Nev, CH):
            g = gelu(Xev[s0:s0 + CH] @ Sg.T)
            SC[s0:s0 + CH] = np.abs(g) * (urms_c if Su is None else np.abs(Xev[s0:s0 + CH] @ Su.T)) * dn
        for b in (0.125, 0.25):
            k = int(b * n)
            for m in (1.5, 2.0, 3.0):
                ks = int(min(n, round(m * k)))
                rec = 0.0
                for s0 in range(0, Nev, CH):                      # how much of the oracle's set the shortlist holds
                    sc = SC[s0:s0 + CH]
                    ts = -np.partition(-sc, ks - 1, axis=1)[:, ks - 1:ks]
                    o = orc[s0:s0 + CH]
                    to = -np.partition(-o, k - 1, axis=1)[:, k - 1:k]
                    rec += float(((o >= to) & (sc >= ts)).sum())
                rec /= (Nev * k)
                for fin_nm, FIN, by in (("oracle score (exact gate and up rows of the shortlist)", orc, pb + m * b * 2 / 3 + b / 3),
                                        ("exact gate x calibrated |u| (exact gate rows of the shortlist)", GC, pb + m * b / 3 + b * 2 / 3)):
                    def mfc(s0, e0, k=k, ks=ks, FIN=FIN):
                        sc = SC[s0:e0]
                        ts = -np.partition(-sc, ks - 1, axis=1)[:, ks - 1:ks]
                        f2 = np.where(sc >= ts, FIN[s0:e0], -1.0)
                        t2 = -np.partition(-f2, k - 1, axis=1)[:, k - 1:k]
                        return f2 >= np.maximum(t2, 0.0)
                    Yk, act = masked(mfc)
                    e, dv, _ = score(Yk)
                    RES["cascade"].append(dict(scan=nm, final=fin_nm, frac=b, m=m, active=act, bytes=by, recall=rec, err=e, dir=dv,
                                               oracle=ORA[b], gate=BASE[b]))
                    rep("| %s | %s | %.4g %% | x%.1f | %.0f %% | %.1f %% | %.4f | %.2f | %.2f |" % (
                        nm, fin_nm, 100 * act, m, 100 * by, 100 * rec, e, e / ORA[b], e / BASE[b]))
        del SC
    del orc, GC
if a.quick:
    rep("")
    rep("quick run: stopped here, %.0f s" % (time.time() - T0))
    raise SystemExit(0)
rep("")
rep("| arm | active neurons | error with bypass | dir | error without bypass |")
rep("|---|---:|---:|---:|---:|")
STA = {}
for b in (0.125, 0.25, 0.5):
    v = static_mask(gorder[:int(b * n)])
    STA[b] = arm("one global static set %.4g %%" % (100 * b), lambda s, e, v=v: np.broadcast_to(v, (e - s, n)))
for nm in SCH:
    for b in (0.125, 0.25, 0.5):
        M = region_sets(nm, int(b * n))
        r1 = ROUTE[nm][:, 0]
        gv = static_mask(gorder[:int(b * n)])

        def mf(s, e, M=M, r1=r1, gv=gv):
            rr = r1[s:e]
            m = M[np.maximum(rr, 0)]
            if (rr < 0).any():
                m = m.copy()
                m[rr < 0] = gv
            return m
        arm("static set per region, %s, %.4g %% (pool not budgeted)" % (nm, 100 * b), mf)

rep("")
rep("budgeted block emulation: shared set + the K nearest regions' expert sets, weights copied from E4B, pool <= 2.25x")
rep("")
rep("| arm | active neurons | error with bypass | dir | error without bypass |")
rep("|---|---:|---:|---:|---:|")
c1, c2, c3 = ["x%d" % C for C in CS]
BUD = [  # nominal active, shared fraction, router, expert fraction, K
    (0.125, 1 / 16, c2, 1 / 64, 4), (0.125, 1 / 16, c3, 1 / 256, 16),
    (0.25, 1 / 8, c1, 1 / 8, 1), (0.25, 1 / 8, c2, 1 / 32, 4), (0.25, 1 / 8, c3, 1 / 128, 16),
    (0.25, 1 / 8, "tok>" + c2, 1 / 32, 4),
    (0.375, 1 / 8, c2, 1 / 32, 8),
    (0.5, 1 / 4, c1, 1 / 8, 2), (0.5, 1 / 4, c2, 1 / 32, 8),
]
for nom, fs, router, fm, K in BUD:
    ns, m = max(1, int(fs * n)), max(1, int(fm * n))
    src = router.split(">")[-1]
    shared = static_mask(gorder[:ns])
    M = region_sets(src, m, exclude=gorder[:ns]).astype(np.float32)
    rt = ROUTE[router][:, :K]
    gv = static_mask(gorder[:int(nom * n)])

    def mf(s, e, M=M, rt=rt, shared=shared, gv=gv, K=K):
        rr = rt[s:e]
        oh = np.zeros((e - s, M.shape[0]), np.float32)
        ok = rr[:, 0] >= 0
        rows = np.flatnonzero(ok)
        for j in range(K):
            oh[rows, rr[rows, j]] = 1.0
        mm = ((oh @ M) > 0) | shared
        if (~ok).any():
            mm[~ok] = gv
        return mm
    pool = (ns + NC[src] * m) / n
    arm("%.4g %% nominal: shared %d + %d x %d of %d experts, router %s (pool %.2fx)"
        % (100 * nom, ns, K, m, NC[src], router, pool), mf)

# ---------------------------------------------------------------------------------------------- B: bottleneck
rep("")
rep("## B. gate and up pre-activations through a rank-r bottleneck (all neurons kept), + linear correction")
Xs = X[isub]
C2 = (Xs.T @ Xs).astype(np.float64)
w_, V_ = np.linalg.eigh(C2)
w_ = np.maximum(w_, w_.max() * 1e-10)
Ch = (V_ * np.sqrt(w_)) @ V_.T                      # C^(1/2)
Wj = np.concatenate([Wg, Wu], 0)                    # [2n, d]
G = (Wj.T @ Wj).astype(np.float64)                  # [d, d]
K_ = Ch @ G @ Ch
s2, A_ = np.linalg.eigh(K_)
s2, A_ = np.maximum(s2[::-1], 1e-30), A_[:, ::-1]
cum = np.cumsum(s2) / s2.sum()
Bfull = (Wj.astype(np.float64) @ (Ch @ A_)) / np.sqrt(s2)   # [2n, d] orthonormal columns: output-space directions
xc = X[tr] - mu
evx = np.linalg.eigvalsh((xc[:20000].T @ xc[:20000]).astype(np.float64))[::-1]
cx = np.cumsum(evx) / evx.sum()
del xc
RES["x_pca"] = [float(cx[min(r, d) - 1]) for r in (128, 256, 512, 1024)]
rep("centred x: the top 128 / 256 / 512 / 1024 principal directions hold %s %% of its variance"
    % " / ".join("%.1f" % (100 * v) for v in RES["x_pca"]))
rep("")
rep("| rank r | pre-activation energy kept | gate+up bytes vs dense | error with correction | dir | error without |")
rep("|---:|---:|---:|---:|---:|---:|")
RES["bottleneck"] = []
for r in (256, 512, 1024, 1536):
    r = min(r, d)
    Br = Bfull[:, :r].astype(np.float32)             # [2n, r]
    Pin = (Br.T @ Wj)                                # [r, d]
    Yk = np.empty((Nev, d), np.float32)
    for s in range(0, Nev, CH):
        Z = (Xev[s:s + CH] @ Pin.T) @ Br.T           # [B, 2n]
        Yk[s:s + CH] = (gelu(Z[:, :n]) * Z[:, n:]) @ WdT
    e, dv, e0 = score(Yk)
    frac = r * (d + 2 * n) / (2 * n * d)
    RES["bottleneck"].append(dict(r=r, kept=float(cum[r - 1]), bytes=frac, err=e, dir=dv, err_nocorr=e0))
    rep("| %d | %.1f %% | %.1f %% | %.4f | %.4f | %.4f |" % (r, 100 * cum[r - 1], 100 * frac, e, dv, e0))

rep("")
rep("done in %.0f s" % (time.time() - T0))
