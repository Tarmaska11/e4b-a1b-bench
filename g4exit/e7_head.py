#!/usr/bin/env python3
"""E7H -- LM head as "every row at 1 bit, then exact re-score of a shortlist".

Truth: the exact logits of the Q4_K head (262 144 x 2560) on the post-norm hidden state of real positions
(EN prose, RU prose, benchmark answer positions). For each cheap sketch of the head, how large must a shortlist
be to contain what the sampler needs (the exact top-1, the exact top-64 and their probability mass)?

Sketches of a row e (full width, no dimension dropped):
  s1    sign(e) * mean|e|                         1 bit per weight, float hidden state
  s1q4  the same, hidden state rounded to 4 bits  (what a popcount kernel over 4 bit-planes computes)
  s1q8  the same, hidden state at 8 bits over its full range (8 bit-planes, no clipping)
  s2    4 levels per weight                       2 bits per weight
"""
import argparse
import json
import os
import struct
import time

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--g4dump", default="")
ap.add_argument("--model", default="")
ap.add_argument("--bench", default="data/bench_gmmlu.jsonl")
ap.add_argument("--cal", default="g4exit/cal_mmlu.jsonl")
ap.add_argument("--layer", type=int, default=41)
ap.add_argument("--nmmlu", type=int, default=160)
ap.add_argument("--nwiki", type=int, default=40)
ap.add_argument("--out", default="e7_out")
ap.add_argument("--selftest", action="store_true")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
T0 = time.time()
REPORT, RES = [], {}
NS = (1024, 4096, 8192, 16384)


def say(*x):
    print("[%5.0fs]" % (time.time() - T0), *x, flush=True)


def rep(s=""):
    REPORT.append(s)
    print(s, flush=True)
    open(os.path.join(a.out, "report_e7_head.md"), "w", encoding="utf-8").write("\n".join(REPORT) + "\n")
    json.dump(RES, open(os.path.join(a.out, "results_e7_head.json"), "w"), indent=1)


if a.selftest:
    rs = np.random.default_rng(0)
    V, d, P = 6000, 64, 900
    E = (rs.standard_normal((V, d)) * (0.5 + rs.random((V, 1)))).astype(np.float32)
    gout = np.ones(d, np.float32)
    nxt = rs.integers(0, V, P)
    H = (E[nxt] * 2.0 + rs.standard_normal((P, d))).astype(np.float32)
    kind = np.array(["prose_en", "prose_ru", "mmlu_last"])[rs.integers(0, 3, P)]
    NS = (64, 256, 512, 1024)
    TOPK = 16
else:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import e7_corpus as e7c
    texts = e7c.build_corpus(a.bench, a.cal, a.nmmlu, a.nwiki, say)
    say("records", len(texts))
    path = e7c.run_dump(a.g4dump, a.model, texts, str(a.layer), "l_out", a.out, "head", say)
    d, D, tok, rec, pos = e7c.read_dump(path)
    Xl = D[0]
    del D
    os.remove(path)
    import gguf
    from gguf.quants import dequantize

    rd = gguf.GGUFReader(a.model)
    TENS = {t.name: t for t in rd.tensors}
    gout = np.array(TENS["output_norm.weight"].data, dtype=np.float32).reshape(-1)
    turn_id = int(tok[np.flatnonzero(pos == 1)[0]])
    keep = e7c.dedup_keep(tok, rec, pos)
    N = len(tok)
    nxt_all = np.full(N, -1, np.int64)
    same = rec[1:] == rec[:-1]
    nxt_all[:-1][same] = tok[1:][same]
    kind_all = np.array(["skip"] * N, dtype=object)
    # per record: where the model turn starts
    starts = np.flatnonzero(np.r_[True, rec[1:] != rec[:-1]])
    ends = np.r_[starts[1:], N]
    for s, e in zip(starts, ends):
        t = texts[rec[s]]
        if t["kind"] == "mmlu":
            kind_all[s:e - 1] = "mmlu_q"
            kind_all[e - 1] = "mmlu_last"
        else:
            tp = np.flatnonzero(tok[s:e] == turn_id)
            if len(tp) >= 2:
                kind_all[s + tp[1] + 2:e - 1] = "prose_" + t["lang"]
    kind_all[~keep] = "skip"
    rg = np.random.default_rng(3)
    pick = []
    for k, cap in (("prose_en", 2400), ("prose_ru", 2400), ("mmlu_last", 10 ** 6), ("mmlu_q", 1000)):
        idx = np.flatnonzero(kind_all == k)
        if len(idx) > cap:
            idx = np.sort(rg.choice(idx, cap, replace=False))
        pick.append(idx)
        say(k, len(idx))
    pick = np.concatenate(pick)
    x = Xl[pick]
    H = (x / np.sqrt((x ** 2).mean(1, keepdims=True) + 1e-6) * gout).astype(np.float32)
    kind = kind_all[pick].astype(str)
    nxt = nxt_all[pick]
    del Xl
    t = TENS["token_embd.weight"]
    V = t.data.shape[0]
    E = np.empty((V, d), np.float32)
    for s in range(0, V, 16384):                                  # row chunks: keeps the peak memory low
        E[s:s + 16384] = dequantize(t.data[s:s + 16384], t.tensor_type)
    P = len(H)
    TOPK = 64
    say("head", E.shape, "positions", P)

# ------------------------------------------------------------------ sketches (built in row chunks)
alpha1, step = np.empty(V, np.float32), np.empty(V, np.float32)
S1, S2 = np.empty((V, d), np.int8), np.empty((V, d), np.int8)     # S2 value = (q + 0.5) * step
n1 = n2 = den = 0.0
for s in range(0, V, 16384):
    Ec = E[s:s + 16384]
    al = np.abs(Ec).mean(1)
    st = 0.9957 * np.sqrt((Ec ** 2).mean(1))
    st = np.maximum(st, 1e-12)
    q1 = np.where(Ec >= 0, 1, -1).astype(np.int8)
    q2 = np.clip(np.floor(Ec / st[:, None]), -2, 1).astype(np.int8)
    alpha1[s:s + 16384], step[s:s + 16384], S1[s:s + 16384], S2[s:s + 16384] = al, st, q1, q2
    n1 += float(((Ec - q1 * al[:, None]) ** 2).sum())
    n2 += float(((Ec - (q2 + 0.5) * st[:, None]) ** 2).sum())
    den += float((Ec ** 2).sum())
w1, w2 = n1 / den, n2 / den
RES["weight_err"] = dict(s1=w1, s2=w2)


def q4(Hb):
    """hidden state rounded to 16 symmetric levels per vector"""
    dlt = 0.42 * np.sqrt((Hb ** 2).mean(1, keepdims=True))
    return (np.clip(np.floor(Hb / dlt), -8, 7) + 0.5) * dlt


def q8(Hb):
    """hidden state at 8 bits, full range of each vector (no clipping)"""
    dlt = np.abs(Hb).max(1, keepdims=True) / 127.5
    return (np.clip(np.floor(Hb / dlt), -128, 127) + 0.5) * dlt


def mm(Hb, M, scale=None, off=0.0):
    out = np.empty((len(Hb), V), np.float32)
    for s in range(0, V, 32768):
        blk = M[s:s + 32768].astype(np.float32)
        if off:
            blk += off
        out[:, s:s + 32768] = Hb @ blk.T
    if scale is not None:
        out *= scale
    return out


SK = ("s1", "s1q8", "s1q4", "s2")
acc = {k: {sk: {} for sk in SK} for k in sorted(set(kind))}
top1_ok = {k: [] for k in acc}
BS = 192
t_exact = 0.0
for b0 in range(0, P, BS):
    Hb = H[b0:b0 + BS]
    kb = kind[b0:b0 + BS]
    t1 = time.time()
    Lx = mm(Hb, E)
    t_exact += time.time() - t1
    Lx = 30.0 * np.tanh(Lx / 30.0)
    rows = np.arange(len(Hb))
    top1 = np.argmax(Lx, 1)
    part = np.argpartition(-Lx, TOPK - 1, axis=1)[:, :TOPK]
    pm = Lx - Lx.max(1, keepdims=True)
    pr = np.exp(pm)
    pr /= pr.sum(1, keepdims=True)
    p64 = pr[rows[:, None], part]
    for i, k in enumerate(kb):
        top1_ok[k].append(float(top1[i] == nxt[b0 + i]) if nxt[b0 + i] >= 0 else np.nan)
    sk_scores = {"s1": mm(Hb, S1, alpha1), "s1q8": mm(q8(Hb), S1, alpha1), "s1q4": mm(q4(Hb), S1, alpha1),
                 "s2": mm(Hb, S2, step, off=0.5)}
    for sk, S in sk_scores.items():
        r1 = (S > S[rows, top1][:, None]).sum(1)                          # rank of the exact top-1 in the sketch
        s64 = S[rows[:, None], part]
        worst = np.array([(S[i] > s64[i].min()).sum() for i in rows])     # rank of the worst of the exact top-K
        thr = -np.partition(-S, [nn - 1 for nn in NS], axis=1)[:, [nn - 1 for nn in NS]]
        for i, k in enumerate(kb):
            A = acc[k][sk]
            A.setdefault("rank1", []).append(int(r1[i]))
            A.setdefault("rankK", []).append(int(worst[i]))
            for j, nn in enumerate(NS):
                inside = s64[i] >= thr[i, j]
                A.setdefault("miss%d" % nn, []).append(float((p64[i] * ~inside).sum() / p64[i].sum()))
                A.setdefault("all%d" % nn, []).append(float(inside.all()))
                A.setdefault("mass%d" % nn, []).append(float(1.0 - pr[i][S[i] >= thr[i, j]].sum()))
    del sk_scores, Lx, pr
    if b0 % (BS * 5) == 0:
        say("positions", b0 + len(Hb), "of", P)

rep("# E7H -- LM head: cheap sketch of every row + exact re-score of a shortlist")
rep("")
rep("head %d x %d (Q4_K, dequantised = the truth). Positions: %s"
    % (V, d, ", ".join("%s %d" % (k, int((kind == k).sum())) for k in acc)))
rep("instrument check, exact top-1 = the actual next token: %s"
    % ", ".join("%s %.1f %%" % (k, 100 * np.nanmean(v)) for k, v in top1_ok.items() if len(v) and not np.all(np.isnan(v))))
rep("weight-space error of the sketches: 1 bit %.3f, 2 bits %.3f. Exact head: %.2f s per %d positions on this runner."
    % (w1, w2, t_exact / max(1, P // BS), BS))
RES["top1_next"] = {k: float(np.nanmean(v)) for k, v in top1_ok.items() if len(v) and not np.all(np.isnan(v))}
RES["tables"] = {}
for sk in SK:
    rep("")
    rep("## sketch %s" % sk)
    rep("")
    rep("| positions | rank of exact top-1: median / 99.9 %% / max | rank of the worst of the exact top-%d: 99 %% / max | "
        % TOPK + " | ".join("N=%d: top-1 in, top-%d all in, missed top-%d mass (mean / max)" % (nn, TOPK, TOPK) for nn in NS) + " |")
    rep("|---|---|---|" + "---|" * len(NS))
    for k in acc:
        A = acc[k][sk]
        if "rank1" not in A:
            continue
        r1, rk = np.array(A["rank1"]), np.array(A["rankK"])
        cells = []
        tab = dict(rank1=[float(np.median(r1)), float(np.percentile(r1, 99.9)), int(r1.max())],
                   rankK=[float(np.percentile(rk, 99)), int(rk.max())])
        for nn in NS:
            in1 = float((r1 < nn).mean())
            al, ms = np.array(A["all%d" % nn]), np.array(A["miss%d" % nn])
            cells.append("%.2f %%, %.1f %%, %.1e / %.1e" % (100 * in1, 100 * al.mean(), ms.mean(), ms.max()))
            tab["N%d" % nn] = dict(top1_in=in1, all_in=float(al.mean()), miss_mean=float(ms.mean()), miss_max=float(ms.max()),
                                   mass_out_mean=float(np.mean(A["mass%d" % nn])))
        RES["tables"]["%s/%s" % (sk, k)] = tab
        rep("| %s | %d / %d / %d | %d / %d | %s |" % (k, np.median(r1), np.percentile(r1, 99.9), r1.max(),
                                                   np.percentile(rk, 99), rk.max(), " | ".join(cells)))
rep("")
rep("done in %.0f s" % (time.time() - T0))
