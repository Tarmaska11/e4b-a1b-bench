#!/usr/bin/env python3
"""E0 analysis: exit-quality map of Gemma 4 E4B from dumped residuals (see research/fable-2026-11/JOURNAL.md).

Reads art/g4exit-e4b-<shard>/ and art/g4exit-e2b-<shard>/ (npz + meta.json), the E4B and E2B GGUFs (for the
output norm and the tied LM head), and writes report.md + results.json.
"""
import argparse
import glob
import json
import math
import os
import time
import traceback

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--art", default="art")
ap.add_argument("--e4b", required=True)
ap.add_argument("--e2b", default="")
ap.add_argument("--out", default="g4exit_out")
ap.add_argument("--max-pos", type=int, default=3000)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
T0 = time.time()
REPORT = []
RES = {}


def say(*x):
    s = " ".join(str(v) for v in x)
    print("[%5.0fs] %s" % (time.time() - T0, s), flush=True)


def rep(s=""):
    REPORT.append(s)
    print(s, flush=True)


def flush_out():
    open(os.path.join(a.out, "report.md"), "w", encoding="utf-8").write("\n".join(REPORT) + "\n")
    json.dump(RES, open(os.path.join(a.out, "results.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)


# ------------------------------------------------------------------ GGUF: output norm + LM head
def load_head(path):
    import gguf
    from gguf.quants import dequantize
    rd = gguf.GGUFReader(path)

    def fval(suffix, default):
        for k, f in rd.fields.items():
            if k.endswith(suffix):
                try:
                    return float(f.contents())
                except Exception:  # noqa: BLE001
                    try:
                        return float(f.parts[f.data[0]][0])
                    except Exception:  # noqa: BLE001
                        pass
        return default

    eps = fval("layer_norm_rms_epsilon", 1e-6)
    cap = fval("final_logit_softcapping", 30.0)
    wn = E = None
    for t in rd.tensors:
        if t.name == "output_norm.weight":
            wn = np.array(dequantize(t.data, t.tensor_type), dtype=np.float32).reshape(-1)
        if t.name == "token_embd.weight":
            say("token_embd type", t.tensor_type.name, "shape", list(t.shape))
            E = np.array(dequantize(t.data, t.tensor_type), dtype=np.float32)
    E = E.reshape(-1, wn.shape[0])
    say("head", E.shape, "eps", eps, "softcap", cap)
    return dict(E=E, wn=wn, eps=eps, cap=cap)


def normed(H, hd):
    H = H.astype(np.float32)
    return H / np.sqrt((H * H).mean(-1, keepdims=True) + hd["eps"]) * hd["wn"]


def softcap(L, hd):
    c = hd["cap"]
    return c * np.tanh(L / c) if c else L


def logsoftmax(L):
    m = L.max(-1, keepdims=True)
    return L - m - np.log(np.exp(L - m).sum(-1, keepdims=True))


def binom2(k, n):
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


# ------------------------------------------------------------------ load artifacts
def shard_dirs(tag):
    ds = sorted(glob.glob(os.path.join(a.art, "g4exit-%s-*" % tag)), key=lambda p: int(p.rsplit("-", 1)[1]))
    return [d for d in ds if os.path.exists(os.path.join(d, "meta.json"))]


def load_bench(dirs, name, layer_pos=None):
    """-> dict key -> (lang, gold, vec[n_layers, d]); layers list"""
    out, layers = {}, None
    for d in dirs:
        p = os.path.join(d, "bench_%s.npz" % name)
        if not os.path.exists(p):
            say("missing", p)
            continue
        meta = json.load(open(os.path.join(d, "meta.json"), encoding="utf-8"))
        z = np.load(p)
        layers = [int(x) for x in z["layers"]]
        if "H" not in z:
            continue
        H = z["H"]
        for k, ri in enumerate(z["rec_idx"]):
            b = meta["bench"][int(ri)]
            out[b["id"] + "|" + b["lang"]] = (b["lang"], b["gold"], H[:, k, :])
    return out, layers


def mc_score(vecs, keys, hd, letters, ref_pred=None):
    """vecs [n,d] raw residuals -> preds, accuracy per language"""
    X = normed(vecs, hd)
    L4 = X @ hd["E"][letters].T
    pred = L4.argmax(1)
    return pred


def acc_table(pred, golds, langs):
    r = {}
    for lg in ("en", "ru"):
        m = langs == lg
        r[lg] = (int((pred[m] == golds[m]).sum()), int(m.sum()))
    return r


def paired(pred, ref, golds, langs):
    r = {}
    for lg in ("en", "ru"):
        m = langs == lg
        ok, okr = pred[m] == golds[m], ref[m] == golds[m]
        b, c = int((okr & ~ok).sum()), int((~okr & ok).sum())
        r[lg] = (c - b, binom2(min(b, c), b + c))
    return r


def fmt_acc(t):
    return " | ".join("%.1f" % (100.0 * t[lg][0] / max(t[lg][1], 1)) for lg in ("en", "ru"))


E4 = shard_dirs("e4b")
E2 = shard_dirs("e2b")
say("e4b shards", len(E4), "e2b shards", len(E2))
rep("# E0 — exit-quality map of Gemma 4 E4B (Q4_K_M), no training")
rep("")
rep("shards: e4b %d, e2b %d" % (len(E4), len(E2)))

hd = load_head(a.e4b)
meta0 = json.load(open(os.path.join(E4[0], "meta.json"), encoding="utf-8"))
letters = [meta0["letters"][c][0] for c in "ABCD"]
RES["template"] = dict(source=meta0.get("template_source"), prefix=meta0.get("prefix"), suffix=meta0.get("suffix"),
                       raw=meta0.get("template_raw"), letters=meta0["letters"])
rep("template source: %s; prefix %r suffix %r; letter ids %s" % (meta0.get("template_source"), meta0.get("prefix"),
                                                                 meta0.get("suffix"), letters))
for c in "ABCD":
    if len(meta0["letters"][c]) != 1:
        rep("WARNING: letter %s is not a single token: %s" % (c, meta0["letters"][c]))

# ------------------------------------------------------------------ 1. benchmark, every arm
G2I = {"A": 0, "B": 1, "C": 2, "D": 3}
bench_full, FULL_LAYERS = load_bench(E4, "full")
keys = sorted(bench_full.keys())
langs = np.array([bench_full[k][0] for k in keys])
golds = np.array([G2I[bench_full[k][1]] for k in keys])
Hb = np.stack([bench_full[k][2] for k in keys])  # [n, n_layers, d]
say("bench full", Hb.shape, "layers", FULL_LAYERS)
LI = {l: i for i, l in enumerate(FULL_LAYERS)}
last = FULL_LAYERS[-1]

pred_full = mc_score(Hb[:, LI[last], :], keys, hd, letters)
t_full = acc_table(pred_full, golds, langs)
rep("")
rep("## 1. Benchmark (1 140 Global-MMLU, restricted A–D argmax at the last position)")
rep("")
rep("**Instrument check.** full model: EN %.1f (n=%d), RU %.1f (n=%d). Anchor through `gguf-mc`: 68.9 / 64.9."
    % (100.0 * t_full["en"][0] / max(1, t_full["en"][1]), t_full["en"][1], 100.0 * t_full["ru"][0] / max(1, t_full["ru"][1]), t_full["ru"][1]))
RES["bench"] = {"full": dict(acc=t_full)}
rep("")
rep("| arm | layers kept | EN | RU | ΔEN (p) | ΔRU (p) |")
rep("|---|---:|---:|---:|---|---|")
rep("| full | 42 | %s | — | — |" % fmt_acc(t_full))
BENCH_PRED = {"full": pred_full}
BENCH_VEC = {"full": Hb[:, LI[last], :]}


def bench_row(name, kept, vec):
    try:
        pred = mc_score(vec, keys, hd, letters)
        t = acc_table(pred, golds, langs)
        pr = paired(pred, pred_full, golds, langs)
        rep("| %s | %s | %s | %+d (%.3f) | %+d (%.3f) |" % (name, kept, fmt_acc(t), pr["en"][0], pr["en"][1], pr["ru"][0], pr["ru"][1]))
        RES["bench"][name] = dict(kept=kept, acc=t, paired=pr)
        BENCH_PRED[name] = pred
        BENCH_VEC[name] = vec
    except Exception:  # noqa: BLE001
        rep("| %s | %s | ERROR | | | |" % (name, kept))
        traceback.print_exc()


for l in FULL_LAYERS[:-1]:
    bench_row("exit%d (plain truncation)" % (l + 1), l + 1, Hb[:, LI[l], :])

SKIPS = []
for ar in meta0.get("arms") or []:
    if not ar or not str(ar.get("name", "")).startswith("bench_") or ar["name"] == "bench_full":
        continue
    kd = ar.get("desc", "")
    fk = ar.get("ffn_kept") or {}
    if fk:
        kd += " [measured: %.1f %% of neurons kept in %d layers]" % (
            100.0 * sum(v[0] for v in fk.values()) / max(1, sum(v[1] for v in fk.values())), len(fk))
    SKIPS.append((ar["name"][6:], kd))
for name, kept in SKIPS:
    d, _ = load_bench(E4, name)
    if len(d) != len(keys):
        rep("| %s | %s | missing (%d/%d records) | | | |" % (name, kept, len(d), len(keys)))
        if not d:
            continue
    try:
        vec = np.stack([d[k][2][0] for k in keys])
        bench_row(name, kept, vec)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
flush_out()

# ------------------------------------------------------------------ 2. chat data
chat = []  # per shard: dict(meta, full npz, offsets)
for d in E4:
    meta = json.load(open(os.path.join(d, "meta.json"), encoding="utf-8"))
    p = os.path.join(d, "chat_full.npz")
    if not os.path.exists(p):
        continue
    z = np.load(p)
    if "H" not in z:
        continue
    chat.append(dict(dir=d, meta=meta, rec_idx=z["rec_idx"], n_tok=z["n_tok"], H=z["H"], tokens=z["tokens"],
                     off=np.concatenate([[0], np.cumsum(z["n_tok"])])))
say("chat shards", len(chat))

# assistant positions: hidden at p predicts token p+1, for p in [P-1, T-2]
POS = []  # (shard_i, global_vec_index, lang, is_test)
bad_boundary = 0
for si, c in enumerate(chat):
    suf = c["meta"].get("suffix_tokens") or []
    for k, ri in enumerate(c["rec_idx"]):
        r = c["meta"]["chat"][int(ri)]
        P, T = r.get("n_prompt"), int(c["n_tok"][k])
        if P is None:
            continue
        toks = c["tokens"][c["off"][k]:c["off"][k] + T]
        if suf and list(toks[P - len(suf):P]) != list(suf):
            bad_boundary += 1
        for p in range(P - 1, T - 1):
            POS.append((si, int(c["off"][k]) + p, r["lang"], si >= len(chat) - 2))
rep("")
rep("## 2. Chat, token level (replies sampled by the full model, teacher-forced)")
rep("")
rep("assistant positions: %d (EN %d, RU %d); records whose prompt boundary did not match the template suffix: %d"
    % (len(POS), sum(1 for x in POS if x[2] == "en"), sum(1 for x in POS if x[2] == "ru"), bad_boundary))

rng = np.random.default_rng(1234)
idx_all = np.arange(len(POS))
if len(idx_all) > a.max_pos:
    idx_all = np.sort(rng.choice(idx_all, a.max_pos, replace=False))
idx_test = np.array([i for i, x in enumerate(POS) if x[3]])


def gather_full(idx, layer):
    return np.stack([chat[POS[i][0]]["H"][LI[layer], POS[i][1], :] for i in idx]).astype(np.float32)


def gather_arm(idx, name):
    cache = {}
    out = []
    for i in idx:
        si = POS[i][0]
        if si not in cache:
            p = os.path.join(chat[si]["dir"], "chat_%s.npz" % name)
            cache[si] = np.load(p)["H"] if os.path.exists(p) else None
        if cache[si] is None:
            return None
        out.append(cache[si][0, POS[i][1], :])
    return np.stack(out).astype(np.float32)


def token_eval(idx, arms, chunk=192):
    """arms: dict name -> He [n,d]. Returns dict name -> dict(agree, pmax, kl, pf) arrays, plus full top-64 ids."""
    Hf = gather_full(idx, last)
    n = len(idx)
    out = {k: dict(agree=np.zeros(n, bool), pmax=np.zeros(n, np.float32), kl=np.zeros(n, np.float32),
                   pf=np.zeros(n, np.float32)) for k in arms}
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        lsf = logsoftmax(softcap(normed(Hf[s:e], hd) @ hd["E"].T, hd))
        af = lsf.argmax(1)
        pf = np.exp(lsf)
        for k, He in arms.items():
            lse = logsoftmax(softcap(normed(He[s:e], hd) @ hd["E"].T, hd))
            o = out[k]
            o["agree"][s:e] = lse.argmax(1) == af
            o["pmax"][s:e] = np.exp(lse.max(1))
            o["kl"][s:e] = (pf * (lsf - lse)).sum(1)
            o["pf"][s:e] = np.exp(lse[np.arange(e - s), af])
        if s % (chunk * 4) == 0:
            say("token_eval %d/%d" % (e, n))
    return out


def gate(agree, pmax, tol):
    """largest fraction of tokens that can exit (highest confidence first) with disagreement <= tol among them;
    returns (fraction, threshold)"""
    o = np.argsort(-pmax)
    dis = np.cumsum(~agree[o]) / np.arange(1, len(o) + 1)
    ok = np.where(dis <= tol)[0]
    if len(ok) == 0:
        return 0.0, 1.0
    k = ok[-1]
    return (k + 1) / len(o), float(pmax[o][k])


def token_rows(stats, idx, kept_of):
    lg = np.array([POS[i][2] for i in idx])
    rep("")
    rep("| arm | kept | top-1 agree EN | RU | mean KL EN | RU | exit@1% EN | RU | exit@2% EN | RU |")
    rep("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for k, o in stats.items():
        row, rr = [], {}
        for name, f in (("agree", lambda m: 100.0 * o["agree"][m].mean()), ("kl", lambda m: float(o["kl"][m].mean())),
                        ("g1", lambda m: 100.0 * gate(o["agree"][m], o["pmax"][m], 0.01)[0]),
                        ("g2", lambda m: 100.0 * gate(o["agree"][m], o["pmax"][m], 0.02)[0])):
            for l in ("en", "ru"):
                v = f(lg == l)
                rr[name + "_" + l] = v
                row.append("%.3f" % v if name == "kl" else "%.1f" % v)
        rr["thr1"] = gate(o["agree"], o["pmax"], 0.01)[1]
        rr["thr2"] = gate(o["agree"], o["pmax"], 0.02)[1]
        rep("| %s | %s | %s |" % (k, kept_of.get(k, ""), " | ".join(row)))
        RES.setdefault("chat", {})[k] = rr


TOK = {}
try:
    arms, kept_of = {}, {}
    for l in FULL_LAYERS[:-1]:
        nm = "exit%d" % (l + 1)
        arms[nm] = gather_full(idx_all, l)
        kept_of[nm] = l + 1
    for name, kept in SKIPS:
        He = gather_arm(idx_all, name)
        if He is not None:
            arms[name] = He
            kept_of[name] = kept
    say("token eval on", len(idx_all), "positions,", len(arms), "arms")
    TOK = token_eval(idx_all, arms)
    rep("")
    rep("%d sampled assistant positions. `exit@x%%` = share of tokens that can leave at this readout, most confident"
        " first, while at most x %% of them differ from the full model's top-1." % len(idx_all))
    token_rows(TOK, idx_all, kept_of)
except Exception:  # noqa: BLE001
    traceback.print_exc()
    rep("token-level section FAILED: see log")
flush_out()

# ------------------------------------------------------------------ 3. closed-form linear patch (ridge), fitted on chat
rep("")
rep("## 3. Closed-form linear patch h_L -> h_41 (ridge towards identity; fitted on chat shards 0..n-3, all positions)")
try:
    tr = [(si, c) for si, c in enumerate(chat) if si < len(chat) - 2]

    def rows(c, l):  # every position except each record's first (BOS), whose residual is an outlier
        m = np.ones(c["H"].shape[1], bool)
        m[c["off"][:-1]] = False
        return c["H"][LI[l]][m]

    Ytr = np.concatenate([rows(c, last) for _, c in tr]).astype(np.float64)
    rep("")
    rep("train tokens: %d; test: benchmark (1 140 last positions) and %d held-out assistant positions" % (len(Ytr), len(idx_test)))
    rep("")
    rep("| arm | kept | alpha | EN | RU | ΔEN (p) | ΔRU (p) |")
    rep("|---|---:|---:|---:|---:|---|---|")
    patch_arms, kept_of = {}, {}
    for l in [x for x in FULL_LAYERS[:-1] if x <= 35]:
        Xtr = np.concatenate([rows(c, l) for _, c in tr]).astype(np.float64)
        A, B = Xtr.T @ Xtr, Xtr.T @ Ytr
        d = A.shape[0]
        best = None
        Xte, Yte = gather_full(idx_test, l).astype(np.float64), gather_full(idx_test, last).astype(np.float64)
        for alpha in (1e-3, 1e-2, 1e-1, 1.0):
            lam = alpha * np.trace(A) / d
            W = np.linalg.solve(A + lam * np.eye(d), B + lam * np.eye(d))
            mse = float(((Xte @ W - Yte) ** 2).mean() / (Yte ** 2).mean())
            if best is None or mse < best[0]:
                best = (mse, alpha, W)
        mse, alpha, W = best
        base = float(((Xte - Yte) ** 2).mean() / (Yte ** 2).mean())
        nm = "patch%d" % (l + 1)
        vec = (Hb[:, LI[l], :].astype(np.float64) @ W).astype(np.float32)
        pred = mc_score(vec, keys, hd, letters)
        t = acc_table(pred, golds, langs)
        pr = paired(pred, pred_full, golds, langs)
        rep("| %s (rel. MSE %.3f -> %.3f) | %d | %g | %s | %+d (%.3f) | %+d (%.3f) |"
            % (nm, base, mse, l + 1, alpha, fmt_acc(t), pr["en"][0], pr["en"][1], pr["ru"][0], pr["ru"][1]))
        RES["bench"][nm] = dict(kept=l + 1, acc=t, paired=pr, alpha=alpha, rel_mse=mse, rel_mse_identity=base)
        BENCH_PRED[nm], BENCH_VEC[nm] = pred, vec
        patch_arms[nm] = (Xte @ W).astype(np.float32)
        patch_arms["exit%d (same positions)" % (l + 1)] = Xte.astype(np.float32)
        kept_of[nm] = l + 1
        kept_of["exit%d (same positions)" % (l + 1)] = l + 1
        np.save(os.path.join(a.out, "W_%s.npy" % nm), W.astype(np.float32))
    flush_out()
    if len(idx_test):
        st = token_eval(idx_test, patch_arms)
        rep("")
        rep("Held-out chat positions (%d):" % len(idx_test))
        token_rows(st, idx_test, kept_of)
        TOK.update({k: v for k, v in st.items() if k.startswith("patch")})
except Exception:  # noqa: BLE001
    traceback.print_exc()
    rep("patch section FAILED: see log")
flush_out()

# ------------------------------------------------------------------ 4. gated benchmark: exit when confident, else full
rep("")
rep("## 4. Benchmark with a confidence gate (threshold = the chat exit@1% threshold of that arm)")
rep("")
rep("| arm | threshold | exits on bench EN | RU | gated EN | RU | Δ vs full EN | RU |")
rep("|---|---:|---:|---:|---:|---:|---:|---:|")
try:
    for nm, vec in BENCH_VEC.items():
        key = nm.replace(" (plain truncation)", "")
        if nm == "full" or key not in RES.get("chat", {}):
            continue
        thr = RES["chat"][key]["thr1"]
        pm = np.zeros(len(vec), np.float32)
        for s in range(0, len(vec), 192):
            ls = logsoftmax(softcap(normed(vec[s:s + 192], hd) @ hd["E"].T, hd))
            pm[s:s + 192] = np.exp(ls.max(1))
        ex = pm >= thr
        gp = np.where(ex, BENCH_PRED[nm], pred_full)
        t = acc_table(gp, golds, langs)
        pr = paired(gp, pred_full, golds, langs)
        rep("| %s | %.3f | %.1f | %.1f | %s | %+d | %+d |" % (nm, thr, 100.0 * ex[langs == "en"].mean(), 100.0 * ex[langs == "ru"].mean(),
                                                           fmt_acc(t), pr["en"][0], pr["ru"][0]))
        RES.setdefault("gated", {})[nm] = dict(thr=thr, exit_en=float(ex[langs == "en"].mean()), exit_ru=float(ex[langs == "ru"].mean()),
                                               acc=t, paired=pr)
except Exception:  # noqa: BLE001
    traceback.print_exc()
    rep("gated section FAILED: see log")
flush_out()

# ------------------------------------------------------------------ 5. LM head: exact-top-k shortlist recall
rep("")
rep("## 5. LM head shortlist (rank-r sketch -> top-N rows -> exact logits on those rows only)")
try:
    E = hd["E"]
    G = (E.T @ E).astype(np.float64)
    w, V = np.linalg.eigh(G)
    V = V[:, ::-1].astype(np.float32)
    sub = idx_all[:1500]
    X = normed(gather_full(sub, last), hd)
    top64 = np.zeros((len(sub), 64), np.int64)
    TOP1 = np.zeros(len(sub), np.int64)
    for s in range(0, len(sub), 192):
        L = X[s:s + 192] @ E.T
        top64[s:s + 192] = np.argpartition(-L, 64, axis=1)[:, :64]
        TOP1[s:s + 192] = L.argmax(1)
    rep("")
    rep("%d positions. Sketch bytes are for 8-bit storage; the exact head is %d MiB at 4.5 bpw."
        % (len(sub), int(E.size * 4.5 / 8 / 2 ** 20)))
    rep("")
    rep("| rank r | shortlist N | sketch MiB | top-1 in shortlist | all of top-64 in shortlist | mean share of top-64 |")
    rep("|---:|---:|---:|---:|---:|---:|")
    for r in (64, 128, 256, 512):
        Ar = E @ V[:, :r]
        Z = X @ V[:, :r]
        for N in (1024, 4096, 16384):
            h1 = hall = 0
            share = 0.0
            for s in range(0, len(sub), 192):
                La = Z[s:s + 192] @ Ar.T
                sl = np.argpartition(-La, N, axis=1)[:, :N]
                for j in range(sl.shape[0]):
                    S = set(sl[j].tolist())
                    h1 += int(TOP1[s + j]) in S
                    c = sum(1 for v in top64[s + j] if int(v) in S)
                    hall += c == 64
                    share += c / 64.0
            n = len(sub)
            rep("| %d | %d | %.0f | %.2f %% | %.2f %% | %.2f %% |" % (r, N, E.shape[0] * r / 2 ** 20, 100.0 * h1 / n, 100.0 * hall / n, 100.0 * share / n))
            RES.setdefault("head", {})["r%d_N%d" % (r, N)] = dict(top1=h1 / n, all64=hall / n, share64=share / n)
        flush_out()
except Exception:  # noqa: BLE001
    traceback.print_exc()
    rep("head section FAILED: see log")
flush_out()

# ------------------------------------------------------------------ 6. E2B control
rep("")
rep("## 6. Control: Gemma 4 E2B (Q4_K_M), same instrument")
try:
    if E2 and a.e2b:
        hd = E = G = None  # free the E4B head before loading E2B's
        hd2 = load_head(a.e2b)
        m2 = json.load(open(os.path.join(E2[0], "meta.json"), encoding="utf-8"))
        let2 = [m2["letters"][c][0] for c in "ABCD"]
        d2, _ = load_bench(E2, "full")
        k2 = sorted(d2.keys())
        lg2 = np.array([d2[k][0] for k in k2])
        g2 = np.array([G2I[d2[k][1]] for k in k2])
        v2 = np.stack([d2[k][2][0] for k in k2])
        p2 = mc_score(v2, k2, hd2, let2)
        t2 = acc_table(p2, g2, lg2)
        rep("")
        rep("E2B: EN %.1f (n=%d), RU %.1f (n=%d)" % (100.0 * t2["en"][0] / t2["en"][1], t2["en"][1], 100.0 * t2["ru"][0] / t2["ru"][1], t2["ru"][1]))
        if k2 == keys:
            pr = paired(p2, pred_full, golds, langs)
            rep("vs E4B full, paired: EN %+d (p %.3f), RU %+d (p %.3f)" % (pr["en"][0], pr["en"][1], pr["ru"][0], pr["ru"][1]))
            RES["e2b"] = dict(acc=t2, paired=pr)
        else:
            RES["e2b"] = dict(acc=t2)
    else:
        rep("no E2B data")
except Exception:  # noqa: BLE001
    traceback.print_exc()
    rep("E2B section FAILED: see log")
flush_out()
say("done")
