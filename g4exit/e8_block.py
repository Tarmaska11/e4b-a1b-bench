#!/usr/bin/env python3
"""E8 -- gradient pilot of ONE rebuilt E4B FFN block, small scale, on a CPU runner.

Block:  y_hat = B x  +  shared GeGLU (n_s neurons)  +  sum over the K routed experts of w_k * GeGLU expert_k(x)
        router: cosine of the centred input against one vector per expert, top-K, softmax weights times K.
Every weight starts from E4B (closed form: k-means regions, region-important neurons copied, least-squares bypass)
and is then trained by SGD to reproduce the dense FFN's output, weighted the way the model's post-norm weighs it.
Data: real FFN inputs of one layer from the Q4_K_M file; optionally half of every batch is a mix of two real
inputs re-normalised onto the input manifold, labelled by the dense FFN itself (unlimited fresh samples).
Reported on REAL held-out positions only.

Arms (fractions of the 10 240 neurons): moe125 = 640 shared + 4 of 64 experts x 160; moe25 = 1280 + 4 of 64 x 320;
moe375 = 1280 + 8 of 64 x 320; moe50 = 2560 + 8 of 64 x 320; static125 / static25 = a narrower dense FFN + bypass;
control = the full FFN with 10 % weight noise (must be recovered, else the trainer proves nothing).
"""
import argparse
import json
import math
import os
import struct
import sys
import time

import numpy as np
import torch
import torch.nn.functional as TF

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e7_corpus as e7c  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--g4dump", default="")
ap.add_argument("--model", default="")
ap.add_argument("--bench", default="data/bench_gmmlu.jsonl")
ap.add_argument("--cal", default="g4exit/cal_mmlu.jsonl")
ap.add_argument("--layer", type=int, default=16)
ap.add_argument("--arms", default="moe25:3.0", help="comma list of arm:hours")
ap.add_argument("--nmmlu", type=int, default=400)
ap.add_argument("--nwiki", type=int, default=80)
ap.add_argument("--chunks", type=int, default=3)
ap.add_argument("--aug", type=float, default=0.5)
ap.add_argument("--batch", type=int, default=1024)
ap.add_argument("--total-min", type=float, default=0.0, help="if > 0, scale the arm hours so the script ends by then")
ap.add_argument("--out", default="e8_out")
ap.add_argument("--selftest", action="store_true")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
T0 = time.time()
L = a.layer
TAG = "L%d_%s" % (L, a.arms.replace(":", "-").replace(",", "_").replace(".", "p")) + ("" if a.aug > 0 else "_noaug")
REPORT, RES = [], {"layer": L, "arms": {}}
torch.set_num_threads(os.cpu_count() or 4)
torch.manual_seed(0)

ARMS = {
    "moe125": dict(fs=1 / 16, N=64, fm=1 / 64, K=4),
    "moe25": dict(fs=1 / 8, N=64, fm=1 / 32, K=4),
    "moe375": dict(fs=1 / 8, N=64, fm=1 / 32, K=8),
    "moe50": dict(fs=1 / 4, N=64, fm=1 / 32, K=8),
    "static125": dict(fs=1 / 8, N=0, fm=0, K=0),
    "static25": dict(fs=1 / 4, N=0, fm=0, K=0),
    "control": dict(fs=1.0, N=0, fm=0, K=0, noise=0.10),
    "topk125": dict(fs=1.0, N=0, fm=0, K=0, topk=0.125),
    "topk25": dict(fs=1.0, N=0, fm=0, K=0, topk=0.25),
    "topk50": dict(fs=1.0, N=0, fm=0, K=0, topk=0.5),
}


def say(*x):
    print("[%5.0fs]" % (time.time() - T0), *x, flush=True)


def rep(s=""):
    REPORT.append(s)
    print(s, flush=True)
    open(os.path.join(a.out, "report_e8_%s.md" % TAG), "w", encoding="utf-8").write("\n".join(REPORT) + "\n")
    json.dump(RES, open(os.path.join(a.out, "results_e8_%s.json" % TAG), "w"), indent=1)


def gelu(g):
    return TF.gelu(g, approximate="tanh")


# ---------------------------------------------------------------------------------------------- data
if a.selftest:
    rs = np.random.default_rng(0)
    d, n, nrec_, rl = 64, 256, 260, 80
    Wg = (rs.standard_normal((n, d)) / np.sqrt(d)).astype(np.float32)
    Wu = (rs.standard_normal((n, d)) / np.sqrt(d)).astype(np.float32)
    Wd = (rs.standard_normal((d, n)) / np.sqrt(n)).astype(np.float32)
    gam = (1 + 0.1 * rs.standard_normal(d)).astype(np.float32)
    gam_f = np.ones(d, np.float32)
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
            y = (np.asarray(gelu(torch.from_numpy(x @ Wg.T))) * (x @ Wu.T)) @ Wd.T
            f.write(struct.pack("<3i", r, rl, rl))
            f.write(t.astype("<i4").tobytes())
            f.write(x.astype("<f4").tobytes())
            f.write(y.astype("<f4").tobytes())
    nrec = nrec_
    for k in ARMS:
        if ARMS[k]["N"]:
            ARMS[k]["N"] = 8
else:
    texts = e7c.build_corpus(a.bench, a.cal, a.nmmlu, a.nwiki, say, chunks=a.chunks)
    nrec = len(texts)
    say("records", nrec)
    path = e7c.run_dump(a.g4dump, a.model, texts, str(L), "ffn_norm,ffn_out", a.out, "L%d" % L, say)
    import gguf  # noqa: E402
    from gguf.quants import dequantize  # noqa: E402

    rd = gguf.GGUFReader(a.model)
    TENS = {t.name: t for t in rd.tensors}

    def weight(name, cols):
        t = TENS[name]
        return np.array(dequantize(t.data, t.tensor_type), dtype=np.float32).reshape(-1, cols)

d, D_, tok, rec, pos = e7c.read_dump(path)
X, Yd = D_
del D_
if not a.selftest:
    os.remove(path)
    Wg, Wu = weight("blk.%d.ffn_gate.weight" % L, d), weight("blk.%d.ffn_up.weight" % L, d)
    n = Wg.shape[0]
    Wd = weight("blk.%d.ffn_down.weight" % L, n)
    gam = np.array(TENS["blk.%d.post_ffw_norm.weight" % L].data, dtype=np.float32).reshape(-1)
    gam_f = np.array(TENS["blk.%d.ffn_norm.weight" % L].data, dtype=np.float32).reshape(-1)
keep = e7c.dedup_keep(tok, rec, pos)
X, Yd, rec = X[keep], Yd[keep], rec[keep]
N_ = len(X)
te = rec >= int(nrec * 0.90)
va = (rec >= int(nrec * 0.85)) & ~te
tr = ~te & ~va
tWg, tWu, tWdT = torch.from_numpy(Wg), torch.from_numpy(Wu), torch.from_numpy(np.ascontiguousarray(Wd.T))
tgam = torch.from_numpy(gam)
g2 = tgam * tgam
dn = torch.from_numpy(np.linalg.norm(Wd, axis=0))


@torch.no_grad()
def teacher(x):
    return (gelu(x @ tWg.T) * (x @ tWu.T)) @ tWdT


Xt = torch.from_numpy(X)
del X
chk = float(((teacher(Xt[:3000]) - torch.from_numpy(Yd[:3000])) ** 2).sum() / (torch.from_numpy(Yd[:3000]) ** 2).sum())
del Yd
Yt = torch.empty_like(Xt)
for s in range(0, N_, 4096):
    Yt[s:s + 4096] = teacher(Xt[s:s + 4096])
itr, iva, ite = [torch.from_numpy(np.flatnonzero(m)) for m in (tr, va, te)]
rs_ = np.random.default_rng(7)
isub = torch.from_numpy(np.sort(rs_.choice(itr.numpy(), min(24000, len(itr)), replace=False)))
ite_s = ite[torch.from_numpy(np.sort(rs_.choice(len(ite), min(8192, len(ite)), replace=False)))]
itr_s = isub[:4096]
rep("# E8 -- gradient pilot of one rebuilt block, layer %d (%s)" % (L, TAG))
rep("")
rep("real positions (distinct token prefixes): train %d, validation %d, held-out test %d; mixed-input share of a batch %.0f %%"
    % (len(itr), len(iva), len(ite), 100 * a.aug))
rep("instrument check (dense FFN recomputed vs the engine's ffn_out): %.2e" % chk)
RES["check"] = chk
RES["tokens"] = [int(len(itr)), int(len(iva)), int(len(ite))]


def nrm(Y):
    return Y / torch.sqrt((Y ** 2).mean(1, keepdim=True) + 1e-6) * tgam


def fvu(Yh, Y):
    return float(((Yh - Y) ** 2).sum() / (Y ** 2).sum())


def dirv(Yh, Y):
    b = nrm(Y)
    return float(((nrm(Yh) - b) ** 2).sum() / (b ** 2).sum())


def ridge(Xa, Ea, Xv, Ev):
    A = (Xa.T @ Xa).double()
    B = (Xa.T @ Ea).double()
    tra = torch.trace(A) / d
    best = None
    for alpha in (1e-4, 1e-3, 1e-2):
        W = torch.linalg.solve(A + alpha * tra * torch.eye(d, dtype=torch.float64), B).float()
        e = fvu(Xv @ W, Ev)
        if best is None or e < best[0]:
            best = (e, W)
    return best[1]            # x @ W


# references on the held-out sample: linear map alone, dynamic oracle + bypass
Wl = ridge(Xt[isub], Yt[isub], Xt[iva], Yt[iva])
lin = fvu(Xt[ite_s] @ Wl, Yt[ite_s])
RES["linear"] = lin


@torch.no_grad()
def oracle(frac):
    k = int(frac * n)

    def out(idx):
        Yk = torch.empty(len(idx), d)
        for s in range(0, len(idx), 2048):
            x = Xt[idx[s:s + 2048]]
            h = gelu(x @ tWg.T) * (x @ tWu.T)
            sc = h.abs() * dn
            thr = sc.kthvalue(n - k + 1, dim=1).values[:, None]
            Yk[s:s + 2048] = (h * (sc >= thr)) @ tWdT
        return Yk
    Ya, Yv, Ys = out(isub[:12000]), out(iva), out(ite_s)
    W = ridge(Xt[isub[:12000]], Yt[isub[:12000]] - Ya, Xt[iva], Yt[iva] - Yv)
    Yh = Ys + Xt[ite_s] @ W
    return fvu(Yh, Yt[ite_s]), dirv(Yh, Yt[ite_s])


mu = Xt[itr].mean(0)


def feat(x):
    return TF.normalize(x - mu, dim=1)


def kmeans(Fm, C, iters=12, seed=0):
    g = torch.Generator().manual_seed(seed)
    cent = Fm[torch.randperm(len(Fm), generator=g)[:C]].clone()
    for _ in range(iters):
        asg = torch.cat([(Fm[s:s + 16384] @ cent.T).argmax(1) for s in range(0, len(Fm), 16384)])
        for c in range(C):
            m = asg == c
            cent[c] = TF.normalize(Fm[m].mean(0), dim=0) if m.any() else Fm[int(torch.randint(len(Fm), (1,), generator=g))]
    return cent


# ---------------------------------------------------------------------------------------------- one arm
def run_arm(name, hours):
    cfg = ARMS[name]
    ns, Ne, K = max(1, int(cfg["fs"] * n)), cfg["N"], cfg["K"]
    m = max(1, int(cfg["fm"] * n)) if Ne else 0
    topk = cfg.get("topk", 0.0)
    active = topk if topk else (ns + K * m) / n
    R = RES["arms"][name] = dict(shared=ns, experts=Ne, expert_size=m, K=K, active=active, hours=hours, topk=topk)
    rep("")
    if topk:
        rep("## arm %s: all %d neurons kept in the file, per token only the top %.1f %% (by their own exact score) are "
            "used, + bypass; every weight trained with that rule in the loop; budget %.2f h" % (name, n, 100 * topk, hours))
    else:
        rep("## arm %s: %d shared + %d of %d experts x %d = %.1f %% of the neurons active, pool %.2fx, budget %.2f h"
            % (name, ns, K, Ne, m, 100 * active, (ns + Ne * m) / n, hours))
    # ---- closed-form start
    samp = isub
    imp_g = torch.zeros(n, dtype=torch.float64)
    cent = kmeans(feat(Xt[samp]), Ne, seed=Ne) if Ne else None
    imp_r = torch.zeros(Ne, n, dtype=torch.float64) if Ne else None
    cnt = torch.zeros(Ne, dtype=torch.float64) if Ne else None
    with torch.no_grad():
        for s in range(0, len(samp), 4096):
            x = Xt[samp[s:s + 4096]]
            h2 = (gelu(x @ tWg.T) * (x @ tWu.T)) ** 2 * (dn * dn)
            imp_g += h2.sum(0).double()
            if Ne:
                oh = TF.one_hot((feat(x) @ cent.T).argmax(1), Ne).float()
                imp_r += (oh.T @ h2).double()
                cnt += oh.sum(0).double()
    gorder = torch.argsort(-imp_g)
    sh = gorder[:ns]
    noise = cfg.get("noise", 0.0)

    def nz(W):
        return W * (1 + noise * torch.randn_like(W)) if noise else W.clone()
    P = dict(Gs=nz(tWg[sh]), Us=nz(tWu[sh]), Ds=nz(tWdT[sh]))          # Ds: [ns, d]  (h @ Ds)
    if Ne:
        imp_r = (imp_r / cnt.clamp_min(1)[:, None]).float()
        imp_r[:, sh] = -1.0
        sets = imp_r.topk(m, dim=1).indices                           # [Ne, m]
        M = torch.zeros(Ne, n)
        M.scatter_(1, sets, 1.0)
        with torch.no_grad():                                          # how often a copy is doubled when routed
            fx = feat(Xt[samp[:20000]])
            top = (fx @ cent.T).topk(K, dim=1).indices
            oh = torch.zeros(len(fx), Ne).scatter_(1, top, 1.0)
            cover = oh @ M                                             # [P, n] multiplicity among the routed experts
            mult = (oh.T @ cover) / oh.sum(0).clamp_min(1)[:, None]    # [Ne, n]
        P["G"], P["U"] = tWg[sets].clone(), tWu[sets].clone()         # [Ne, m, d]
        P["D"] = tWdT[sets] / mult.gather(1, sets).clamp_min(1.0)[:, :, None]   # [Ne, m, d]
        P["R"], P["tau"] = cent.clone(), torch.tensor(1.0)
    P["B"] = torch.zeros(d, d)
    for v in P.values():
        v.requires_grad_(True)

    def block(x):
        h = gelu(x @ P["Gs"].T) * (x @ P["Us"].T)
        if topk:                                                       # per-token top share by the current exact score
            sc = h.detach().abs() * P["Ds"].detach().norm(dim=1)
            kk = max(1, int(topk * h.shape[1]))
            thr = sc.kthvalue(h.shape[1] - kk + 1, dim=1).values[:, None]
            h = h * (sc >= thr)
        y = x @ P["B"] + h @ P["Ds"]
        if Ne:
            lg = (feat(x) @ P["R"].T) * P["tau"]
            top = lg.topk(K, dim=1)
            w = torch.softmax(top.values, 1) * K
            fe, fb, fw = top.indices.reshape(-1), torch.arange(len(x)).repeat_interleave(K), w.reshape(-1)
            order = torch.argsort(fe)
            fe, fb, fw = fe[order], fb[order], fw[order]
            bounds = torch.searchsorted(fe, torch.arange(Ne + 1))
            for e in range(Ne):
                s0, s1 = int(bounds[e]), int(bounds[e + 1])
                if s1 > s0:
                    rows = fb[s0:s1]
                    xe = x[rows]
                    ye = (gelu(xe @ P["G"][e].T) * (xe @ P["U"][e].T)) @ P["D"][e]
                    y = y.index_add(0, rows, ye * fw[s0:s1, None])
        return y

    @torch.no_grad()
    def evaluate(idx):
        Yh = torch.cat([block(Xt[idx[s:s + 2048]]) for s in range(0, len(idx), 2048)])
        return fvu(Yh, Yt[idx]), dirv(Yh, Yt[idx])

    with torch.no_grad():                                              # least-squares bypass on what the start misses
        E_a = torch.cat([Yt[isub[s:s + 2048]] - block(Xt[isub[s:s + 2048]]) for s in range(0, len(isub), 2048)])
        E_v = torch.cat([Yt[iva[s:s + 2048]] - block(Xt[iva[s:s + 2048]]) for s in range(0, len(iva), 2048)])
        P["B"].copy_(ridge(Xt[isub], E_a, Xt[iva], E_v))
        del E_a, E_v
    e0 = evaluate(ite_s)
    ref = oracle(min(0.5, max(active, 1 / 16))) if name != "control" else (0.0, 0.0)
    R["start"], R["oracle_same_size"] = e0, ref
    rep("closed-form start: error %.4f (dir %.4f). References on the same positions: linear map alone %.4f; "
        "dynamic oracle + bypass at the same active share %.4f (dir %.4f)" % (e0[0], e0[1], lin, ref[0], ref[1]))
    # ---- SGD
    lr_w, lr_r = 5e-5, 5e-4
    groups = [dict(params=[P[k] for k in P if k not in ("R", "tau")], lr=lr_w, base=lr_w)]
    if Ne:
        groups.append(dict(params=[P["R"], P["tau"]], lr=lr_r, base=lr_r))
    opt = torch.optim.Adam(groups, betas=(0.9, 0.99))
    budget = hours * 3600.0
    t_start, step, seen = time.time(), 0, 0
    perm, pp = itr[torch.randperm(len(itr))], 0
    curve = []
    next_eval = 0.0
    gf = torch.from_numpy(gam_f)
    gf_ok = gf.abs() > 1e-6
    while True:
        el = time.time() - t_start
        if el >= budget:
            break
        fr = el / budget
        sc_ = min(1.0, (step + 1) / 30) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * fr)))
        for g in opt.param_groups:
            g["lr"] = g["base"] * sc_
        if pp + a.batch > len(perm):
            perm, pp = itr[torch.randperm(len(itr))], 0
        bi = perm[pp:pp + a.batch]
        pp += a.batch
        x, y = Xt[bi], Yt[bi]
        na = int(a.aug * a.batch)
        if na:
            xi, xj = x[:na], Xt[itr[torch.randint(len(itr), (na,))]]
            al = 0.6 + 0.4 * torch.rand(na, 1)
            u = torch.where(gf_ok, (al * xi + (1 - al) * xj) / torch.where(gf_ok, gf, torch.ones_like(gf)), torch.zeros(()))
            u = u / torch.sqrt((u ** 2).mean(1, keepdim=True) + 1e-6)
            xm = u * gf
            x = torch.cat([xm, x[na:]])
            y = torch.cat([teacher(xm), y[na:]])
        yh = block(x)
        loss = (((yh - y) ** 2) * g2).sum() / ((y ** 2) * g2).sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        step += 1
        seen += len(x)
        if el >= next_eval:
            ev, et = evaluate(ite_s), evaluate(itr_s)
            curve.append(dict(step=step, samples=seen, hours=el / 3600, test=ev[0], test_dir=ev[1], train=et[0], loss=loss.item()))
            say("%s step %d  samples %.2fM  test %.4f (dir %.4f)  train(real) %.4f  loss %.4f  %.1f s/step"
                % (name, step, seen / 1e6, ev[0], ev[1], et[0], loss.item(), el / max(step, 1)))
            next_eval = el + max(240.0, budget / 14)
            R["curve"] = curve
            rep_partial()
    fin = evaluate(ite)
    fin_tr = evaluate(itr_s)
    R.update(final=fin, final_train=fin_tr, steps=step, samples=seen, sec_per_step=(time.time() - t_start) / max(step, 1),
             passes_over_real=seen * (1 - a.aug) / max(1, len(itr)), tau=float(P["tau"]) if Ne else None)
    rep("after %d steps (%.2f M samples, %.1f passes over the real positions, %.1f s/step): **held-out error %.4f (dir %.4f)**, "
        "on train positions %.4f" % (step, seen / 1e6, R["passes_over_real"], R["sec_per_step"], fin[0], fin[1], fin_tr[0]))
    rep("curve (hours: held-out / train): " + ", ".join("%.2f: %.4f / %.4f" % (c["hours"], c["test"], c["train"]) for c in curve))
    if name == "control":
        rep("control verdict: start %.4f -> %.4f (%s)" % (e0[0], fin[0], "recovered" if fin[0] < e0[0] / 5 else "NOT recovered"))


def rep_partial():
    json.dump(RES, open(os.path.join(a.out, "results_e8_%s.json" % TAG), "w"), indent=1)


specs = [(s.split(":")[0], float(s.split(":")[1])) for s in a.arms.split(",")]
if a.total_min > 0:        # fit the training time to what the dump left: everything must end total_min after start
    avail = a.total_min * 60.0 - (time.time() - T0) - 420.0 * len(specs)
    scale = max(0.15, avail / (3600.0 * sum(h for _, h in specs)))
    specs = [(nm, h * scale) for nm, h in specs]
    say("time budget: %.0f min used so far, arm hours scaled by %.2f -> %s"
        % ((time.time() - T0) / 60, scale, ", ".join("%s %.2f h" % x for x in specs)))
for nm, hrs in specs:
    run_arm(nm, hrs)
rep("")
rep("done in %.0f s" % (time.time() - T0))
