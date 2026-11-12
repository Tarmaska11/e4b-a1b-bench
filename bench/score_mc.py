#!/usr/bin/env python3
"""score_mc.py -- one scorer for every harness that answered the 1 140 Global-MMLU prompts.

usage: score_mc.py <name>=<file.jsonl> [<name>=<file.jsonl> ...] [--ref <name>]

A file is either replies (`reply` field: the phone's BenchMc, the llama.cpp shards) or the Kaggle
harness's rows (`pred` = restricted A-D argmax, `free` = free first token; both are scored, as
<name>:restricted and <name>:free). The SAME parser reads every free-text reply, so systems that
answered in text are compared like for like. Reports accuracy per language with a 95 % Wilson
interval, the letter-answer rate, and -- against --ref -- the paired difference on the questions
both answered (net wins, and an exact McNemar sign-test p-value).
"""
import json
import math
import re
import sys

CYR = {"А": "A", "В": "B", "С": "C", "Д": "D"}


def parse(reply):
    if reply is None:
        return None
    s = str(reply).strip()
    m = re.search(r"(?:answer|ответ)\s*(?:is|:|—|-)?\s*\**\(?([A-DАВСД])\b", s, re.I)
    if m:
        return CYR.get(m.group(1).upper(), m.group(1).upper())
    s = s.lstrip("*#>\"'`«( \n\t")
    if s[:1] in ("A", "B", "C", "D"):
        return s[0]
    if s[:1] in CYR:
        return CYR[s[0]]
    return None


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def binom_two_sided(k, n):
    """exact two-sided sign test p for k successes of n at p=0.5"""
    if n == 0:
        return 1.0
    from math import comb
    tail = sum(comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def load(name, path):
    out = {}
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    if rows and "reply" in rows[0]:
        sysrows = {name: {}}
        for r in rows:
            key = r.get("key") or (r["id"] + "|" + r["lang"])
            sysrows[name][key] = (r["lang"], r.get("gold"), parse(r["reply"]))
        return sysrows
    res, fre = {}, {}
    for r in rows:
        key = r["id"] + "|" + r["lang"]
        res[key] = (r["lang"], r["gold"], r.get("pred"))
        f = r.get("free")
        fre[key] = (r["lang"], r["gold"], CYR.get(f, f) if f in ("A", "B", "C", "D", "А", "В", "С") else None)
    out[name + ":restricted"] = res
    out[name + ":free"] = fre
    return out


def main():
    args = sys.argv[1:]
    ref = None
    if "--ref" in args:
        i = args.index("--ref")
        ref = args[i + 1]
        args = args[:i] + args[i + 2:]
    systems = {}
    for a in args:
        name, path = a.split("=", 1)
        systems.update(load(name, path))
    print("%-28s %-4s %6s %7s %15s %7s" % ("system", "lang", "n", "acc", "95% CI", "letter"))
    for name, rows in systems.items():
        for lang in ("en", "ru"):
            rr = [v for v in rows.values() if v[0] == lang]
            if not rr:
                continue
            k = sum(1 for _, g, p in rr if p == g)
            lo, hi = wilson(k, len(rr))
            let = sum(1 for _, _, p in rr if p is not None) / len(rr)
            print("%-28s %-4s %6d %6.1f%% %6.1f-%5.1f%% %6.1f%%" % (name, lang, len(rr), 100 * k / len(rr),
                                                                   100 * lo, 100 * hi, 100 * let))
    if ref:
        R = systems[ref]
        print("\npaired vs %s (same questions):" % ref)
        for name, rows in systems.items():
            if name == ref:
                continue
            for lang in ("en", "ru"):
                keys = [k for k in rows if k in R and rows[k][0] == lang]
                if not keys:
                    continue
                a = sum(1 for k in keys if rows[k][2] == rows[k][1] and R[k][2] != R[k][1])
                b = sum(1 for k in keys if rows[k][2] != rows[k][1] and R[k][2] == R[k][1])
                d = 100.0 * (a - b) / len(keys)
                print("  %-28s %-4s n=%4d  %+5.1f pts  (wins %d / losses %d, sign-test p=%.3f)" %
                      (name, lang, len(keys), d, a, b, binom_two_sided(a, a + b)))


if __name__ == "__main__":
    main()
