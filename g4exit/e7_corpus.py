#!/usr/bin/env python3
"""Shared by the E7 scripts: the text corpus, the g4dump call, the dump reader, prefix de-duplication."""
import json
import os
import random
import struct
import subprocess
import time
import urllib.parse
import urllib.request

import numpy as np

PRE, SUF = "<|turn>user\n", "<turn|>\n<|turn>model\n"
SEP = "\n<<<G4SEP>>>\n"

WIKI = {
    "en": ["Photosynthesis", "World War II", "Jazz", "Black hole", "Roman Empire", "Machine learning", "Coffee",
           "Mount Everest", "William Shakespeare", "Association football", "Immune system", "Great Depression",
           "Volcano", "Chess", "Impressionism", "Internet", "Democracy", "Western honey bee", "Renaissance",
           "Electric car", "Plate tectonics", "Leonardo da Vinci", "Diabetes", "Stock market", "Buddhism",
           "Antarctica", "DNA", "Printing press", "Olympic Games", "Climate change", "Ancient Egypt",
           "Quantum mechanics", "Tea", "Amazon rainforest", "Mathematics", "Violin", "Vaccination", "Silk Road",
           "Bicycle", "Moon"],
    "ru": ["Фотосинтез", "Вторая мировая война", "Джаз", "Чёрная дыра", "Римская империя", "Машинное обучение",
           "Кофе", "Джомолунгма", "Пушкин, Александр Сергеевич", "Футбол", "Иммунная система",
           "Великая депрессия", "Вулкан", "Шахматы", "Импрессионизм", "Интернет", "Демократия", "Медоносная пчела",
           "Возрождение", "Электромобиль", "Тектоника плит", "Толстой, Лев Николаевич", "Сахарный диабет",
           "Фондовый рынок", "Буддизм", "Антарктида", "Дезоксирибонуклеиновая кислота", "Байкал",
           "Олимпийские игры", "Изменение климата", "Древний Египет", "Квантовая механика", "Чай",
           "Санкт-Петербург", "Математика", "Скрипка", "Вакцинация", "Великий шёлковый путь", "Велосипед", "Луна"],
}


def fetch_wiki(lang, title, clip, say=print):
    url = ("https://%s.wikipedia.org/w/api.php?action=query&format=json&prop=extracts&explaintext=1"
           "&exsectionformat=plain&redirects=1&titles=%s" % (lang, urllib.parse.quote(title)))
    for _ in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "e7-structure-test/1.0 (research script)"})
            with urllib.request.urlopen(req, timeout=60) as r:
                pages = json.loads(r.read().decode("utf-8"))["query"]["pages"]
            txt = next(iter(pages.values())).get("extract", "")
            return " ".join(txt.split())[:clip]
        except Exception as e:  # noqa: BLE001
            say("wiki retry", lang, title, repr(e)[:80])
            time.sleep(3)
    return ""


def build_corpus(bench, cal, nmmlu, nwiki, say=print, chunks=1):
    """list of dicts: kind ('mmlu' | 'wiki'), lang, text. MMLU prompts end at the start of the model turn;
    Wikipedia extracts sit INSIDE the model turn, so their positions are assistant-side prose.
    chunks > 1: each article gives that many consecutive pieces, one record each."""
    rows = []
    if cal and os.path.exists(cal):
        rows += [json.loads(l) for l in open(cal, encoding="utf-8") if l.strip()]
    more = [json.loads(l) for l in open(bench, encoding="utf-8") if l.strip()]
    random.Random(0).shuffle(more)
    rows += more
    en = [r for r in rows if r.get("lang") == "en"][:nmmlu // 2]
    ru = [r for r in rows if r.get("lang") == "ru"][:nmmlu - len(en)]
    out = [dict(kind="mmlu", lang=r["lang"], text=PRE + r["prompt"] + SUF) for r in en + ru]
    per = nwiki // 2
    for lang, ask, clip in (("en", "Tell me about %s.", 3000), ("ru", "Расскажи про «%s».", 2200)):
        got = 0
        for title in WIKI[lang][:per]:
            full = fetch_wiki(lang, title, clip * chunks, say)
            for c in range(chunks):
                txt = full[c * clip:(c + 1) * clip]
                if len(txt) > 400:
                    out.append(dict(kind="wiki", lang=lang, text=PRE + (ask % title) + SUF + txt))
                    got += 1
        say("wiki", lang, got, "extracts")
    random.Random(1).shuffle(out)
    return out


def run_dump(g4dump, model, texts, layers, names, out, tag, say=print):
    rf, ob = os.path.join(out, "recs_%s.txt" % tag), os.path.join(out, "dump_%s.bin" % tag)
    open(rf, "w", encoding="utf-8", newline="\n").write(SEP.join(t["text"] for t in texts) + SEP + "<<<END>>>")
    env = dict(os.environ, G4_LAYERS=layers, G4_NAMES=names, G4_OUT=ob, G4_LAST="0")
    t1 = time.time()
    with open(os.path.join(out, "dump_%s.log" % tag), "w") as lg:
        rc = subprocess.call([g4dump, "-m", model, "-f", rf, "--no-escape", "--no-repack", "-c", "2048", "-b", "2048",
                              "-t", str(os.cpu_count() or 4), "-ngl", "0"], env=env, stdout=lg, stderr=subprocess.STDOUT)
    say("dump rc", rc, "size %.2f GB in %.0f s" % (os.path.getsize(ob) / 1e9, time.time() - t1))
    assert rc == 0
    return ob


def read_dump(path):
    """-> d, [slot arrays (N, d) float32], token id, record index, position in record (one row per position)"""
    with open(path, "rb") as f:
        magic, d, last_only, n_slots = struct.unpack("<4i", f.read(16))
        f.read(4 * n_slots)
        recs = []
        while True:
            h = f.read(12)
            if len(h) < 12:
                break
            r, n_tok, n_vec = struct.unpack("<3i", h)
            if n_tok < 0:
                continue
            recs.append((r, n_tok, n_vec, f.tell()))
            f.seek(4 * n_tok + 4 * n_vec * d * n_slots, 1)
        N = sum(x[2] for x in recs)
        D = [np.zeros((N, d), np.float32) for _ in range(n_slots)]
        tok, rec, pos = np.zeros(N, np.int32), np.zeros(N, np.int32), np.zeros(N, np.int32)
        o = 0
        for r, n_tok, n_vec, off in recs:
            f.seek(off)
            t = np.frombuffer(f.read(4 * n_tok), dtype="<i4")
            for s in range(n_slots):
                D[s][o:o + n_vec] = np.frombuffer(f.read(4 * n_vec * d), dtype="<f4").reshape(n_vec, d)
            tok[o:o + n_vec] = t[n_tok - n_vec:]
            rec[o:o + n_vec] = r
            pos[o:o + n_vec] = np.arange(n_tok - n_vec, n_tok)
            o += n_vec
    return d, D, tok, rec, pos


def dedup_keep(tok, rec, pos, skip=4):
    """a position is kept once per distinct token prefix (identical prefix = identical activations); the first
    `skip` positions of every record (BOS and the turn header) are dropped"""
    keep = np.zeros(len(tok), bool)
    seen = set()
    h, last = 0, -1
    for i in range(len(tok)):
        if rec[i] != last or pos[i] == 0:
            h, last = 1469598103934665603, rec[i]
        h = ((h ^ int(tok[i])) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
        if h not in seen:
            seen.add(h)
            keep[i] = pos[i] >= skip
    return keep


def build_corpus_e2b(bench, chat):
    """the exact population of E2b: 200 benchmark prompts + the chat prompts, every one ending at the model turn"""
    rows = [json.loads(l) for l in open(bench, encoding="utf-8") if l.strip()]
    rows = random.Random(0).sample(rows, min(200, len(rows)))
    out = [dict(kind="mmlu", lang=r.get("lang", "?"), text=PRE + r["prompt"] + SUF) for r in rows]
    out += [dict(kind="chat", lang=json.loads(l).get("lang", "?"), text=PRE + json.loads(l)["prompt"] + SUF)
            for l in open(chat, encoding="utf-8") if l.strip()]
    random.Random(1).shuffle(out)
    return out
