#!/usr/bin/env python
"""Drop whole sections from a .litertlm (Track A of research/e4b-a1b/PLAN_2026-10.md).

WHY. Gemma 4 E4B ships the token embedding twice -- the 171 MB tf_lite_embedder section and the LM
head inside tf_lite_prefill_decode (same bytes, same row scales) -- and 110 MB of audio sections the
app never loads (voice goes through its own ASR). Engine code-mod 0084 (CB_SHAREDEMB) reads the
embedding from the LM head when tf_lite_embedder is absent, so both can go: -281 MB of file.

HOW. The header is edited IN PLACE, never re-serialised: the section-metadata `objects` vector is
compacted (its count and its uoffset elements rewritten to reference only the kept SectionObject
tables; the dropped tables stay behind as unreferenced bytes), and each kept section's
begin_offset/end_offset u64 fields are patched, exactly as litertlm_splice.py patches them. Kept
sections are copied byte for byte, in their original order, each at a 16 KiB-aligned offset. Offsets
inside a TFLite section are section-relative, so moving a section changes nothing inside it.

THE STOCK ENGINE CANNOT READ THE RESULT (it has no 0084): Google's Gallery/Maven runtime, our API-29
build until it is rebuilt with 0084, and any app path that falls back to them need the full file.

RESTORE A DROPPED SECTION LATER (e.g. audio): splice it back from the full file with
litertlm_splice.py, or rebuild from the full file with a shorter --drop list.

Usage:
    python litertlm_slim.py <in.litertlm> <out.litertlm> [--drop a,b,...] [--no-verify]
    default --drop: tf_lite_embedder,tf_lite_audio_encoder_hw,tf_lite_audio_adapter,tf_lite_end_of_audio
"""
import hashlib
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from litertlm_splice import FB_BASE, _root, _sub, header, sections  # noqa: E402

DEFAULT_DROP = ("tf_lite_embedder", "tf_lite_audio_encoder_hw", "tf_lite_audio_adapter",
                "tf_lite_end_of_audio")
ALIGN = 16384
CHUNK = 1 << 23


def _align_up(x, a=ALIGN):
    return (x + a - 1) // a * a


def _md5_range(path, begin, n):
    h = hashlib.md5()
    with open(path, "rb") as f:
        f.seek(begin)
        left = n
        while left > 0:
            b = f.read(min(CHUNK, left))
            assert b, "short read"
            h.update(b)
            left -= len(b)
    return h.hexdigest()


def slim(src, dst, drop, verify=True):
    he, secs = sections(src)
    first = min(s["begin"] for s in secs)
    assert he <= first, "header runs into the first section"
    for s in secs:
        assert s["begin_field"] < first and s["end_field"] < first, "offset field outside header"
    names = [s["model_type"] for s in secs]
    missing = [d for d in drop if d not in names]
    assert not missing, "not in %s: %s" % (src, missing)
    keep = [i for i, s in enumerate(secs) if s["model_type"] not in drop]
    assert keep, "nothing left"
    assert "tf_lite_prefill_decode" in [secs[i]["model_type"] for i in keep], "main section dropped"

    # Locate the objects vector of the section metadata (same path as sections()).
    _he, fb = header(src)
    sm = _sub(_root(fb), 6)
    o = sm.Offset(4)
    vec = sm.Vector(o)                      # position of element 0, relative to FB_BASE
    count = sm.VectorLen(o)
    assert count == len(secs)
    table_pos = [vec + 4 * i + struct.unpack_from("<I", fb, vec + 4 * i)[0] for i in range(count)]

    # New layout.
    layout, cur = [], secs[keep[0]]["begin"]
    for i in keep:
        s = secs[i]
        nb = _align_up(cur) if layout else s["begin"]
        n = s["end"] - s["begin"]
        layout.append((i, nb, nb + n))
        cur = nb + n

    with open(src, "rb") as fin:
        head = bytearray(fin.read(first))
    # Compact the vector: element j references kept table keep[j].
    struct.pack_into("<I", head, FB_BASE + vec - 4, len(keep))
    for j, i in enumerate(keep):
        el = vec + 4 * j
        rel = table_pos[i] - el
        assert rel > 0, "a kept table sits before its vector slot"
        struct.pack_into("<I", head, FB_BASE + el, rel)
    for i, nb, ne in layout:
        struct.pack_into("<Q", head, secs[i]["begin_field"], nb)
        struct.pack_into("<Q", head, secs[i]["end_field"], ne)

    with open(src, "rb") as fin, open(dst, "wb") as fout:
        fout.write(head)
        for i, nb, ne in layout:
            s = secs[i]
            pos = fout.tell()
            assert pos <= nb
            fout.write(b"\0" * (nb - pos))
            fin.seek(s["begin"])
            left = ne - nb
            while left > 0:
                b = fin.read(min(CHUNK, left))
                assert b, "short read"
                fout.write(b)
                left -= len(b)
    src_size, dst_size = os.path.getsize(src), os.path.getsize(dst)
    print("%s: %d bytes -> %s: %d bytes (%.1f MB less)" % (src, src_size, dst, dst_size,
                                                          (src_size - dst_size) / 1e6))
    for i in range(len(secs)):
        if i not in keep:
            s = secs[i]
            print("  dropped [%d] %-28s %12d bytes" % (i, s["model_type"], s["end"] - s["begin"]))

    # Re-read the result through the same parser and check every kept section.
    _he2, secs2 = sections(dst)
    assert [s["model_type"] for s in secs2] == [secs[i]["model_type"] for i in keep], "section list"
    for (i, nb, ne), s2 in zip(layout, secs2):
        assert (s2["begin"], s2["end"]) == (nb, ne), "offsets not written"
        assert s2["dtype"] == secs[i]["dtype"], "data type changed"
    if verify:
        for (i, nb, ne), s2 in zip(layout, secs2):
            a = _md5_range(src, secs[i]["begin"], ne - nb)
            b = _md5_range(dst, nb, ne - nb)
            assert a == b, "section %s differs" % secs[i]["model_type"]
            print("  kept  [%d] %-28s %12d bytes at %12d  md5 %s  identical" % (
                i, secs[i]["model_type"] or "-", ne - nb, nb, a))
    return layout


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    drop = DEFAULT_DROP
    for a in sys.argv[1:]:
        if a.startswith("--drop="):
            drop = tuple(x for x in a[len("--drop="):].split(",") if x)
    if len(args) != 2:
        raise SystemExit(__doc__)
    slim(args[0], args[1], drop, verify="--no-verify" not in sys.argv)
