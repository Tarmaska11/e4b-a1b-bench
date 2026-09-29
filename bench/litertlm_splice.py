#!/usr/bin/env python
"""Replace one section of a .litertlm in place, rewriting the container header.

WHY THIS EXISTS. graphtools could only ever LIST sections (litertlm_sections.py,
sections.py). The prior recipe for a clamped model was ad-hoc: extract
prefill_decode, run ext_pack.py, splice the bytes back by hand. That is fine
exactly once. It is not fine now, because a STADIUM clamp GROWS the flatbuffer
structure (step 4 inserts SLICE ops and clones shape consts), so the replacement
section is a different length and every following section moves. ext_pack.py
already handles the inside of a .tflite -- shifting the external weight region and
patching every Buffer.offset -- but nothing handled the OUTSIDE, i.e. the
container header byte ranges.

THE CONTAINER, as far as this tool needs it:

    0        magic LITERTLM              8 bytes
    8        (unused here)              16 bytes
    24       header_end                  u64
    32       header flatbuffer          ... up to header_end
    ...      zero padding
    <begin>  section payloads, each at its own aligned offset

Each section object carries beginOffset (field 6) and endOffset (field 8) as u64.
Both are PATCHED IN PLACE: the header flatbuffer is ~800 bytes inside a 16 KB
reserved block, so rewriting two u64 scalars changes no sizes and needs no
re-serialisation. That is the whole trick, and it is why this is safe.

OFFSETS INSIDE A TFLITE SECTION ARE SECTION-RELATIVE, so a section extracted to a
standalone .tflite is valid as-is and can be spliced back at any aligned offset.
The extract command ASSERTS this rather than assuming it.

Usage:
    python litertlm_splice.py list    <in.litertlm>
    python litertlm_splice.py extract <in.litertlm> <index> <out.tflite>
    python litertlm_splice.py splice  <in.litertlm> <index> <new.tflite> <out.litertlm>
"""
import os
import struct
import sys

import flatbuffers
from flatbuffers import table as fbtable, number_types as N

MAGIC = b"LITERTLM"
FB_BASE = 32
CHUNK = 1 << 23
DTYPE = {0: "NONE", 1: "GenericBinaryData", 2: "Deprecated", 3: "TFLiteModel",
         4: "SP_Tokenizer", 5: "LlmMetadataProto", 6: "HF_Tokenizer_Zlib",
         7: "TFLiteWeights"}


def _root(b):
    return fbtable.Table(b, flatbuffers.encode.Get(N.UOffsetTFlags.packer_type, b, 0))


def _sub(t, v):
    o = t.Offset(v)
    return None if o == 0 else fbtable.Table(t.Bytes, t.Indirect(o + t.Pos))


def _vec(t, v):
    o = t.Offset(v)
    if o == 0:
        return []
    base = t.Vector(o)
    return [fbtable.Table(t.Bytes, t.Indirect(base + i * 4)) for i in range(t.VectorLen(o))]


def _str(t, v):
    o = t.Offset(v)
    return None if o == 0 else t.String(o + t.Pos)


def header(path):
    with open(path, "rb") as f:
        assert f.read(8) == MAGIC, "not a litertlm container"
        f.read(16)
        (he,) = struct.unpack("<Q", f.read(8))
        f.seek(FB_BASE)
        fb = f.read(he - FB_BASE)
    return he, bytearray(fb)


def sections(path):
    """Every section, with the ABSOLUTE file position of its begin/end u64 fields."""
    he, fb = header(path)
    sm = _sub(_root(fb), 6)
    out = []
    for so in _vec(sm, 4):
        ob, oe = so.Offset(6), so.Offset(8)
        assert ob and oe, "section without begin/end"
        odt = so.Offset(10)
        mt = None
        for kvp in _vec(so, 4):
            k = _str(kvp, 4)
            if k and k.decode() == "model_type":
                vt = _sub(kvp, 8)
                s = _str(vt, 4) if vt else None
                mt = s.decode() if s else None
        out.append(dict(
            begin=so.Get(N.Uint64Flags, ob + so.Pos),
            end=so.Get(N.Uint64Flags, oe + so.Pos),
            dtype=so.Get(N.Uint8Flags, odt + so.Pos) if odt else 0,
            model_type=mt,
            begin_field=FB_BASE + so.Pos + ob,
            end_field=FB_BASE + so.Pos + oe,
        ))
    return he, out


def _align_of(x, cap=1 << 16):
    a = 1
    while a < cap and x % (a * 2) == 0:
        a *= 2
    return a


def _align_up(x, a):
    return (x + a - 1) // a * a


def cmd_list(path):
    he, secs = sections(path)
    print("%s  %d bytes  header_end=%d" % (path, os.path.getsize(path), he))
    for i, s in enumerate(secs):
        n = s["end"] - s["begin"]
        print("  [%d] %-18s begin=%12d end=%12d size=%12d (%8.2f MiB) align=%6d  %s"
              % (i, DTYPE.get(s["dtype"], s["dtype"]), s["begin"], s["end"], n,
                 n / 2 ** 20, _align_of(s["begin"]), s["model_type"] or ""))
    return secs


def _copy(fin, fout, begin, n):
    fin.seek(begin)
    left = n
    while left > 0:
        b = fin.read(min(CHUNK, left))
        assert b, "short read"
        fout.write(b)
        left -= len(b)


def _assert_section_relative(tfl, n):
    """A TFLite section must carry offsets relative to ITS OWN start, or splicing it
    back at a different container offset would silently point every weight buffer at
    the wrong bytes. Check it instead of trusting it."""
    with open(tfl, "rb") as f:
        prefix = f.read(min(200 * 1024 * 1024, n))
    root = _root(bytearray(prefix))
    lo, hi, cnt = None, 0, 0
    for bt in _vec(root, 12):
        oo = bt.Offset(6)
        if oo == 0:
            continue
        off = bt.Get(N.Uint64Flags, oo + bt.Pos)
        if off is None or off <= 1:
            continue
        so = bt.Offset(8)
        sz = bt.Get(N.Uint64Flags, so + bt.Pos) if so else 0
        lo = off if lo is None else min(lo, off)
        hi = max(hi, off + sz)
        cnt += 1
    if cnt == 0:
        print("    (no external buffers; nothing to check)")
        return
    assert 0 < lo < n and hi <= n, (
        "external buffer offsets are NOT section-relative: min=%d max_end=%d len=%d"
        % (lo, hi, n))
    print("    external buffers: %d, offsets in [%d, %d) within %d -- section-relative OK"
          % (cnt, lo, hi, n))


def cmd_extract(path, idx, out):
    _he, secs = sections(path)
    s = secs[idx]
    n = s["end"] - s["begin"]
    with open(path, "rb") as fin, open(out, "wb") as fout:
        _copy(fin, fout, s["begin"], n)
    print("extracted section %d (%s) -> %s  %d bytes" % (idx, s["model_type"], out, n))
    if s["dtype"] == 3:
        _assert_section_relative(out, n)
    return out


def cmd_splice(path, idx, newfile, out):
    he, secs = sections(path)
    first = min(s["begin"] for s in secs)
    for s in secs:
        assert s["begin_field"] < first and s["end_field"] < first, \
            "a section offset field lies outside the header block"

    new_len = os.path.getsize(newfile)
    old_len = secs[idx]["end"] - secs[idx]["begin"]
    layout, cur = [], 0
    for i, s in enumerate(secs):
        if i < idx:
            nb, ne = s["begin"], s["end"]
        elif i == idx:
            nb, ne = s["begin"], s["begin"] + new_len
        else:
            nb = _align_up(cur, _align_of(s["begin"]))
            ne = nb + (s["end"] - s["begin"])
        layout.append((nb, ne))
        cur = ne

    with open(path, "rb") as f:
        hdr = bytearray(f.read(first))
    for s, (nb, ne) in zip(secs, layout):
        hdr[s["begin_field"]:s["begin_field"] + 8] = struct.pack("<Q", nb)
        hdr[s["end_field"]:s["end_field"] + 8] = struct.pack("<Q", ne)

    with open(path, "rb") as fin, open(newfile, "rb") as fnew, open(out, "wb") as fout:
        fout.write(hdr)
        for i, (s, (nb, ne)) in enumerate(zip(secs, layout)):
            pad = nb - fout.tell()
            assert pad >= 0, "layout went backwards at section %d" % i
            fout.write(b"\x00" * pad)
            if i == idx:
                fnew.seek(0)
                left = new_len
                while left > 0:
                    b = fnew.read(min(CHUNK, left))
                    assert b, "short read"
                    fout.write(b)
                    left -= len(b)
            else:
                _copy(fin, fout, s["begin"], s["end"] - s["begin"])
            assert fout.tell() == ne, \
                "section %d ended at %d, expected %d" % (i, fout.tell(), ne)
        total = fout.tell()

    print("spliced section %d: %d -> %d bytes (%+d)"
          % (idx, old_len, new_len, new_len - old_len))
    print("wrote %s: %d bytes (orig %d, %+d)"
          % (out, total, os.path.getsize(path), total - os.path.getsize(path)))
    _he2, secs2 = sections(out)
    for i, (s2, (nb, ne)) in enumerate(zip(secs2, layout)):
        assert (s2["begin"], s2["end"]) == (nb, ne), \
            "header round-trip failed at section %d" % i
    assert secs2[-1]["end"] == total, \
        "last section end %d != file size %d" % (secs2[-1]["end"], total)
    print("header round-trip OK: %d sections, last end == file size" % len(secs2))
    return out


if __name__ == "__main__":
    c = sys.argv[1] if len(sys.argv) > 1 else ""
    if c == "list":
        cmd_list(sys.argv[2])
    elif c == "extract":
        cmd_extract(sys.argv[2], int(sys.argv[3]), sys.argv[4])
    elif c == "splice":
        cmd_splice(sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5])
    else:
        sys.exit(__doc__)
