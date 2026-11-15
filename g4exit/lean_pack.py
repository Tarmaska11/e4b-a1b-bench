#!/usr/bin/env python3
"""lean_pack.py --base <Q4_K_M gguf, path> --donor <Q4_K_S gguf, path or URL> --overlay out.gguf [--full out_full.gguf]

Lean-A of Gemma 4 E4B: the same model with fewer bytes read per token. No weight is re-derived with a new method:
  * every tensor the donor file stores in a narrower type than the base (ffn_down of 21 layers, attn_v) is taken
    from the donor BIT FOR BIT (the donor is unsloth's own Q4_K_S, measured equivalent to Q4_K_M on 11-13);
  * blk.N.inp_gate / blk.N.proj (F32) and per_layer_model_proj (BF16) are stored as Q8_0 (gguf-py's quantiser).

--overlay : a small GGUF holding ONLY the replaced tensors (the research engine reads it with G4_OVERLAY=<file>;
            it is also the small download).
--full    : a complete GGUF (base metadata, replaced tensors swapped in) that any llama.cpp loads. Needs --base local.

A remote donor is read with HTTP range requests: only the header and the tensors that are taken are downloaded.
"""
import argparse
import hashlib
import struct
import sys
import urllib.request

import numpy as np

import gguf
from gguf import GGMLQuantizationType as QT

SIZE = {QT.Q4_K: 144, QT.Q5_K: 176, QT.Q6_K: 210, QT.Q8_0: 34, QT.F32: 4, QT.F16: 2, QT.BF16: 2}
BLCK = {QT.Q4_K: 256, QT.Q5_K: 256, QT.Q6_K: 256, QT.Q8_0: 32, QT.F32: 1, QT.F16: 1, QT.BF16: 1}


class Src:
    """Random access to a local file or an HTTP(S) URL."""

    def __init__(self, loc):
        self.loc, self.remote = loc, loc.startswith("http")
        self.f = None if self.remote else open(loc, "rb")
        self.fetched = 0

    def read(self, off, n):
        if not self.remote:
            self.f.seek(off)
            return self.f.read(n)
        out = b""
        while len(out) < n:
            a = off + len(out)
            b = min(off + n, a + (32 << 20)) - 1
            for attempt in range(6):
                try:
                    req = urllib.request.Request(self.loc, headers={"Range": f"bytes={a}-{b}", "User-Agent": "lean-pack"})
                    with urllib.request.urlopen(req, timeout=180) as r:
                        chunk = r.read()
                    break
                except Exception as e:  # noqa: BLE001
                    print(f"  retry {attempt + 1} on bytes {a}-{b}: {e}", file=sys.stderr)
            else:
                raise RuntimeError("range request failed")
            out += chunk
            self.fetched += len(chunk)
        return out


class Header:
    """GGUF header: tensor name -> (dims in GGUF order, type id, absolute file offset, bytes)."""

    def __init__(self, loc):
        self.src = Src(loc)
        self.buf, self.p = b"", 0
        assert self.take(4) == b"GGUF", "not a GGUF file"
        _ver, nt, nkv = self.u32(), self.u64(), self.u64()
        self.kv = {}
        for _ in range(nkv):
            k = self.string()
            self.kv[k] = self.value(self.u32())
        ti = []
        for _ in range(nt):
            name = self.string()
            dims = [self.u64() for _ in range(self.u32())]
            ti.append((name, dims, self.u32(), self.u64()))
        align = self.kv.get("general.alignment", 32)
        self.data0 = (self.p + align - 1) // align * align
        self.t = {}
        self.order = []
        for name, dims, t, off in ti:
            qt = QT(t)
            n = 1
            for d in dims:
                n *= d
            nbytes = n // BLCK[qt] * SIZE[qt] if qt in SIZE else None
            self.t[name] = (dims, qt, self.data0 + off, nbytes)
            self.order.append(name)

    def need(self, n):
        while len(self.buf) < n:
            self.buf += self.src.read(len(self.buf), max(4 << 20, n - len(self.buf)))

    def take(self, n):
        self.need(self.p + n)
        b = self.buf[self.p:self.p + n]
        self.p += n
        return b

    def u32(self): return struct.unpack("<I", self.take(4))[0]
    def u64(self): return struct.unpack("<Q", self.take(8))[0]
    def string(self): return self.take(self.u64()).decode("utf-8", "replace")

    def value(self, t):
        fmt = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
        if t in fmt:
            return struct.unpack(fmt[t], self.take(struct.calcsize(fmt[t])))[0]
        if t == 8:
            return self.string()
        if t == 9:
            et, n = self.u32(), self.u64()
            if et in fmt:
                self.take(struct.calcsize(fmt[et]) * n)
                return ("array", n)
            for _ in range(n):
                self.value(et)
            return ("array", n)
        raise ValueError(f"type {t}")

    def raw(self, name):
        dims, qt, off, nbytes = self.t[name]
        return self.src.read(off, nbytes)


def byte_shape(dims, qt):
    """numpy shape of the raw bytes of a quantised tensor whose GGUF dims are `dims`."""
    shp = list(reversed(dims))
    shp[-1] = shp[-1] // BLCK[qt] * SIZE[qt]
    return tuple(shp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--donor", required=True)
    ap.add_argument("--overlay", required=True)
    ap.add_argument("--full")
    a = ap.parse_args()

    base, donor = Header(a.base), Header(a.donor)
    repl = {}  # name -> (uint8 array in byte shape, QT, note)
    saved = 0
    for name in base.order:
        dims, qt, _off, nbytes = base.t[name]
        if name in donor.t and donor.t[name][1] != qt:
            ddims, dqt, _doff, dn = donor.t[name]
            assert ddims == dims, name
            if dn >= nbytes:
                continue  # only take narrower tensors
            raw = np.frombuffer(donor.raw(name), dtype=np.uint8).reshape(byte_shape(dims, dqt))
            repl[name] = (raw, dqt, f"{qt.name} -> {dqt.name} (donor bytes)")
            saved += nbytes - dn
        elif (name.endswith(".inp_gate.weight") or name.endswith(".proj.weight") or name == "per_layer_model_proj.weight") \
                and qt in (QT.F32, QT.BF16, QT.F16):
            buf = base.raw(name)
            shp = tuple(reversed(dims))
            if qt == QT.F32:
                x = np.frombuffer(buf, dtype="<f4").reshape(shp)
            elif qt == QT.F16:
                x = np.frombuffer(buf, dtype="<f2").astype(np.float32).reshape(shp)
            else:
                x = (np.frombuffer(buf, dtype="<u2").astype(np.uint32) << 16).view(np.float32).reshape(shp)
            q = gguf.quants.quantize(np.ascontiguousarray(x), QT.Q8_0)
            assert q.dtype == np.uint8 and q.shape == byte_shape(dims, QT.Q8_0), (name, q.shape)
            back = gguf.quants.dequantize(q, QT.Q8_0)
            err = float(((back - x) ** 2).sum() / max((x ** 2).sum(), 1e-30))
            repl[name] = (q, QT.Q8_0, f"{qt.name} -> Q8_0, squared relative error {err:.2e}")
            saved += nbytes - q.nbytes
    assert repl, "nothing to replace"
    kinds = {}
    for name, (arr, qt, note) in repl.items():
        key = ".".join(p for i, p in enumerate(name.split(".")) if not (i == 1 and name.startswith("blk.")))
        k = kinds.setdefault(key, [0, 0, set(), []])
        k[0] += 1
        k[1] += arr.nbytes
        k[2].add(note.split(",")[0])
        if "error" in note:
            k[3].append(float(note.split()[-1]))
    print(f"replaced {len(repl)} tensors, {saved / 2**20:.1f} MiB fewer in the file; remote bytes fetched: "
          f"{(base.src.fetched + donor.src.fetched) / 2**20:.1f} MiB")
    for key, (n, b, notes, errs) in kinds.items():
        e = f", worst squared relative error {max(errs):.2e}" if errs else ""
        print(f"  {key:34s} x{n:<3d} {b / 2**20:8.1f} MiB  {'; '.join(sorted(notes))}{e}")

    # ---- overlay: only the replaced tensors
    w = gguf.GGUFWriter(a.overlay, "gemma4-overlay")
    w.add_string("general.name", "Gemma 4 E4B lean-A overlay (replacement tensors for gemma-4-E4B-it-Q4_K_M)")
    w.add_uint32("g4.overlay.version", 1)
    names = [n for n in base.order if n in repl]
    for n in names:
        arr, qt, _ = repl[n]
        w.add_tensor_info(n, arr.shape, arr.dtype, arr.nbytes, qt)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    for n in names:
        w.write_tensor_data(repl[n][0])
    w.close()
    h = hashlib.sha256()
    with open(a.overlay, "rb") as f:
        for blk in iter(lambda: f.read(1 << 22), b""):
            h.update(blk)
    print(f"overlay written: {a.overlay}  sha256 {h.hexdigest()}")

    # ---- full file
    if a.full:
        assert not a.base.startswith("http"), "--full needs a local base"
        rd = gguf.GGUFReader(a.base)
        arch = rd.fields[gguf.Keys.General.ARCHITECTURE].contents()
        fw = gguf.GGUFWriter(a.full, arch)
        for field in rd.fields.values():
            if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
                continue
            vt = field.types[0]
            st = field.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
            fw.add_key_value(field.name, field.contents(), vt, sub_type=st)
        for t in rd.tensors:
            if t.name in repl:
                arr, qt, _ = repl[t.name]
                fw.add_tensor_info(t.name, arr.shape, arr.dtype, arr.nbytes, qt)
            else:
                fw.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
        fw.write_header_to_file()
        fw.write_kv_data_to_file()
        fw.write_ti_data_to_file()
        for t in rd.tensors:
            fw.write_tensor_data(repl[t.name][0] if t.name in repl else t.data)
        fw.close()
        print(f"full file written: {a.full}")


if __name__ == "__main__":
    main()
