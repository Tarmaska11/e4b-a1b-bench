"""Reader for g4dump's binary output."""
import struct

import numpy as np


def parse_bin(path):
    """-> dict(layers, n_embd, last_only, rec_idx[int], tokens[list of int32 arrays], H[list of (n_layers, n_vec, n_embd) f32])"""
    out = dict(rec_idx=[], tokens=[], H=[], skipped=[])
    with open(path, "rb") as f:
        magic, n_embd, last_only, n_layers = struct.unpack("<4i", f.read(16))
        assert magic == 0x31443447, "bad magic"
        layers = list(struct.unpack("<%di" % n_layers, f.read(4 * n_layers)))
        out.update(layers=layers, n_embd=n_embd, last_only=bool(last_only))
        while True:
            h = f.read(12)
            if len(h) < 12:
                break
            r, n_tok, n_vec = struct.unpack("<3i", h)
            if n_tok < 0:
                out["skipped"].append((r, -n_tok - 1))
                continue
            toks = np.frombuffer(f.read(4 * n_tok), dtype="<i4").copy()
            vec = np.frombuffer(f.read(4 * n_vec * n_embd * n_layers), dtype="<f4").reshape(n_layers, n_vec, n_embd).copy()
            out["rec_idx"].append(r)
            out["tokens"].append(toks)
            out["H"].append(vec)
    return out


def save_npz(d, path):
    """Concatenate records; store float16 when it is safe, float32 otherwise."""
    if not d["H"]:
        np.savez(path, layers=np.array(d["layers"]), rec_idx=np.array([], dtype=np.int32))
        return
    H = np.concatenate(d["H"], axis=1)
    mx = float(np.abs(H).max())
    dtype = np.float16 if mx < 3.0e4 else np.float32
    np.savez(path,
             layers=np.array(d["layers"], dtype=np.int32),
             rec_idx=np.array(d["rec_idx"], dtype=np.int32),
             n_vec=np.array([h.shape[1] for h in d["H"]], dtype=np.int32),
             n_tok=np.array([len(t) for t in d["tokens"]], dtype=np.int32),
             tokens=np.concatenate(d["tokens"]),
             H=H.astype(dtype),
             absmax=np.array([mx], dtype=np.float32))
