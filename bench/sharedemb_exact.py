"""Is the engine's shared-embedding lookup (cb_sharedemb.inc) bit-exact with Google's embedder model?

Gemma 4 E4B's .litertlm stores the token embedding twice: a `tf_lite_embedder` section (an INT2
table, one float32 scale per row, times float32(sqrt(2560)), ids outside the vocabulary mapped to
row 0) and the LM head inside `tf_lite_prefill_decode`. An engine patch drops the first and reads
rows from the second with cb_sharedemb.inc. This checks that code against Google's own model:

1. cb_sharedemb::FindInt2Head locates the LM head in the main section and the table in the
   embedder section; their codes and row scales are compared byte for byte.
2. For EVERY id 0..V-1, plus -1, V and V+12345, Google's embedder section run by the LiteRT
   interpreter is compared with cb_sharedemb::DecodeRow over the MAIN section's LM head.
   (-1 never reaches the embedder in the engine -- it returns the row-0 default -- so for -1 the
   reference is row 0 as well.)
3. Prints an FNV-1a 64 hash over all V decoded rows; the phone test prints the same hash.

Usage: sharedemb_exact.py <embedder.tflite> <main.tflite> <libsharedemb.so>
"""
import ctypes
import sys
import time

import numpy as np

EMB, MAIN, LIB = sys.argv[1], sys.argv[2], sys.argv[3]
lib = ctypes.CDLL(LIB)
lib.se_find_head.restype = ctypes.c_int
lib.se_find_head.argtypes = [ctypes.c_void_p, ctypes.c_uint64] + [ctypes.POINTER(ctypes.c_int)] * 3 + \
    [ctypes.POINTER(ctypes.c_uint64)] * 2 + [ctypes.c_char_p, ctypes.c_int]
lib.se_normaliser.restype = ctypes.c_float
lib.se_normaliser.argtypes = [ctypes.c_int]
lib.se_decode_rows.restype = None
lib.se_decode_rows.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_float,
                               ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
lib.se_fnv_all.restype = ctypes.c_uint64
lib.se_fnv_all.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_float]


def find(arr, label):
    v, d, r = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
    co, so = ctypes.c_uint64(), ctypes.c_uint64()
    why = ctypes.create_string_buffer(256)
    ok = lib.se_find_head(arr.ctypes.data, arr.size, ctypes.byref(v), ctypes.byref(d), ctypes.byref(r),
                          ctypes.byref(co), ctypes.byref(so), why, 256)
    if not ok:
        sys.exit("%s: FindInt2Head failed: %s" % (label, why.value.decode()))
    print("%s: INT2 table vocab %d dim %d refs %d codes at +%d scales at +%d" % (
        label, v.value, d.value, r.value, co.value, so.value))
    return v.value, d.value, co.value, so.value


main = np.memmap(MAIN, dtype=np.uint8, mode="r")
emb = np.memmap(EMB, dtype=np.uint8, mode="r")
V, D, mco, mso = find(main, "main section (LM head)")
EV, ED, eco, eso = find(emb, "embedder section")
assert (V, D) == (EV, ED), "shapes differ"
nbytes = V * D // 4
same_codes = np.array_equal(main[mco:mco + nbytes], emb[eco:eco + nbytes])
same_scales = np.array_equal(main[mso:mso + 4 * V], emb[eso:eso + 4 * V])
print("codes identical: %s; row scales identical: %s" % (same_codes, same_scales))

scales = np.frombuffer(main[mso:mso + 4 * V].tobytes(), dtype="<f4").copy()
mul = lib.se_normaliser(D)
print("normaliser %r bits 0x%08x" % (mul, np.float32(mul).view(np.uint32)))
codes_addr = main.ctypes.data + mco

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    from tflite_runtime.interpreter import Interpreter
it = Interpreter(model_path=EMB)
it.allocate_tensors()
inp, out = it.get_input_details()[0], it.get_output_details()[0]
print("embedder input", inp["shape"], inp["dtype"], "output", out["shape"], out["dtype"])

ids = list(range(V)) + [-1, V, V + 12345]
BATCH = 4096
rows = same = 0
worst = 0.0
t0 = time.time()
for b0 in range(0, len(ids), BATCH):
    chunk = ids[b0:b0 + BATCH]
    toks = np.array(chunk, dtype=np.int32)
    nat = np.empty((len(chunk), D), dtype=np.float32)
    lib.se_decode_rows(codes_addr, scales.ctypes.data, V, D, mul, toks.ctypes.data, len(chunk), nat.ctypes.data)
    for i, tok in enumerate(chunk):
        it.set_tensor(inp["index"], np.array(0 if tok < 0 else tok, dtype=inp["dtype"]).reshape(inp["shape"]))
        it.invoke()
        ref = it.get_tensor(out["index"]).reshape(-1)
        rows += 1
        if np.array_equal(ref.view(np.uint32), nat[i].view(np.uint32)):
            same += 1
        else:
            worst = max(worst, float(np.max(np.abs(ref.astype(np.float64) - nat[i].astype(np.float64)))))
    if b0 % (BATCH * 16) == 0:
        print("  %d/%d rows, %d identical, %.0f s" % (rows, len(ids), same, time.time() - t0), flush=True)
print("rows compared %d, bit-identical %d, max |diff| %.3g (%.0f s)" % (rows, same, worst, time.time() - t0))
h = lib.se_fnv_all(codes_addr, scales.ctypes.data, V, D, mul)
print("FNV-1a over all %d rows: 0x%016x" % (V, h))
sys.exit(0 if (same_codes and same_scales and rows == same) else 1)
