"""Is a native embedding lookup bit-exact with Google's embedder model?

Google's Gemma 4 E4B .litertlm has a small `tf_lite_embedder` TFLite section: EMBEDDING_LOOKUP on an INT2
table (262144 x 2560, one float32 scale per row), then MUL by a float32 constant. This runs that section
with the LiteRT interpreter for many token ids (including out-of-range ones, which the model maps to row 0)
and compares with (code * row_scale) * constant computed in float32 with numpy. Prints max |diff| and the
number of exactly equal rows.
"""
import sys
import numpy as np
import tflite

PATH = sys.argv[1] if len(sys.argv) > 1 else "embedder.tflite"
buf = open(PATH, "rb").read()
m = tflite.Model.GetRootAs(buf, 0)
sg = m.Subgraphs(0)
table = scales = const = None
for t in range(sg.TensorsLength()):
    te = sg.Tensors(t)
    b = m.Buffers(te.Buffer())
    if int(te.Type()) == 19 and b.DataLength():               # INT2 table
        table = b.DataAsNumpy().astype(np.uint8)
        scales = te.Quantization().ScaleAsNumpy().astype(np.float32)
        shape = tuple(int(x) for x in te.ShapeAsNumpy())
    if int(te.Type()) == 0 and b.DataLength() == 4:            # the float32 MUL constant
        const = np.frombuffer(b.DataAsNumpy().tobytes(), np.float32)[0]
print("table", shape, "bytes", table.size, "scales", scales.size, "constant", repr(const))


def native_row(tok):
    if tok < 0 or tok >= shape[0]:
        tok = 0
    per = shape[1] // 4
    raw = table[tok * per:(tok + 1) * per]
    v = np.empty(shape[1], np.int8)
    for k in range(4):
        v[k::4] = ((raw >> (2 * k)) & 3).astype(np.int8)
    v[v >= 2] -= 4
    return (v.astype(np.float32) * scales[tok]) * np.float32(const)


try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    from tflite_runtime.interpreter import Interpreter
it = Interpreter(model_path=PATH)
it.allocate_tensors()
inp = it.get_input_details()[0]
out = it.get_output_details()[0]
print("input", inp["shape"], inp["dtype"], "output", out["shape"], out["dtype"])
rng = np.random.default_rng(0)
toks = list(rng.integers(0, shape[0], 3000)) + [0, 1, 2, shape[0] - 1, -1, shape[0], shape[0] + 12345]
equal = 0
worst = 0.0
for tok in toks:
    it.set_tensor(inp["index"], np.array(tok, dtype=inp["dtype"]).reshape(inp["shape"]))
    it.invoke()
    got = it.get_tensor(out["index"]).reshape(-1)
    ref = native_row(int(tok))
    d = float(np.max(np.abs(got.astype(np.float64) - ref.astype(np.float64))))
    worst = max(worst, d)
    equal += int(np.array_equal(got, ref))
print("rows compared %d, bit-identical %d, max |diff| %.3g" % (len(toks), equal, worst))
