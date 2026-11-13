#!/usr/bin/env python3
"""Add the REAL gate-first sparse FFN (G4S) to llama.cpp's gemma4.cpp. Run after patch_gemma4.py.

It copies g4s_kernel.h next to gemma4.cpp, inserts g4s_helpers.cpp before the graph constructor and routes the dense
FFN of a layer through g4s_op when G4S selects that layer and the graph holds exactly one token.
"""
import pathlib
import shutil
import sys

here = pathlib.Path(__file__).resolve().parent
src = pathlib.Path(sys.argv[1]) / "src" / "models"
p = src / "gemma4.cpp"
s = p.read_text(encoding="utf-8").replace("\r\n", "\n")
shutil.copy(here / "g4s_kernel.h", src / "g4s_kernel.h")

inc = '#include "models.h"\n'
assert s.count(inc) == 1, "include anchor"
s = s.replace(inc, inc + '\n#include "ggml-cpu.h"\n#include "g4s_kernel.h"\n#include <atomic>\n#include <memory>\n#include <mutex>\n#include <thread>\n'
                     '#if !defined(_WIN32)\n#include <fcntl.h>\n#include <sys/mman.h>\n#include <sys/stat.h>\n#include <unistd.h>\n#endif\n')

anchor = "llama_model_gemma4::graph::graph("
assert s.count(anchor) == 1, "graph ctor anchor"
s = s.replace(anchor, (here / "g4s_helpers.cpp").read_text(encoding="utf-8").replace("\r\n", "\n") + "\n" + anchor)

a = "            if (g4ffn_on(il, n_layer, model.layers[il].ffn_down, model.layers[il].ffn_up)) {\n"
assert s.count(a) == 1, "sim branch anchor (run patch_gemma4.py first)"
b = '''            if (g4s_on(il, n_layer) && n_tokens == 1) {
                g4s_layers[il].up   = model.layers[il].ffn_up;
                g4s_layers[il].down = model.layers[il].ffn_down;
                ggml_tensor * g4_x  = cur;
                ggml_tensor * g4_gg = ggml_gelu(ctx0, build_lora_mm(model.layers[il].ffn_gate, cur));
                cur = ggml_map_custom2(ctx0, g4_x, g4_gg, g4s_op, GGML_N_TASKS_MAX, &g4s_layers[il]);
            } else if (g4ffn_on(il, n_layer, model.layers[il].ffn_down, model.layers[il].ffn_up)) {
'''
s = s.replace(a, b)
p.write_text(s, encoding="utf-8", newline="\n")
print("G4S patched", p)
