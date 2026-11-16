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
s = s.replace(inc, inc + '\n#include "ggml-cpu.h"\n#include "g4s_kernel.h"\n#include <atomic>\n#include <memory>\n#include <mutex>\n#include <thread>\n#include <vector>\n#include <cstdint>\n#include "gguf.h"\n'
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

# G4_OVERLAY: re-point replaced tensors at a second file, once, before the first graph is built
o = "    // do the PLE first to guarantee it is done in the host buffer\n"
assert s.count(o) == 1, "graph ctor body anchor"
s = s.replace(o, "    g4o_apply(model);\n\n" + o)

# G4TICK: a pass-through op right before the LM head that timestamps every decoded token (all modes)
h = "    // lm_head\n    cur = build_lora_mm(model.output, cur, model.output_s);\n"
assert s.count(h) == 1, "lm_head anchor"
s = s.replace(h, "    if (n_tokens == 1) {\n        cur = ggml_map_custom1(ctx0, cur, g4tick_op, 1, nullptr);\n    }\n" + h)
# G4H: the 1-bit scan + exact shortlist head, decode only (the dense head stays for batches and when G4H is unset)
assert s.count(h) == 1, "lm_head anchor (second use)"
s = s.replace(h, """    // lm_head
    if (n_tokens == 1 && g4h_on(model.output, model.output_s)) {
        const int64_t g4h_v     = model.output->ne[1];
        ggml_tensor * g4h_a1[1] = { cur };
        ggml_tensor * g4h_sc    = ggml_custom_4d(ctx0, GGML_TYPE_F32, g4h_v, 1, 1, 1, g4h_a1, 1, g4h_scan, GGML_N_TASKS_MAX, nullptr);
        ggml_tensor * g4h_a2[1] = { g4h_sc };
        ggml_tensor * g4h_fl    = ggml_custom_4d(ctx0, GGML_TYPE_F32, g4h_v, 1, 1, 1, g4h_a2, 1, g4h_select, 1, nullptr);
        ggml_tensor * g4h_a3[2] = { cur, g4h_fl };
        cur = ggml_custom_4d(ctx0, GGML_TYPE_F32, g4h_v, 1, 1, 1, g4h_a3, 2, g4h_exact, GGML_N_TASKS_MAX, nullptr);
    } else {
        cur = build_lora_mm(model.output, cur, model.output_s);
    }
""")
p.write_text(s, encoding="utf-8", newline="\n")
print("G4S patched", p)

# G4PROF: a per-kind-of-node wall-clock table in ggml's compute loop
c = pathlib.Path(sys.argv[1]) / "ggml" / "src" / "ggml-cpu" / "ggml-cpu.c"
cs = c.read_text(encoding="utf-8").replace("\r\n", "\n")
a0 = "static thread_ret_t ggml_graph_compute_thread(void * data) {\n"
assert cs.count(a0) == 1, "compute thread anchor"
cs = cs.replace(a0, (here / "g4prof.c.inc").read_text(encoding="utf-8").replace("\r\n", "\n") + a0)
a1 = "        const int n_fused = ggml_cpu_try_fuse_ops(cgraph, node_n, &params, cplan);\n"
assert cs.count(a1) == 1, "fuse anchor"
cs = cs.replace(a1, "        const int64_t g4p_t0 = (state->ith == 0 && g4prof_enabled() == 1) ? ggml_time_us() : 0;\n"
                    "        const struct ggml_tensor * g4p_node = node;\n" + a1)
a2 = "        if (node_n + 1 < cgraph->n_nodes) {\n            ggml_barrier(state->threadpool);\n        }\n"
assert cs.count(a2) == 1, "barrier anchor"
cs = cs.replace(a2, a2 + "        if (g4p_t0) {\n            g4prof_add(g4p_node, ggml_time_us() - g4p_t0);\n        }\n")
c.write_text(cs, encoding="utf-8", newline="\n")
print("G4PROF patched", c)

# G4A: decode attention that reads the key cache once (g4a.cpp.inc), in front of upstream's split-KV decode path
o = pathlib.Path(sys.argv[1]) / "ggml" / "src" / "ggml-cpu" / "ops.cpp"
os_ = o.read_text(encoding="utf-8").replace("\r\n", "\n")
b0 = "#include <algorithm>\n"
assert os_.count(b0) == 1, "ops.cpp include anchor"
os_ = os_.replace(b0, b0 + "#include <atomic>\n#include <cstdio>\n#include <cstdlib>\n#include <vector>\n")
b1 = "static void ggml_compute_forward_flash_attn_ext_f16(\n"
assert os_.count(b1) == 1, "flash attention anchor"
os_ = os_.replace(b1, (here / "g4a.cpp.inc").read_text(encoding="utf-8").replace("\r\n", "\n")
                      + (here / "g4a_mass.cpp.inc").read_text(encoding="utf-8").replace("\r\n", "\n") + b1)
b2 = "    if (use_split_kv_path) {\n"
assert os_.count(b2) == 1, "split path anchor"
os_ = os_.replace(b2, b2 + "        if (g4a_mode() > 0) {\n"
                           "            if (g4a_mode() >= 4 && g4a_mass_ok(params, dst)) {\n"
                           "                g4a_decode_mass(params, dst);\n"
                           "            } else {\n"
                           "                g4a_decode(params, dst);\n"
                           "            }\n"
                           "            return;\n        }\n")
o.write_text(os_, encoding="utf-8", newline="\n")
print("G4A patched", o)

# shared-library builds (the Android app): gemma4.cpp now calls into the CPU backend library (type traits, G4PROF),
# which libllama does not link by default
cm = pathlib.Path(sys.argv[1]) / "src" / "CMakeLists.txt"
if cm.exists():
    ct = cm.read_text(encoding="utf-8")
    if "TARGET ggml-cpu" not in ct:
        ct += "\n".join(["", "# G4S / G4H / G4PROF: symbols of the CPU backend used by src/models/gemma4.cpp",
                         "if (TARGET ggml-cpu)", "    target_link_libraries(llama PRIVATE ggml-cpu)", "endif()", ""])
        cm.write_text(ct, encoding="utf-8", newline="\n")
        print("CMake patched", cm)
