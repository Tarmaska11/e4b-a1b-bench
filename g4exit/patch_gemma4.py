#!/usr/bin/env python3
"""Add two research knobs to llama.cpp's src/models/gemma4.cpp (b11235).

G4_SKIP="a:b,c:d"   skip layers a <= il < b. Only KV-less layers may be skipped (anything else aborts,
                    because it would leave a hole in the KV cache).
G4_FFN="m:a:b:v"    in layers a <= il < b, zero part of the FFN's hidden vector h = gelu(gate x) * (up x)
                    before the down projection. Neuron j is scored by |h_j| * ||W_down[:, j]||, the norm of
                    its contribution to the layer output.
                      m = k : keep the top v fraction of neurons per token (the per-token neuron ORACLE)
                      m = t : drop the lowest-scored neurons while their score sum stays <= v * (sum of all
                              scores). By the triangle inequality the FFN output then moves by at most that sum.
                    Per-layer kept counts are printed at exit as "G4FFN_STAT il=.. kept=.. total=..".
"""
import pathlib
import sys

p = pathlib.Path(sys.argv[1]) / "src" / "models" / "gemma4.cpp"
s = p.read_text(encoding="utf-8").replace("\r\n", "\n")

inc = '#include "models.h"\n'
assert s.count(inc) == 1, "include anchor"
s = s.replace(inc, inc + "\n#include <algorithm>\n#include <cmath>\n#include <cstdio>\n#include <cstdlib>\n#include <cstring>\n"
                         "#include <string>\n#include <utility>\n#include <vector>\n")

# ------------------------------------------------------------------ helpers
HELPERS = r'''
struct g4ffn_layer {
    const ggml_tensor * down = nullptr;
    std::vector<float>  norm;
    long long           kept  = 0;
    long long           total = 0;
};
struct g4ffn_cfg {
    bool                     init = false;
    char                     mode = 0;
    int                      from = 0;
    int                      to   = 0;
    float                    val  = 1.0f;
    std::vector<g4ffn_layer> layers;
};
static g4ffn_cfg g4ffn;

static void g4ffn_report() {
    for (size_t i = 0; i < g4ffn.layers.size(); ++i) {
        if (g4ffn.layers[i].total > 0) {
            fprintf(stderr, "G4FFN_STAT il=%d kept=%lld total=%lld\n", (int) i, g4ffn.layers[i].kept, g4ffn.layers[i].total);
        }
    }
}

static bool g4ffn_on(int il, int n_layer, const ggml_tensor * down) {
    if (!g4ffn.init) {
        g4ffn.init = true;
        g4ffn.layers.resize(n_layer);
        const char * e = getenv("G4_FFN");
        if (e && sscanf(e, "%c:%d:%d:%f", &g4ffn.mode, &g4ffn.from, &g4ffn.to, &g4ffn.val) == 4) {
            fprintf(stderr, "G4_FFN mode=%c layers %d..%d value %g\n", g4ffn.mode, g4ffn.from, g4ffn.to - 1, g4ffn.val);
            atexit(g4ffn_report);
        } else {
            g4ffn.mode = 0;
        }
    }
    if (g4ffn.mode == 0 || il < g4ffn.from || il >= g4ffn.to) {
        return false;
    }
    g4ffn.layers[il].down = down;
    return true;
}

// dst = a with the low-contribution neurons zeroed; a is [n_ff, n_tokens] F32
static void g4ffn_op(ggml_tensor * dst, const ggml_tensor * a, int ith, int nth, void * ud) {
    (void) nth;
    if (ith != 0) {
        return;
    }
    g4ffn_layer & L  = *(g4ffn_layer *) ud;
    const int64_t n  = a->ne[0];
    const int64_t nt = a->ne[1];
    GGML_ASSERT(a->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    if (L.norm.empty()) {
        GGML_ASSERT(L.down->ne[0] == n);
        const int64_t       rows = L.down->ne[1];
        const size_t        rs   = ggml_row_size(L.down->type, n);
        const auto *        tt   = ggml_get_type_traits(L.down->type);
        std::vector<float>  buf(n);
        std::vector<double> sq(n, 0.0);
        for (int64_t r = 0; r < rows; ++r) {
            const uint8_t * src = (const uint8_t *) L.down->data + r * rs;
            if (L.down->type == GGML_TYPE_F32) {
                memcpy(buf.data(), src, n * sizeof(float));
            } else {
                GGML_ASSERT(tt->to_float != nullptr);
                tt->to_float(src, buf.data(), n);
            }
            for (int64_t j = 0; j < n; ++j) {
                sq[j] += (double) buf[j] * buf[j];
            }
        }
        L.norm.resize(n);
        for (int64_t j = 0; j < n; ++j) {
            L.norm[j] = (float) sqrt(sq[j]);
        }
    }
    std::vector<float> sc(n), tmp(n);
    for (int64_t t = 0; t < nt; ++t) {
        const float * h = (const float *) a->data + t * n;
        float *       o = (float *) dst->data + t * n;
        for (int64_t j = 0; j < n; ++j) {
            sc[j] = fabsf(h[j]) * L.norm[j];
        }
        tmp = sc;
        float cut; // keep neurons with score >= cut
        if (g4ffn.mode == 'k') {
            int64_t k = (int64_t) ceil((double) g4ffn.val * (double) n);
            k         = std::max<int64_t>(1, std::min<int64_t>(n, k));
            std::nth_element(tmp.begin(), tmp.begin() + (k - 1), tmp.end(), [](float x, float y) { return x > y; });
            cut = tmp[k - 1];
        } else {
            std::sort(tmp.begin(), tmp.end());
            double tot = 0.0;
            for (int64_t j = 0; j < n; ++j) {
                tot += tmp[j];
            }
            double  acc = 0.0;
            int64_t m   = 0;
            while (m < n - 1 && acc + tmp[m] <= (double) g4ffn.val * tot) {
                acc += tmp[m];
                ++m;
            }
            cut = tmp[m];
        }
        long long kept = 0;
        for (int64_t j = 0; j < n; ++j) {
            const bool keep = sc[j] >= cut;
            o[j]            = keep ? h[j] : 0.0f;
            kept += keep;
        }
        L.kept += kept;
        L.total += n;
    }
}

'''
anchor = "llama_model_gemma4::graph::graph(const llama_model & model"
assert s.count(anchor) == 1, "graph ctor anchor"
s = s.replace(anchor, HELPERS + anchor)

# ------------------------------------------------------------------ layer skip
anchor = "    for (int il = 0; il < n_layer; ++il) {\n"
assert s.count(anchor) == 1, "loop anchor"
inject = anchor + r'''        {
            static bool g4_init = false;
            static std::vector<std::pair<int, int>> g4_ranges;
            if (!g4_init) {
                g4_init = true;
                const char * e = getenv("G4_SKIP");
                std::string s = e ? e : "";
                size_t p = 0;
                while (p < s.size()) {
                    size_t q = s.find(',', p);
                    if (q == std::string::npos) { q = s.size(); }
                    const std::string part = s.substr(p, q - p);
                    const size_t c = part.find(':');
                    if (c != std::string::npos) {
                        g4_ranges.push_back({ atoi(part.substr(0, c).c_str()), atoi(part.substr(c + 1).c_str()) });
                    }
                    p = q + 1;
                }
            }
            bool g4_skip = false;
            for (const auto & r : g4_ranges) {
                if (il >= r.first && il < r.second) { g4_skip = true; }
            }
            if (g4_skip) {
                GGML_ASSERT(!hparams.has_kv(il) && "G4_SKIP may only skip KV-less layers");
                continue;
            }
        }
'''
s = s.replace(anchor, inject)

# ------------------------------------------------------------------ sparse FFN (dense branch only)
a0 = s.index("            cur = build_ffn(cur,\n")
assert s.count("            cur = build_ffn(cur,\n") == 1, "dense ffn anchor"
end = "LLM_FFN_GELU, LLM_FFN_PAR, il);\n"
a1 = s.index(end, a0) + len(end)
orig = s[a0:a1]
new = r'''            if (g4ffn_on(il, n_layer, model.layers[il].ffn_down)) {
                ggml_tensor * g4_up   = build_lora_mm(model.layers[il].ffn_up, cur);
                ggml_tensor * g4_gate = build_lora_mm(model.layers[il].ffn_gate, cur);
                ggml_tensor * g4_h    = ggml_mul(ctx0, ggml_gelu(ctx0, g4_gate), g4_up);
                g4_h = ggml_map_custom1(ctx0, g4_h, g4ffn_op, 1, &g4ffn.layers[il]);
                cur  = build_lora_mm(model.layers[il].ffn_down, g4_h);
            } else {
''' + "\n".join(("    " + l) if l else l for l in orig.split("\n")) + "            }\n"
s = s[:a0] + new + s[a1:]
p.write_text(s, encoding="utf-8", newline="\n")
print("patched", p)
