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
    const ggml_tensor * up   = nullptr;
    std::vector<float>  norm;  // ||W_down[:, j]||
    std::vector<float>  unorm; // ||W_up[j, :]||
    char                mode  = 0;
    float               val   = 1.0f;
    long long           kept  = 0;
    long long           total = 0;
};
struct g4ffn_cfg {
    bool                     init = false;
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

static bool g4ffn_on(int il, int n_layer, const ggml_tensor * down, const ggml_tensor * up) {
    if (!g4ffn.init) {
        g4ffn.init = true;
        g4ffn.layers.resize(n_layer);
        const char * e   = getenv("G4_FFN");
        std::string  s   = e ? e : "";
        size_t       p   = 0;
        bool         any = false;
        while (p < s.size()) {
            size_t q = s.find(',', p);
            if (q == std::string::npos) {
                q = s.size();
            }
            char  m    = 0;
            int   from = 0;
            int   to   = 0;
            float v    = 1.0f;
            if (sscanf(s.substr(p, q - p).c_str(), "%c:%d:%d:%f", &m, &from, &to, &v) == 4) {
                for (int l = std::max(0, from); l < std::min(n_layer, to); ++l) {
                    g4ffn.layers[l].mode = m;
                    g4ffn.layers[l].val  = v;
                    any                  = true;
                }
                fprintf(stderr, "G4_FFN mode=%c layers %d..%d value %g\n", m, from, to - 1, v);
            }
            p = q + 1;
        }
        if (any) {
            atexit(g4ffn_report);
        }
    }
    if (g4ffn.layers[il].mode == 0) {
        return false;
    }
    g4ffn.layers[il].down = down;
    g4ffn.layers[il].up   = up;
    return true;
}

// dequantise row r of a 2-D weight tensor (needs --no-repack: repacked Q4_K data has another layout)
static void g4_row(const ggml_tensor * w, int64_t r, float * buf) {
    const int64_t   n   = w->ne[0];
    const size_t    rs  = ggml_row_size(w->type, n);
    const uint8_t * src = (const uint8_t *) w->data + r * rs;
    if (w->type == GGML_TYPE_F32) {
        memcpy(buf, src, n * sizeof(float));
    } else {
        const auto * tt = ggml_get_type_traits(w->type);
        GGML_ASSERT(tt->to_float != nullptr);
        tt->to_float(src, buf, n);
    }
}

// dst = h with part of the neurons zeroed. a = h = gelu(gate x) * (up x), b = gelu(gate x); both [n_ff, n_tokens] F32
//   k / t : scored by the true contribution norm |h_j| * ||down_j||       (needs gate AND up: an oracle)
//   g / G : scored by |gelu(g_j)| * ||up_j|| * ||down_j||                 (needs the gate projection only)
//   k, g  : keep the top `val` fraction;  t, G : drop the lowest scores while their sum stays <= val * total
static void g4ffn_op(ggml_tensor * dst, const ggml_tensor * a, const ggml_tensor * b, int ith, int nth, void * ud) {
    (void) nth;
    if (ith != 0) {
        return;
    }
    g4ffn_layer & L  = *(g4ffn_layer *) ud;
    const int64_t n  = a->ne[0];
    const int64_t nt = a->ne[1];
    GGML_ASSERT(a->type == GGML_TYPE_F32 && b->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    if (L.norm.empty()) {
        GGML_ASSERT(L.down->ne[0] == n && L.up->ne[1] == n);
        std::vector<float>  buf(std::max<int64_t>(n, L.up->ne[0]));
        std::vector<double> sq(n, 0.0);
        for (int64_t r = 0; r < L.down->ne[1]; ++r) {
            g4_row(L.down, r, buf.data());
            for (int64_t j = 0; j < n; ++j) {
                sq[j] += (double) buf[j] * buf[j];
            }
        }
        L.norm.resize(n);
        L.unorm.resize(n);
        for (int64_t j = 0; j < n; ++j) {
            L.norm[j] = (float) sqrt(sq[j]);
            g4_row(L.up, j, buf.data());
            double u = 0.0;
            for (int64_t i = 0; i < L.up->ne[0]; ++i) {
                u += (double) buf[i] * buf[i];
            }
            L.unorm[j] = (float) sqrt(u);
            GGML_ASSERT(std::isfinite(L.norm[j]) && std::isfinite(L.unorm[j]) && "weight norms not finite: run with --no-repack");
        }
    }
    const bool by_gate = L.mode == 'g' || L.mode == 'G';
    const bool top_k   = L.mode == 'k' || L.mode == 'g';
    std::vector<float> sc(n), tmp(n);
    for (int64_t t = 0; t < nt; ++t) {
        const float * h  = (const float *) a->data + t * n;
        const float * gg = (const float *) b->data + t * n;
        float *       o  = (float *) dst->data + t * n;
        for (int64_t j = 0; j < n; ++j) {
            sc[j] = by_gate ? fabsf(gg[j]) * L.unorm[j] * L.norm[j] : fabsf(h[j]) * L.norm[j];
        }
        tmp = sc;
        float cut; // keep neurons with score >= cut
        if (top_k) {
            int64_t k = (int64_t) ceil((double) L.val * (double) n);
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
            while (m < n - 1 && acc + tmp[m] <= (double) L.val * tot) {
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
anchor = "llama_model_gemma4::graph::graph("   # b11235: (const llama_model & model, ...); b12317: (const llm_graph_params & params)
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
new = r'''            if (g4ffn_on(il, n_layer, model.layers[il].ffn_down, model.layers[il].ffn_up)) {
                ggml_tensor * g4_up = build_lora_mm(model.layers[il].ffn_up, cur);
                ggml_tensor * g4_gg = ggml_gelu(ctx0, build_lora_mm(model.layers[il].ffn_gate, cur));
                ggml_tensor * g4_h  = ggml_mul(ctx0, g4_gg, g4_up);
                g4_h = ggml_map_custom2(ctx0, g4_h, g4_gg, g4ffn_op, 1, &g4ffn.layers[il]);
                cur  = build_lora_mm(model.layers[il].ffn_down, g4_h);
            } else {
''' + "\n".join(("    " + l) if l else l for l in orig.split("\n")) + "            }\n"
s = s[:a0] + new + s[a1:]
p.write_text(s, encoding="utf-8", newline="\n")
print("patched", p)
