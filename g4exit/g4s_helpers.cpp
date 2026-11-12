
// ====================================================================================================================
// G4S: the real gate-first sparse FFN (not a simulation). Decode only (one token per graph); batches stay dense.
//
//   G4S="m:from:to:frac,..."   m = g : select by |gelu(g_j)| * ||up_j|| * ||down_j||     (no calibration needed)
//                              m = c : select by |gelu(g_j)| * c_j, plus the fitted bypass  (needs G4_CAL)
//   G4S_CACHE=<dir>            records are built once per layer (the transposed down matrix is re-quantised to Q4_K)
//                              and stored here as g4s_LNN.bin
// Per layer:  g = W_gate x (ordinary ggml matvec, dense)  ->  pick the top `frac` neurons  ->  for each picked neuron
// read ONE record [up row (Q4_K) | down column (Q4_K row of the transpose + unpacked scales)]:
//     u_j = up_j . x ;  y += gelu(g_j) * u_j * down_j        (g4s_kernel.h)
// ====================================================================================================================
struct g4s_layer {
    char                mode = 0;
    float               frac = 1.0f;
    int                 il   = 0;
    const ggml_tensor * up   = nullptr;
    const ggml_tensor * down = nullptr;
    bool                ready = false;
    std::mutex          mtx;
    int                 n = 0, d = 0, uprow = 0, rec = 0;
    const uint8_t *     recs = nullptr;
    std::vector<uint8_t> own;
    std::vector<float>  c;
    std::vector<float>  A1t, A2;
    int                 r = 0;
    std::vector<std::vector<float>> part;
    std::atomic<int>    done{ 0 };
    long long           kept = 0, total = 0, calls = 0;
    double              us = 0.0;
};
static std::unique_ptr<g4s_layer[]> g4s_layers;
static int                          g4s_n    = 0;
static bool                         g4s_init = false;

static float g4s_f16(uint16_t v) {
    return ggml_fp16_to_fp32((ggml_fp16_t) v);
}

static void g4s_report() {
    double us = 0.0;
    long long kept = 0, total = 0, calls = 0;
    for (int i = 0; i < g4s_n; ++i) {
        const g4s_layer & L = g4s_layers[i];
        if (L.calls > 0) {
            fprintf(stderr, "G4S_STAT il=%d kept=%lld total=%lld calls=%lld us_per_call=%.1f\n", i, L.kept, L.total, L.calls, L.us / L.calls);
            us += L.us;
            kept += L.kept;
            total += L.total;
            calls = std::max(calls, L.calls);
        }
    }
    if (calls > 0) {
        fprintf(stderr, "G4S_TOTAL sparse-FFN time per token %.2f ms, kept %.2f %% of the neurons in the sparse layers\n", us / calls / 1000.0,
                100.0 * kept / std::max(1LL, total));
    }
}

static bool g4s_on(int il, int n_layer) {
    if (!g4s_init) {
        g4s_init = true;
        g4s_n    = n_layer;
        g4s_layers.reset(new g4s_layer[n_layer]);
        const char * e   = getenv("G4S");
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
                    g4s_layers[l].mode = m;
                    g4s_layers[l].frac = v;
                    g4s_layers[l].il   = l;
                    any                = true;
                }
                fprintf(stderr, "G4S mode=%c layers %d..%d fraction %g\n", m, from, to - 1, v);
            }
            p = q + 1;
        }
        if (any) {
            atexit(g4s_report);
        }
    }
    return g4s_layers[il].mode != 0;
}

static void g4s_prepare(g4s_layer & L) {
    const ggml_tensor * up = L.up;
    GGML_ASSERT(up->type == GGML_TYPE_Q4_K && "G4S expects Q4_K up projections");
    L.n     = (int) up->ne[1];
    L.d     = (int) up->ne[0];
    L.uprow = (int) ggml_row_size(GGML_TYPE_Q4_K, L.d);
    L.rec   = L.uprow + g4s_rec_size(L.d);
    const int    n = L.n, d = L.d;
    const size_t bytes = (size_t) n * L.rec;
    L.c.resize(n);
    char         path[1024] = { 0 };
    const char * dir        = getenv("G4S_CACHE");
    if (dir) {
        snprintf(path, sizeof(path), "%s/g4s_L%02d.bin", dir, L.il);
        FILE * f = fopen(path, "rb");
        if (f) {
            int32_t h[4];
            if (fread(h, sizeof(int32_t), 4, f) == 4 && h[0] == 0x31533447 && h[1] == n && h[2] == d && h[3] == L.rec &&
                fread(L.c.data(), sizeof(float), n, f) == (size_t) n) {
                L.own.resize(bytes);
                if (fread(L.own.data(), 1, bytes, f) == bytes) {
                    L.recs = L.own.data();
                }
            }
            fclose(f);
        }
    }
    if (!L.recs) {
        const int64_t t0 = ggml_time_us();
        L.own.resize(bytes);
        std::vector<float> buf(std::max(n, d));
        std::vector<float> un(n);
        for (int j = 0; j < n; ++j) {
            memcpy(L.own.data() + (size_t) j * L.rec, (const uint8_t *) up->data + (size_t) j * L.uprow, L.uprow);
            g4_row(up, j, buf.data());
            double u = 0.0;
            for (int i = 0; i < d; ++i) {
                u += (double) buf[i] * buf[i];
            }
            un[j] = (float) sqrt(u);
            GGML_ASSERT(std::isfinite(un[j]) && "run with --no-repack");
        }
        std::vector<float> W((size_t) d * n);
        for (int i = 0; i < d; ++i) {
            g4_row(L.down, i, W.data() + (size_t) i * n);
        }
        std::vector<float>   col(d);
        std::vector<uint8_t> q(ggml_row_size(GGML_TYPE_Q4_K, d));
        for (int j = 0; j < n; ++j) {
            double s = 0.0;
            for (int i = 0; i < d; ++i) {
                col[i] = W[(size_t) i * n + j];
                s += (double) col[i] * col[i];
            }
            ggml_quantize_chunk(GGML_TYPE_Q4_K, col.data(), q.data(), 0, 1, d, nullptr);
            g4s_build_record(q.data(), d, g4s_f16, L.own.data() + (size_t) j * L.rec + L.uprow);
            L.c[j] = un[j] * (float) sqrt(s);
        }
        L.recs = L.own.data();
        fprintf(stderr, "G4S layer %d: %d records of %d bytes built in %.1f s\n", L.il, n, L.rec, (ggml_time_us() - t0) / 1e6);
        if (dir) {
            FILE * f = fopen(path, "wb");
            if (f) {
                const int32_t h[4] = { 0x31533447, n, d, L.rec };
                fwrite(h, sizeof(int32_t), 4, f);
                fwrite(L.c.data(), sizeof(float), n, f);
                fwrite(L.own.data(), 1, bytes, f);
                fclose(f);
            }
        }
    }
    if (L.mode == 'c') {
        g4ffn_layer tmp;
        tmp.val = L.frac;
        g4_load_cal(tmp, L.il);
        GGML_ASSERT((int) tmp.c.size() == n);
        L.c   = tmp.c;
        L.A1t = tmp.A1t;
        L.A2  = tmp.A2;
        L.r   = tmp.r;
    }
}

// dst = sparse FFN output [n_embd, 1]; a = x (the FFN input) [n_embd, 1]; b = gelu(W_gate x) [n_ff, 1]
static void g4s_op(ggml_tensor * dst, const ggml_tensor * a, const ggml_tensor * b, int ith, int nth, void * ud) {
    g4s_layer & L = *(g4s_layer *) ud;
    {
        std::lock_guard<std::mutex> lk(L.mtx);
        if (!L.ready) {
            g4s_prepare(L);
            L.part.assign(64, std::vector<float>());
            L.ready = true;
        }
        if (L.part[ith].empty()) {
            L.part[ith].assign(L.d, 0.0f);
        }
    }
    const int64_t t0 = ith == 0 ? ggml_time_us() : 0;
    const int     n = L.n, d = L.d;
    GGML_ASSERT(a->ne[1] == 1 && a->ne[0] == d && b->ne[0] == n && nth <= 64);
    const float * x  = (const float *) a->data;
    const float * gg = (const float *) b->data;
    static const ggml_type_traits_cpu * tc = ggml_get_type_traits_cpu(GGML_TYPE_Q4_K);
    static const ggml_type_traits_cpu * tq = ggml_get_type_traits_cpu(tc->vec_dot_type);
    thread_local std::vector<uint8_t> xq;
    thread_local std::vector<float>   sc, tmp;
    thread_local std::vector<int>     sel;
    xq.resize(ggml_row_size(tc->vec_dot_type, d));
    tq->from_float(x, xq.data(), d);
    sc.resize(n);
    for (int j = 0; j < n; ++j) {
        sc[j] = fabsf(gg[j]) * L.c[j];
    }
    int k = (int) ceil((double) L.frac * (double) n);
    k     = std::max(1, std::min(n, k));
    tmp   = sc;
    std::nth_element(tmp.begin(), tmp.begin() + (k - 1), tmp.end(), [](float p, float q) { return p > q; });
    const float cut = tmp[k - 1];
    sel.clear();
    for (int j = 0; j < n; ++j) {
        if (sc[j] >= cut) {
            sel.push_back(j);
        }
    }
    const int nsel = (int) sel.size();
    const int k0 = (int) ((int64_t) nsel * ith / nth), k1 = (int) ((int64_t) nsel * (ith + 1) / nth);
    float *   y = L.part[ith].data();
    memset(y, 0, sizeof(float) * d);
    for (int kk = k0; kk < k1; kk += G4S_GROUP) {
        const int       ng = std::min(G4S_GROUP, k1 - kk);
        const uint8_t * rows[G4S_GROUP];
        float           hv[G4S_GROUP];
        for (int g = 0; g < ng; ++g) {
            const int       j = sel[kk + g];
            const uint8_t * r = L.recs + (size_t) j * L.rec;
            float           u = 0.0f;
            tc->vec_dot(d, &u, 0, r, 0, xq.data(), 0, 1);
            hv[g]   = gg[j] * u;
            rows[g] = r + L.uprow;
        }
        g4s_axpy_group(rows, hv, ng, d, y);
    }
    if (L.r > 0) { // the bypass, as r always-on linear neurons
        const int r0 = (int) ((int64_t) L.r * ith / nth), r1 = (int) ((int64_t) L.r * (ith + 1) / nth);
        for (int q = r0; q < r1; ++q) {
            const float * w1 = L.A1t.data() + (size_t) q * d;
            const float * w2 = L.A2.data() + (size_t) q * d;
            float         z  = 0.0f;
            for (int i = 0; i < d; ++i) {
                z += w1[i] * x[i];
            }
            for (int i = 0; i < d; ++i) {
                y[i] += z * w2[i];
            }
        }
    }
    if (ith != 0) {
        L.done.fetch_add(1);
        return;
    }
    while (L.done.load() < nth - 1) {
        std::this_thread::yield();
    }
    float * o = (float *) dst->data;
    memcpy(o, y, sizeof(float) * d);
    for (int t = 1; t < nth; ++t) {
        const float * p = L.part[t].data();
        for (int i = 0; i < d; ++i) {
            o[i] += p[i];
        }
    }
    L.done.store(0);
    L.kept += nsel;
    L.total += n;
    L.calls += 1;
    L.us += (double) (ggml_time_us() - t0);
}
