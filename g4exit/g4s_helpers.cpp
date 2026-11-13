
// ====================================================================================================================
// G4S v2: the real gate-first sparse FFN (not a simulation). Decode only (one token per graph); batches stay dense.
//
//   G4S="m:from:to:frac,..."   m = g : select by |gelu(g_j)| * ||up_j|| * ||down_j||     (no calibration needed)
//                              m = c : select by |gelu(g_j)| * c_j, plus the fitted bypass  (needs G4_CAL)
//   G4S_CACHE=<dir>            per-layer files, built once and then MEMORY-MAPPED (file-backed, no anonymous copy):
//                                g4s_LNN.bin        the down matrix, neuron-major, as Q4_K records
//                                g4b_LNN_fFFF.bin   mode c: selection weights + bypass (f16) for that fraction
// Per layer:  g = W_gate x (ordinary ggml matvec, dense)  ->  pick the top `frac` neurons  ->  for each picked neuron
//     u_j = up_j . x   (the model's own Q4_K row, read in place)      y += gelu(g_j) * u_j * down_j   (g4s_kernel.h)
// ====================================================================================================================
struct g4s_layer {
    char                mode = 0;
    float               frac = 1.0f;
    int                 il   = 0;
    const ggml_tensor * up   = nullptr;
    const ggml_tensor * down = nullptr;
    bool                ready = false;
    std::mutex          mtx;
    int                 n = 0, d = 0, uprow = 0, rec = 0, r = 0;
    const uint8_t *     updata = nullptr;
    const uint8_t *     recs   = nullptr;
    const float *       c      = nullptr;
    const ggml_fp16_t * A1t    = nullptr; // [r][d]
    const ggml_fp16_t * A2     = nullptr; // [r][d]
    std::vector<uint8_t> own_rec, own_byp; // only when there is no cache directory
    std::vector<float>   own_c;
    std::vector<uint8_t> xq;
    std::vector<float>   sc, tmp;
    std::vector<int>     sel;
    uint64_t             sel_gen = 1, cur_gen = 0;
    std::vector<std::vector<float>> part;
    std::atomic<int>    done{ 0 };
    long long           kept = 0, total = 0, calls = 0;
    double              us = 0.0, us_sel = 0.0, us_rec = 0.0, us_byp = 0.0;
};
static std::unique_ptr<g4s_layer[]> g4s_layers;
static int                          g4s_n    = 0;
static bool                         g4s_init = false;

static float g4s_f16(uint16_t v) {
    return ggml_fp16_to_fp32((ggml_fp16_t) v);
}

static const uint8_t * g4s_map(const char * path, size_t * len) {
#if !defined(_WIN32)
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        return nullptr;
    }
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size <= 0) {
        close(fd);
        return nullptr;
    }
    void * p = mmap(nullptr, (size_t) st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (p == MAP_FAILED) {
        return nullptr;
    }
    *len = (size_t) st.st_size;
    return (const uint8_t *) p;
#else
    (void) path;
    (void) len;
    return nullptr;
#endif
}

static inline float g4s_dot_f16(const ggml_fp16_t * w, const float * x, int n) {
#if defined(__ARM_NEON)
    float32x4_t a0 = vdupq_n_f32(0.0f), a1 = a0, a2 = a0, a3 = a0;
    int         i  = 0;
    for (; i + 16 <= n; i += 16) {
        a0 = vfmaq_f32(a0, vcvt_f32_f16(vld1_f16((const float16_t *) (w + i))), vld1q_f32(x + i));
        a1 = vfmaq_f32(a1, vcvt_f32_f16(vld1_f16((const float16_t *) (w + i + 4))), vld1q_f32(x + i + 4));
        a2 = vfmaq_f32(a2, vcvt_f32_f16(vld1_f16((const float16_t *) (w + i + 8))), vld1q_f32(x + i + 8));
        a3 = vfmaq_f32(a3, vcvt_f32_f16(vld1_f16((const float16_t *) (w + i + 12))), vld1q_f32(x + i + 12));
    }
    float s = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
    for (; i < n; ++i) {
        s += ggml_fp16_to_fp32(w[i]) * x[i];
    }
    return s;
#else
    float s = 0.0f;
    for (int i = 0; i < n; ++i) {
        s += ggml_fp16_to_fp32(w[i]) * x[i];
    }
    return s;
#endif
}

static inline void g4s_axpy_f16(float * y, const ggml_fp16_t * w, float z, int n) {
#if defined(__ARM_NEON)
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        vst1q_f32(y + i, vfmaq_n_f32(vld1q_f32(y + i), vcvt_f32_f16(vld1_f16((const float16_t *) (w + i))), z));
        vst1q_f32(y + i + 4, vfmaq_n_f32(vld1q_f32(y + i + 4), vcvt_f32_f16(vld1_f16((const float16_t *) (w + i + 4))), z));
    }
    for (; i < n; ++i) {
        y[i] += z * ggml_fp16_to_fp32(w[i]);
    }
#else
    for (int i = 0; i < n; ++i) {
        y[i] += z * ggml_fp16_to_fp32(w[i]);
    }
#endif
}

static void g4s_report() {
    double    us = 0.0, sel = 0.0, rec = 0.0, byp = 0.0;
    long long kept = 0, total = 0, calls = 0;
    for (int i = 0; i < g4s_n; ++i) {
        const g4s_layer & L = g4s_layers[i];
        if (L.calls > 0) {
            fprintf(stderr, "G4S_STAT il=%d frac=%.3f kept=%lld total=%lld calls=%lld us_per_call=%.1f sel=%.1f rec=%.1f byp=%.1f\n", i, L.frac, L.kept,
                    L.total, L.calls, L.us / L.calls, L.us_sel / L.calls, L.us_rec / L.calls, L.us_byp / L.calls);
            us += L.us;
            sel += L.us_sel;
            rec += L.us_rec;
            byp += L.us_byp;
            kept += L.kept;
            total += L.total;
            calls = std::max(calls, L.calls);
        }
    }
    if (calls > 0) {
        fprintf(stderr, "G4S_TOTAL sparse-FFN time per token %.2f ms (select %.2f, records %.2f, bypass %.2f), kept %.2f %% of the neurons in the sparse layers\n",
                us / calls / 1000.0, sel / calls / 1000.0, rec / calls / 1000.0, byp / calls / 1000.0, 100.0 * kept / std::max(1LL, total));
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
            }
            p = q + 1;
        }
        if (any) {
            fprintf(stderr, "G4S enabled: %s\n", s.c_str());
            atexit(g4s_report);
        }
    }
    return g4s_layers[il].mode != 0;
}

static void g4s_prepare(g4s_layer & L) {
    const ggml_tensor * up = L.up;
    GGML_ASSERT(up->type == GGML_TYPE_Q4_K && "G4S expects Q4_K up projections (and --no-repack)");
    L.n      = (int) up->ne[1];
    L.d      = (int) up->ne[0];
    L.uprow  = (int) ggml_row_size(GGML_TYPE_Q4_K, L.d);
    L.rec    = g4s_rec_size(L.d);
    L.updata = (const uint8_t *) up->data;
    const int    n = L.n, d = L.d;
    const size_t hdr = 4 * sizeof(int32_t), cbytes = (size_t) n * sizeof(float), rbytes = (size_t) n * L.rec;
    const char * dir = getenv("G4S_CACHE");
    char         path[1024] = { 0 };

    // ---------------- records
    const uint8_t * base = nullptr;
    size_t          len  = 0;
    if (dir) {
        snprintf(path, sizeof(path), "%s/g4s_L%02d.bin", dir, L.il);
        base = g4s_map(path, &len);
        if (base) {
            int32_t h[4];
            memcpy(h, base, sizeof(h));
            if (!(h[0] == 0x32533447 && h[1] == n && h[2] == d && h[3] == L.rec && len == hdr + cbytes + rbytes)) {
                base = nullptr;
            }
        }
    }
    if (!base) {
        const int64_t t0 = ggml_time_us();
        L.own_c.resize(n);
        L.own_rec.resize(rbytes);
        std::vector<float> buf(std::max(n, d));
        std::vector<float> un(n);
        for (int j = 0; j < n; ++j) {
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
            g4s_build_record(q.data(), d, g4s_f16, L.own_rec.data() + (size_t) j * L.rec);
            L.own_c[j] = un[j] * (float) sqrt(s);
        }
        fprintf(stderr, "G4S layer %d: %d down records of %d bytes built in %.1f s\n", L.il, n, L.rec, (ggml_time_us() - t0) / 1e6);
        bool mapped = false;
        if (dir) {
            FILE * f = fopen(path, "wb");
            if (f) {
                const int32_t h[4] = { 0x32533447, n, d, L.rec };
                fwrite(h, sizeof(int32_t), 4, f);
                fwrite(L.own_c.data(), sizeof(float), n, f);
                fwrite(L.own_rec.data(), 1, rbytes, f);
                fclose(f);
                base = g4s_map(path, &len);
                mapped = base != nullptr && len == hdr + cbytes + rbytes;
            }
        }
        if (mapped) {
            std::vector<uint8_t>().swap(L.own_rec);
            std::vector<float>().swap(L.own_c);
        } else {
            base = nullptr;
        }
    }
    if (base) {
        L.c    = (const float *) (base + hdr);
        L.recs = base + hdr + cbytes;
    } else {
        L.c    = L.own_c.data();
        L.recs = L.own_rec.data();
    }

    // ---------------- mode c: calibrated selection weights + bypass
    if (L.mode == 'c') {
        const uint8_t * bb   = nullptr;
        size_t          blen = 0;
        if (dir) {
            snprintf(path, sizeof(path), "%s/g4b_L%02d_f%03d.bin", dir, L.il, (int) lrintf(L.frac * 1000.0f));
            bb = g4s_map(path, &blen);
            if (bb) {
                int32_t h[4];
                memcpy(h, bb, sizeof(h));
                if (!(h[0] == 0x31423447 && h[1] == n && h[2] == d && blen == hdr + cbytes + 2 * (size_t) h[3] * d * sizeof(ggml_fp16_t))) {
                    bb = nullptr;
                }
            }
        }
        if (!bb) {
            g4ffn_layer tmp;
            tmp.val = L.frac;
            g4_load_cal(tmp, L.il);
            GGML_ASSERT((int) tmp.c.size() == n);
            const int r = tmp.r;
            L.own_byp.resize(hdr + cbytes + 2 * (size_t) r * d * sizeof(ggml_fp16_t));
            const int32_t h[4] = { 0x31423447, n, d, r };
            memcpy(L.own_byp.data(), h, sizeof(h));
            memcpy(L.own_byp.data() + hdr, tmp.c.data(), cbytes);
            ggml_fp16_t * o = (ggml_fp16_t *) (L.own_byp.data() + hdr + cbytes);
            for (size_t i = 0; i < (size_t) r * d; ++i) {
                o[i] = ggml_fp32_to_fp16(tmp.A1t[i]);
            }
            for (size_t i = 0; i < (size_t) r * d; ++i) {
                o[(size_t) r * d + i] = ggml_fp32_to_fp16(tmp.A2[i]);
            }
            if (dir) {
                FILE * f = fopen(path, "wb");
                if (f) {
                    fwrite(L.own_byp.data(), 1, L.own_byp.size(), f);
                    fclose(f);
                    bb = g4s_map(path, &blen);
                    if (bb && blen == L.own_byp.size()) {
                        std::vector<uint8_t>().swap(L.own_byp);
                    } else {
                        bb = nullptr;
                    }
                }
            }
            if (!bb) {
                bb = L.own_byp.data();
            }
        }
        int32_t h[4];
        memcpy(h, bb, sizeof(h));
        L.r   = h[3];
        L.c   = (const float *) (bb + hdr);
        L.A1t = (const ggml_fp16_t *) (bb + hdr + cbytes);
        L.A2  = L.A1t + (size_t) L.r * d;
    }
}

// dst = sparse FFN output [n_embd, 1]; a = x (the FFN input) [n_embd, 1]; b = gelu(W_gate x) [n_ff, 1]
static void g4s_op(ggml_tensor * dst, const ggml_tensor * a, const ggml_tensor * b, int ith, int nth, void * ud) {
    g4s_layer &   L  = *(g4s_layer *) ud;
    const int64_t t0 = ith == 0 ? ggml_time_us() : 0;
    const float * x  = (const float *) a->data;
    const float * gg = (const float *) b->data;
    static const ggml_type_traits_cpu * tc = ggml_get_type_traits_cpu(GGML_TYPE_Q4_K);
    static const ggml_type_traits_cpu * tq = ggml_get_type_traits_cpu(tc->vec_dot_type);
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
        if (L.sel_gen != L.cur_gen) { // the first thread of this call quantises x and selects; the others reuse it
            const int64_t ts = ggml_time_us();
            const int     n  = L.n;
            L.xq.resize(ggml_row_size(tc->vec_dot_type, L.d));
            tq->from_float(x, L.xq.data(), L.d);
            L.sc.resize(n);
            for (int j = 0; j < n; ++j) {
                L.sc[j] = fabsf(gg[j]) * L.c[j];
            }
            int k = (int) ceil((double) L.frac * (double) n);
            k     = std::max(1, std::min(n, k));
            L.tmp = L.sc;
            std::nth_element(L.tmp.begin(), L.tmp.begin() + (k - 1), L.tmp.end(), [](float p, float q) { return p > q; });
            const float cut = L.tmp[k - 1];
            L.sel.clear();
            for (int j = 0; j < n; ++j) {
                if (L.sc[j] >= cut) {
                    L.sel.push_back(j);
                }
            }
            L.sel_gen = L.cur_gen;
            L.us_sel += (double) (ggml_time_us() - ts);
        }
    }
    const int n = L.n, d = L.d;
    GGML_ASSERT(a->ne[1] == 1 && a->ne[0] == d && b->ne[0] == n && nth <= 64);
    const int nsel = (int) L.sel.size();
    const int k0 = (int) ((int64_t) nsel * ith / nth), k1 = (int) ((int64_t) nsel * (ith + 1) / nth);
    float *   y = L.part[ith].data();
    memset(y, 0, sizeof(float) * d);
    const int64_t t1 = ith == 0 ? ggml_time_us() : 0;
    for (int kk = k0; kk < k1; kk += G4S_GROUP) {
        const int       ng = std::min(G4S_GROUP, k1 - kk);
        const uint8_t * rows[G4S_GROUP];
        float           hv[G4S_GROUP];
        for (int g = 0; g < ng; ++g) {
            const int j = L.sel[kk + g];
            float     u = 0.0f;
            tc->vec_dot(d, &u, 0, L.updata + (size_t) j * L.uprow, 0, L.xq.data(), 0, 1);
            hv[g]   = gg[j] * u;
            rows[g] = L.recs + (size_t) j * L.rec;
        }
        g4s_axpy_group(rows, hv, ng, d, y);
    }
    const int64_t t2 = ith == 0 ? ggml_time_us() : 0;
    if (L.r > 0) { // the bypass, as r always-on linear neurons
        const int r0 = (int) ((int64_t) L.r * ith / nth), r1 = (int) ((int64_t) L.r * (ith + 1) / nth);
        for (int q = r0; q < r1; ++q) {
            const float z = g4s_dot_f16(L.A1t + (size_t) q * d, x, d);
            g4s_axpy_f16(y, L.A2 + (size_t) q * d, z, d);
        }
    }
    if (ith != 0) {
        L.done.fetch_add(1);
        return;
    }
    const int64_t t3 = ggml_time_us();
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
    L.cur_gen += 1;
    L.kept += nsel;
    L.total += n;
    L.calls += 1;
    L.us_rec += (double) (t2 - t1);
    L.us_byp += (double) (t3 - t2);
    L.us += (double) (ggml_time_us() - t0);
}
