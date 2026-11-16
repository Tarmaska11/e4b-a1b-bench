
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
    std::atomic<int>    next{ 0 }; // v3: next unclaimed position in sel (work is handed out in chunks)
    long long           kept = 0, total = 0, calls = 0;
    double              us = 0.0, us_sel = 0.0, us_rec = 0.0, us_byp = 0.0, us_wait = 0.0;
};
static std::unique_ptr<g4s_layer[]> g4s_layers;
static int                          g4s_n    = 0;
static bool                         g4s_init = false;
// v3 knobs (environment, read once):
//   G4S_PF=<groups>     prefetch the rows of the neurons this many groups of 16 ahead (default 1; 0 = off = v2)
//   G4S_CHUNK=<neurons> neurons per chunk taken from the shared counter (default 128; 0 = v2's fixed split)
//   G4S_POPULATE=0      do not pre-map the cache files (default: MAP_POPULATE)
static int       g4s_pf       = 1;
static int       g4s_chunk    = 128;
static bool      g4s_populate = true;
//   G4S_DROPDOWN=0      keep the model's own down matrices mapped (default: once a layer's records are mapped,
//                       hand the dense down matrix of that layer back to the kernel; sparse decode never reads it)
static bool      g4s_dropdown = true;
static double    g4s_prep_us  = 0.0;
static int       g4s_last     = -1; // highest sparse layer: when its first call ends, every layer is prepared
static long long g4s_flt0[2]  = { 0, 0 };
static int64_t   g4s_t_first  = 0;

static void g4s_faults(long long * minflt, long long * majflt) {
    *minflt = 0;
    *majflt = 0;
#if !defined(_WIN32)
    FILE * f = fopen("/proc/self/stat", "r");
    if (!f) {
        return;
    }
    char   buf[2048];
    size_t n = fread(buf, 1, sizeof(buf) - 1, f);
    fclose(f);
    buf[n]         = 0;
    const char * p = strrchr(buf, ')'); // the fields after the command name: state ppid pgrp session tty tpgid flags minflt cminflt majflt
    if (!p) {
        return;
    }
    char      st = 0;
    long long v[9];
    if (sscanf(p + 1, " %c %lld %lld %lld %lld %lld %lld %lld %lld %lld", &st, &v[0], &v[1], &v[2], &v[3], &v[4], &v[5], &v[6], &v[7], &v[8]) == 10) {
        *minflt = v[6];
        *majflt = v[8];
    }
#endif
}

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
    int flags = MAP_PRIVATE;
#if defined(MAP_POPULATE)
    if (g4s_populate) {
        flags |= MAP_POPULATE; // v3: map every page now, so the decode loop takes no first-touch faults
    }
#endif
    void * p = mmap(nullptr, (size_t) st.st_size, PROT_READ, flags, fd, 0);
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
    double    us = 0.0, sel = 0.0, rec = 0.0, byp = 0.0, wait = 0.0;
    long long kept = 0, total = 0, calls = 0;
    for (int i = 0; i < g4s_n; ++i) {
        const g4s_layer & L = g4s_layers[i];
        if (L.calls > 0) {
            fprintf(stderr, "G4S_STAT il=%d frac=%.3f kept=%lld total=%lld calls=%lld us_per_call=%.1f sel=%.1f rec=%.1f byp=%.1f wait=%.1f\n", i, L.frac,
                    L.kept, L.total, L.calls, L.us / L.calls, L.us_sel / L.calls, L.us_rec / L.calls, L.us_byp / L.calls, L.us_wait / L.calls);
            us += L.us;
            sel += L.us_sel;
            rec += L.us_rec;
            byp += L.us_byp;
            wait += L.us_wait;
            kept += L.kept;
            total += L.total;
            calls = std::max(calls, L.calls);
        }
    }
    if (calls > 0) {
        fprintf(stderr, "G4S_TOTAL sparse-FFN time per token %.2f ms (select %.2f, records %.2f, bypass %.2f), kept %.2f %% of the neurons in the sparse layers\n",
                us / calls / 1000.0, sel / calls / 1000.0, rec / calls / 1000.0, byp / calls / 1000.0, 100.0 * kept / std::max(1LL, total));
        long long f1[2];
        g4s_faults(&f1[0], &f1[1]);
        fprintf(stderr, "G4S_V3 pf=%d chunk=%d populate=%d dropdown=%d prepare %.0f ms | per selected neuron %.3f us (records, wall) | wait %.2f ms per token | after the first token: %.1f s, minflt %lld, majflt %lld\n",
                g4s_pf, g4s_chunk, g4s_populate ? 1 : 0, g4s_dropdown ? 1 : 0, g4s_prep_us / 1000.0, rec / std::max(1LL, kept), wait / calls / 1000.0,
                g4s_t_first ? (ggml_time_us() - g4s_t_first) / 1e6 : 0.0, f1[0] - g4s_flt0[0], f1[1] - g4s_flt0[1]);
    }
}

static inline void g4s_prefetch_neuron(const g4s_layer & L, int j) {
#if defined(__GNUC__) || defined(__clang__)
    const uint8_t * u = L.updata + (size_t) j * L.uprow;
    for (int o = 0; o < L.uprow; o += 64) {
        __builtin_prefetch(u + o, 0, 2);
    }
    __builtin_prefetch(u + L.uprow - 1, 0, 2);
    const uint8_t * r = L.recs + (size_t) j * L.rec;
    for (int o = 0; o < L.rec; o += 64) {
        __builtin_prefetch(r + o, 0, 2);
    }
    __builtin_prefetch(r + L.rec - 1, 0, 2);
#else
    (void) L;
    (void) j;
#endif
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
                    g4s_last           = std::max(g4s_last, l);
                }
            }
            p = q + 1;
        }
        if (any) {
            if (const char * v = getenv("G4S_PF")) {
                g4s_pf = std::max(0, atoi(v));
            }
            if (const char * v = getenv("G4S_CHUNK")) {
                g4s_chunk = std::max(0, atoi(v)) / G4S_GROUP * G4S_GROUP;
            }
            if (const char * v = getenv("G4S_POPULATE")) {
                g4s_populate = atoi(v) != 0;
            }
            if (const char * v = getenv("G4S_DROPDOWN")) {
                g4s_dropdown = atoi(v) != 0;
            }
            fprintf(stderr, "G4S enabled: %s (v3: pf=%d chunk=%d populate=%d dropdown=%d)\n", s.c_str(), g4s_pf, g4s_chunk, g4s_populate ? 1 : 0, g4s_dropdown ? 1 : 0);
            atexit(g4s_report);
        }
    }
    return g4s_layers[il].mode != 0;
}

static void g4s_prepare(g4s_layer & L) {
    const int64_t       t_prep = ggml_time_us();
    const ggml_tensor * up     = L.up;
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
#if !defined(_WIN32)
        if (g4s_dropdown && L.down && L.down->data) { // v3: the dense down matrix is dead weight in sparse decode: give its pages back
            const size_t    pg  = (size_t) sysconf(_SC_PAGESIZE);
            const uintptr_t beg = ((uintptr_t) L.down->data + pg - 1) & ~(uintptr_t) (pg - 1);
            const uintptr_t end = ((uintptr_t) L.down->data + ggml_nbytes(L.down)) & ~(uintptr_t) (pg - 1);
            if (end > beg) {
#if defined(MADV_PAGEOUT)
                int rc = madvise((void *) beg, end - beg, MADV_PAGEOUT);
                if (rc != 0) {
                    rc = madvise((void *) beg, end - beg, MADV_DONTNEED);
                }
#else
                int rc = madvise((void *) beg, end - beg, MADV_DONTNEED);
#endif
                static bool warned = false;
                if (rc != 0 && !warned) {
                    warned = true;
                    fprintf(stderr, "G4S: could not release the dense down matrix of layer %d\n", L.il);
                }
            }
        }
#endif
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
    g4s_prep_us += (double) (ggml_time_us() - t_prep);
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
            L.next.store(0);
            L.sel_gen = L.cur_gen;
            L.us_sel += (double) (ggml_time_us() - ts);
        }
    }
    const int n = L.n, d = L.d;
    GGML_ASSERT(a->ne[1] == 1 && a->ne[0] == d && b->ne[0] == n && nth <= 64);
    const int nsel = (int) L.sel.size();
    float *   y = L.part[ith].data();
    memset(y, 0, sizeof(float) * d);
    const int64_t t1 = ith == 0 ? ggml_time_us() : 0;
    const int     PF = g4s_pf * G4S_GROUP;
    for (;;) {
        int k0, k1;
        if (g4s_chunk > 0) { // v3: chunks from a shared counter, so a thread that is descheduled does not stall the layer
            k0 = L.next.fetch_add(g4s_chunk);
            if (k0 >= nsel) {
                break;
            }
            k1 = std::min(nsel, k0 + g4s_chunk);
        } else { // v2: one fixed range per thread
            k0 = (int) ((int64_t) nsel * ith / nth);
            k1 = (int) ((int64_t) nsel * (ith + 1) / nth);
        }
        if (PF > 0) {
            for (int q = k0; q < std::min(k1, k0 + PF); ++q) {
                g4s_prefetch_neuron(L, L.sel[q]);
            }
        }
        for (int kk = k0; kk < k1; kk += G4S_GROUP) {
            if (PF > 0) {
                for (int q = kk + PF; q < std::min(k1, kk + PF + G4S_GROUP); ++q) {
                    g4s_prefetch_neuron(L, L.sel[q]);
                }
            }
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
        if (g4s_chunk <= 0) {
            break;
        }
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
    L.us_wait += (double) (ggml_time_us() - t3);
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
    if (L.il == g4s_last && L.calls == 1) { // every sparse layer has now been prepared and mapped: start the decode-phase counters
        g4s_faults(&g4s_flt0[0], &g4s_flt0[1]);
        g4s_t_first = ggml_time_us();
    }
}

// ====================================================================================================================
// G4TICK: one timestamp (and the process's fault counters) per decoded token, in EVERY mode, dense included, so that
// both arms of a comparison are timed by the same clock and the first token (cache preparation) is left out.
// The op sits right before the LM head and copies its input through.
// ====================================================================================================================
extern "C" void g4prof_mark(void); // ggml-cpu.c (G4PROF): one call per decoded token
static std::vector<int64_t>   g4tick_t;
static std::vector<long long> g4tick_min, g4tick_maj;

static void g4tick_report() {
    const int n = (int) g4tick_t.size();
    if (n < 6) {
        return;
    }
    const int k0 = n / 3;
    fprintf(stderr, "G4TICK n=%d | tokens 2..n %.3f tok/s, minflt %lld, majflt %lld | last two thirds %.3f tok/s, minflt %lld, majflt %lld\n", n,
            (n - 1) * 1e6 / (double) (g4tick_t[n - 1] - g4tick_t[0]), g4tick_min[n - 1] - g4tick_min[0], g4tick_maj[n - 1] - g4tick_maj[0],
            (n - 1 - k0) * 1e6 / (double) (g4tick_t[n - 1] - g4tick_t[k0]), g4tick_min[n - 1] - g4tick_min[k0], g4tick_maj[n - 1] - g4tick_maj[k0]);
}

static void g4tick_op(ggml_tensor * dst, const ggml_tensor * a, int ith, int nth, void * ud) {
    (void) nth;
    (void) ud;
    if (ith != 0) {
        return;
    }
    memcpy(dst->data, a->data, ggml_nbytes(a));
    static bool reg = false;
    if (!reg) {
        reg = true;
        atexit(g4tick_report);
    }
    long long f[2];
    g4s_faults(&f[0], &f[1]);
    g4prof_mark();
    g4tick_t.push_back(ggml_time_us());
    g4tick_min.push_back(f[0]);
    g4tick_maj.push_back(f[1]);
}

// ====================================================================================================================
// G4_OVERLAY=<file.gguf>: replacement tensors for the loaded model, read from a second GGUF (same names and shapes,
// other storage types; built by tools/lean_pack.py). The overlay is memory-mapped; a replaced tensor's header is
// re-pointed at the overlay's bytes and the bytes it used to own are handed back to the kernel.
// A research switch: one binary and one model file run "as shipped" and "lean" back to back on a phone that has
// no room for a second 4.4 GB file. A product file would simply contain the lean tensors.
// ====================================================================================================================
static const void * g4o_model = nullptr; // the model the overlay was last applied to

static void g4o_release(void * data, size_t n) {
#if !defined(_WIN32)
    const size_t    pg  = (size_t) sysconf(_SC_PAGESIZE);
    const uintptr_t beg = ((uintptr_t) data + pg - 1) & ~(uintptr_t) (pg - 1);
    const uintptr_t end = ((uintptr_t) data + n) & ~(uintptr_t) (pg - 1);
    if (end > beg) {
#if defined(MADV_PAGEOUT)
        if (madvise((void *) beg, end - beg, MADV_PAGEOUT) != 0) {
            madvise((void *) beg, end - beg, MADV_DONTNEED);
        }
#else
        madvise((void *) beg, end - beg, MADV_DONTNEED);
#endif
    }
#else
    (void) data;
    (void) n;
#endif
}

static void g4o_apply(const llama_model & model) {
    if (g4o_model == (const void *) &model) {
        return;
    }
    const char * path = getenv("G4_OVERLAY");
    if (!path || !*path) {
        return;
    }
    if (model.layers.empty() || model.layers[0].ffn_down == nullptr || model.layers[0].ffn_down->data == nullptr) {
        return; // a metadata-only model (the memory-fit pass): nothing to re-point, and do not mark it as done
    }
    g4o_model = (const void *) &model;
    ggml_context *   meta = nullptr;
    gguf_init_params ip   = { /*.no_alloc =*/ true, /*.ctx =*/ &meta };
    gguf_context *   g    = gguf_init_from_file(path, ip);
    GGML_ASSERT(g != nullptr && meta != nullptr && "G4_OVERLAY: cannot read the overlay file");
    size_t          len  = 0;
    const uint8_t * base = g4s_map(path, &len);
    GGML_ASSERT(base != nullptr && "G4_OVERLAY: cannot map the overlay file");
    const size_t d0 = gguf_get_data_offset(g);
    long long    n = 0, b_old = 0, b_new = 0;
    auto swap = [&](ggml_tensor * t) {
        if (!t) {
            return;
        }
        const int64_t id = gguf_find_tensor(g, ggml_get_name(t));
        if (id < 0) {
            return;
        }
        const ggml_tensor * m = ggml_get_tensor(meta, ggml_get_name(t));
        GGML_ASSERT(m != nullptr);
        for (int i = 0; i < GGML_MAX_DIMS; ++i) {
            GGML_ASSERT(t->ne[i] == m->ne[i] && "G4_OVERLAY: shape mismatch");
        }
        void *       old_data  = t->data;
        const size_t old_bytes = ggml_nbytes(t);
        t->type                = m->type;
        for (int i = 0; i < GGML_MAX_DIMS; ++i) {
            t->nb[i] = m->nb[i];
        }
        t->data = (void *) (base + d0 + gguf_get_tensor_offset(g, id));
        GGML_ASSERT(d0 + gguf_get_tensor_offset(g, id) + ggml_nbytes(t) <= len);
        g4o_release(old_data, old_bytes);
        n += 1;
        b_old += (long long) old_bytes;
        b_new += (long long) ggml_nbytes(t);
    };
    for (size_t il = 0; il < model.layers.size(); ++il) {
        const auto & L = model.layers[il];
        swap(L.ffn_down);
        swap(L.ffn_gate);
        swap(L.ffn_up);
        swap(L.wq);
        swap(L.wk);
        swap(L.wv);
        swap(L.wo);
        swap(L.per_layer_inp_gate);
        swap(L.per_layer_proj);
    }
    swap(model.per_layer_model_proj);
    fprintf(stderr, "G4_OVERLAY: %lld tensors replaced from %s, %.1f MiB -> %.1f MiB\n", n, path, b_old / 1048576.0, b_new / 1048576.0);
}

// ====================================================================================================================
// G4H=<N>: the LM head as "every row at 1 bit, then the exact logit of a shortlist of N rows". Decode only.
//   scan    score_j = alpha_j * sum_i sign(e_ji) * (q_i + 0.5), q = the hidden state at 8 bits over its full range.
//           This is the `s1q8` sketch measured in E7H / E9 (N = 8192: the exact top token inside in 100 % of 5 960
//           positions). Computed with AND + popcount over the 8 bit-planes of q.
//   select  the N best scores
//   exact   for those rows, the model's own Q4_K row against the hidden state with the kernel the dense head uses
//           (so a shortlisted logit is bit-identical to the dense one); every other logit is set far below, which
//           the soft-cap turns into -30.
// G4H_CACHE=<dir>: the sketch (alpha, popcounts, sign bits: ~82 MiB) is built once from token_embd and then mapped.
// ====================================================================================================================
struct g4h_state {
    int                 N = 0;
    const ggml_tensor * W = nullptr;
    bool                ready = false;
    std::mutex          mtx;
    int                 V = 0, d = 0, nb = 0, rowsz = 0;
    const float *       alpha = nullptr;
    const uint16_t *    pc    = nullptr;
    const uint8_t *     bits  = nullptr;
    std::vector<uint8_t> own;
    std::atomic<int>    next{ 0 };
    long long           calls = 0, picked = 0;
    double              us_scan = 0.0, us_sel = 0.0, us_exact = 0.0, us_prep = 0.0;
};
static g4h_state g4h;
static int       g4h_init = 0; // 0 = not read, 1 = off, 2 = on

static void g4h_report() {
    if (g4h.calls > 0) {
        fprintf(stderr, "G4H N=%d | per token: scan %.2f ms, select %.2f ms, exact %.2f ms | rows re-scored per token %.0f | sketch prepared in %.0f ms\n", g4h.N,
                g4h.us_scan / g4h.calls / 1000.0, g4h.us_sel / g4h.calls / 1000.0, g4h.us_exact / g4h.calls / 1000.0, (double) g4h.picked / g4h.calls,
                g4h.us_prep / 1000.0);
    }
}

static bool g4h_on(const ggml_tensor * W, const ggml_tensor * W_s) {
    if (g4h_init == 0) {
        g4h_init = 1;
        const char * e = getenv("G4H");
        const int    N = e ? atoi(e) : 0;
        if (N > 0 && W && W_s == nullptr && W->type == GGML_TYPE_Q4_K && W->ne[0] % 256 == 0 && N < W->ne[1]) {
            g4h.N    = N;
            g4h.W    = W;
            g4h_init = 2;
            fprintf(stderr, "G4H enabled: 1-bit scan of %lld rows, exact logits for the best %d\n", (long long) W->ne[1], N);
            atexit(g4h_report);
        } else if (N > 0) {
            fprintf(stderr, "G4H requested but not applicable to this head (type / scale / size): dense head stays\n");
        }
    }
    if (g4h_init == 2 && g4h.W != W) { // a new model object (the first one may have been the metadata-only fit pass)
        std::lock_guard<std::mutex> lk(g4h.mtx);
        g4h.W     = W;
        g4h.ready = false;
    }
    return g4h_init == 2;
}

static void g4h_prepare() {
    const int64_t t0 = ggml_time_us();
    const ggml_tensor * W = g4h.W;
    g4h.V     = (int) W->ne[1];
    g4h.d     = (int) W->ne[0];
    g4h.nb    = g4h.d / 8;
    g4h.rowsz = (int) ggml_row_size(GGML_TYPE_Q4_K, g4h.d);
    const int    V = g4h.V, d = g4h.d, nb = g4h.nb;
    const size_t hdr = 4 * sizeof(int32_t), ab = (size_t) V * sizeof(float), pb = (size_t) V * sizeof(uint16_t);
    const size_t off_bits = (hdr + ab + pb + 63) / 64 * 64, total = off_bits + (size_t) V * nb;
    const char * dir = getenv("G4H_CACHE");
    char         path[1024] = { 0 };
    const uint8_t * base = nullptr;
    size_t          len  = 0;
    if (dir) {
        snprintf(path, sizeof(path), "%s/g4h_sketch.bin", dir);
        base = g4s_map(path, &len);
        if (base) {
            int32_t h[4];
            memcpy(h, base, sizeof(h));
            if (!(h[0] == 0x31483447 && h[1] == V && h[2] == d && len == total)) {
                base = nullptr;
            }
        }
    }
    if (!base) {
        g4h.own.assign(total, 0);
        const int32_t h[4] = { 0x31483447, V, d, 0 };
        memcpy(g4h.own.data(), h, sizeof(h));
        float *    al = (float *) (g4h.own.data() + hdr);
        uint16_t * pc = (uint16_t *) (g4h.own.data() + hdr + ab);
        uint8_t *  bt = g4h.own.data() + off_bits;
        std::vector<float> buf(d);
        for (int j = 0; j < V; ++j) {
            g4_row(W, j, buf.data());
            double    a = 0.0;
            int       c = 0;
            uint8_t * r = bt + (size_t) j * nb;
            for (int i = 0; i < d; ++i) {
                a += fabs((double) buf[i]);
                if (buf[i] >= 0.0f) {
                    r[i >> 3] |= (uint8_t) (1u << (i & 7));
                    c += 1;
                }
            }
            al[j] = (float) (a / d);
            pc[j] = (uint16_t) c;
        }
        bool mapped = false;
        if (dir) {
            FILE * f = fopen(path, "wb");
            if (f) {
                fwrite(g4h.own.data(), 1, total, f);
                fclose(f);
                base   = g4s_map(path, &len);
                mapped = base != nullptr && len == total;
            }
        }
        if (mapped) {
            std::vector<uint8_t>().swap(g4h.own);
        } else {
            base = g4h.own.data();
        }
    }
    g4h.alpha = (const float *) (base + hdr);
    g4h.pc    = (const uint16_t *) (base + hdr + ab);
    g4h.bits  = base + off_bits;
    g4h.us_prep = (double) (ggml_time_us() - t0);
}

// sum over b of 2^b * popcount(bits & plane_b): the sum of u_i over the dimensions whose sign bit is set
static inline uint32_t g4h_masked(const uint8_t * bits, const uint8_t * planes, int nb) {
#if defined(__ARM_NEON)
    // per plane: AND, per-byte popcount, byte-wise add. A byte lane gains at most 8 per step, so 31 steps fit in
    // u8 (nb = 320 is 20 steps); widening happens once per row instead of once per step.
    uint32_t s = 0;
    for (int o0 = 0; o0 < nb; o0 += 16 * 31) {
        const int  o1 = std::min(nb, o0 + 16 * 31);
        uint8x16_t acc[8];
        for (int b = 0; b < 8; ++b) {
            acc[b] = vdupq_n_u8(0);
        }
        for (int o = o0; o < o1; o += 16) {
            const uint8x16_t w = vld1q_u8(bits + o);
            for (int b = 0; b < 8; ++b) {
                acc[b] = vaddq_u8(acc[b], vcntq_u8(vandq_u8(w, vld1q_u8(planes + (size_t) b * nb + o))));
            }
        }
        for (int b = 0; b < 8; ++b) {
            s += (uint32_t) vaddlvq_u8(acc[b]) << b;
        }
    }
    return s;
#else
    uint32_t s = 0;
    for (int b = 0; b < 8; ++b) {
        const uint8_t * pl = planes + (size_t) b * nb;
        uint32_t        c  = 0;
        for (int o = 0; o < nb; o += 8) {
            uint64_t x, y;
            memcpy(&x, bits + o, 8);
            memcpy(&y, pl + o, 8);
            c += (uint32_t) __builtin_popcountll(x & y);
        }
        s += c << b;
    }
    return s;
#endif
}

// dst = scores [V]; src[0] = the hidden state [d, 1]
static void g4h_scan(ggml_tensor * dst, int ith, int nth, void * ud) {
    (void) ud;
    {
        std::lock_guard<std::mutex> lk(g4h.mtx);
        if (!g4h.ready) {
            g4h_prepare();
            g4h.ready = true;
        }
    }
    const int64_t t0 = ith == 0 ? ggml_time_us() : 0; // after the one-time sketch preparation
    const int     V = g4h.V, d = g4h.d, nb = g4h.nb;
    const float * h = (const float *) dst->src[0]->data;
    GGML_ASSERT(dst->src[0]->ne[0] == d && dst->ne[0] == V && nb % 16 == 0);
    float mx = 0.0f;
    for (int i = 0; i < d; ++i) {
        mx = std::max(mx, fabsf(h[i]));
    }
    const float          inv = mx > 0.0f ? 127.5f / mx : 0.0f;
    std::vector<uint8_t> planes((size_t) 8 * nb, 0);
    long long            Q = 0;
    for (int i = 0; i < d; ++i) {
        int q = (int) floorf(h[i] * inv);
        q     = q < -128 ? -128 : (q > 127 ? 127 : q);
        Q += q;
        const unsigned u = (unsigned) (q + 128);
        for (int b = 0; b < 8; ++b) {
            if ((u >> b) & 1u) {
                planes[(size_t) b * nb + (i >> 3)] |= (uint8_t) (1u << (i & 7));
            }
        }
    }
    float *   out = (float *) dst->data;
    const int j0 = (int) ((int64_t) V * ith / nth), j1 = (int) ((int64_t) V * (ith + 1) / nth);
    const float halfd = 0.5f * (float) d;
    for (int j = j0; j < j1; ++j) {
        const long long m = (long long) g4h_masked(g4h.bits + (size_t) j * nb, planes.data(), nb);
        out[j]            = g4h.alpha[j] * ((float) (2 * m - 255LL * g4h.pc[j] - Q) - halfd);
    }
    if (ith == 0) {
        g4h.us_scan += (double) (ggml_time_us() - t0);
    }
}

// dst = flags [V] (1 = re-score exactly); src[0] = scores [V]. One thread.
static void g4h_select(ggml_tensor * dst, int ith, int nth, void * ud) {
    (void) nth;
    (void) ud;
    if (ith != 0) {
        return;
    }
    const int64_t t0 = ggml_time_us();
    const int     V  = g4h.V, N = g4h.N;
    const float * sc = (const float *) dst->src[0]->data;
    // the cut is taken from every 8th score (the shortlist then holds N within a few per cent, which is all it needs)
    static std::vector<float> tmp;
    const int                 ST = 8, ns = V / ST, kth = std::max(1, std::min(ns, (int) (1.05 * N / ST))); // 5 % more than N
    tmp.resize(ns);
    for (int j = 0; j < ns; ++j) {
        tmp[j] = sc[(size_t) j * ST];
    }
    std::nth_element(tmp.begin(), tmp.begin() + (kth - 1), tmp.end(), [](float p, float q) { return p > q; });
    const float cut = tmp[kth - 1];
    float *     fl  = (float *) dst->data;
    long long   n   = 0;
    for (int j = 0; j < V; ++j) {
        const bool in = sc[j] >= cut;
        fl[j]         = in ? 1.0f : 0.0f;
        n += in ? 1 : 0;
    }
    g4h.next.store(0);
    g4h.picked += n;
    g4h.calls += 1;
    g4h.us_sel += (double) (ggml_time_us() - t0);
}

// dst = logits [V]; src[0] = the hidden state [d, 1]; src[1] = flags [V]
static void g4h_exact(ggml_tensor * dst, int ith, int nth, void * ud) {
    (void) nth;
    (void) ud;
    const int64_t t0 = ith == 0 ? ggml_time_us() : 0;
    static const ggml_type_traits_cpu * tc = ggml_get_type_traits_cpu(GGML_TYPE_Q4_K);
    static const ggml_type_traits_cpu * tq = ggml_get_type_traits_cpu(tc->vec_dot_type);
    const int       V = g4h.V, d = g4h.d, rs = g4h.rowsz;
    const float *   h  = (const float *) dst->src[0]->data;
    const float *   fl = (const float *) dst->src[1]->data;
    const uint8_t * W  = (const uint8_t *) g4h.W->data;
    float *         out = (float *) dst->data;
    std::vector<uint8_t> xq(ggml_row_size(tc->vec_dot_type, d));
    tq->from_float(h, xq.data(), d);
    const int CH = 2048;
    for (;;) {
        const int j0 = g4h.next.fetch_add(CH);
        if (j0 >= V) {
            break;
        }
        const int j1 = std::min(V, j0 + CH);
        for (int j = j0; j < j1; ++j) {
            if (fl[j] != 0.0f) {
                float u = 0.0f;
                tc->vec_dot(d, &u, 0, W + (size_t) j * rs, 0, xq.data(), 0, 1);
                out[j] = u;
            } else {
                out[j] = -1.0e9f;
            }
        }
    }
    if (ith == 0) {
        g4h.us_exact += (double) (ggml_time_us() - t0);
    }
}
