// ---- G4A mode 4 (research, 2026-11-16): exact scores for every key, V rows only where the mass is -----------
// Inserted into ggml/src/ggml-cpu/ops.cpp by patch_g4s.py, after g4a.cpp.inc.
//
// The K pass is the same as in mode 3 (every live key gets its exact score, every query head). Then the
// softmax weights of ALL keys are computed (so the normaliser S is exact), a histogram of attention mass by
// "nats under the maximum" is merged across threads, and the lightest keys are dropped while the mass they
// carry stays under G4A_DELTA of the total. Only the remaining keys have their V row read and accumulated.
//
// Error bound, by construction: output = (sum over kept keys of w_i V_i) / S with the exact S, so the output
// moves by at most delta * max|V| per component. Default delta 1e-4, far under the f16 rounding of V itself.
//
//   G4A=4            this path (falls back to mode 3 with sinks, more than 16 threads or more than 64 heads)
//   G4A_DELTA=<f>    mass allowed to be dropped, as a fraction of the total (default 1e-4; 0 = keep every key)

#define G4A_MAX_T 16
#define G4A_MAX_H 64
#define G4A_BINS  64   // half-nat bins: bin b holds keys with 2*(M - s) in [b, b+1); the last bin is open-ended

static float g4a_hist[G4A_MAX_T][G4A_MAX_H][G4A_BINS];

static float g4a_delta(void) {
    static const float v = [] {
        const char * e = getenv("G4A_DELTA");
        return e ? (float) atof(e) : 1e-4f;
    }();
    return v;
}

static std::atomic<int64_t> g4a_n_mass_visit{0};
static std::atomic<int64_t> g4a_n_mass_vrows{0};

static void g4a_mass_stat_print(void) {
    const long long n = (long long) g4a_n_mass_visit.load();
    const long long r = (long long) g4a_n_mass_vrows.load();
    fprintf(stderr, "G4A mode 4 delta %g | head-key visits %lld | V rows accumulated %lld (%.2f %%)\n",
            (double) g4a_delta(), n, r, n > 0 ? 100.0 * (double) r / (double) n : 0.0);
}

static bool g4a_mass_stat(void) {
    static const bool v = [] {
        const bool on = g4a_env_int("G4A_STAT", 0) > 0;
        if (on) {
            atexit(g4a_mass_stat_print);
        }
        return on;
    }();
    return v;
}

static bool g4a_mass_ok(const ggml_compute_params * params, const ggml_tensor * dst) {
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * v = dst->src[2];
    return dst->src[4] == nullptr && params->nth <= G4A_MAX_T && q->ne[2] <= G4A_MAX_H &&
           (v->type == GGML_TYPE_F16 || v->type == GGML_TYPE_F32);
}

static void g4a_decode_mass(const ggml_compute_params * params, ggml_tensor * dst) {
    const ggml_tensor * q    = dst->src[0];
    const ggml_tensor * k    = dst->src[1];
    const ggml_tensor * v    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    GGML_TENSOR_LOCALS(int64_t, neq, q, ne)
    GGML_TENSOR_LOCALS(size_t,  nbq, q, nb)
    GGML_TENSOR_LOCALS(int64_t, nek, k, ne)
    GGML_TENSOR_LOCALS(size_t,  nbk, k, nb)
    GGML_TENSOR_LOCALS(int64_t, nev, v, ne)
    GGML_TENSOR_LOCALS(size_t,  nbv, v, nb)

    const int64_t DK = nek0;
    const int64_t DV = nev0;
    const int64_t H  = neq2;

    const int ith = params->ith;
    const int nth = params->nth;

    const int64_t rk2 = neq2/nek2;
    const int64_t rv2 = neq2/nev2;

    float scale         = 1.0f;
    float max_bias      = 0.0f;
    float logit_softcap = 0.0f;

    memcpy(&scale,         (float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias,      (float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (float *) dst->op_params + 2, sizeof(float));

    if (logit_softcap != 0) {
        scale /= logit_softcap;
    }

    const uint32_t n_head      = (uint32_t) neq2;
    const uint32_t n_head_log2 = 1u << (uint32_t) floor(log2(n_head));

    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    ggml_type         const k_vec_dot_type = ggml_get_type_traits_cpu(k->type)->vec_dot_type;
    ggml_from_float_t const q_to_vec_dot   = ggml_get_type_traits_cpu(k_vec_dot_type)->from_float;
    ggml_vec_dot_t    const kq_vec_dot     = ggml_get_type_traits_cpu(k->type)->vec_dot;

    GGML_ASSERT(q_to_vec_dot && "g4a: unsupported K-type");

    // partials in upstream's layout: [q_head][slice][M, S, VKQ[DV]]
    const int64_t partial_size   = 2 + DV;
    float *       partials_base  = (float *) params->wdata + nth * (DK + 2*DV + CACHE_LINE_SIZE_F32);
    const int64_t partial_stride = nth * partial_size;
    float *       my_partials    = partials_base + ith * partial_size;

    // ---- this thread's slice: live keys (by the mask row of head 0), cut by rank
    static thread_local std::vector<int32_t> ixbuf;
    static thread_local std::vector<float>   wbuf;    // H x n: scores, then weights
    static thread_local std::vector<float>   tbuf;    // n: one head's weights before they replace its scores
    static thread_local std::vector<float>   qbuf;    // H x DK

    const ggml_fp16_t * mp_cut    = mask ? (const ggml_fp16_t *) mask->data : nullptr;
    const ggml_fp16_t   neg_inf16 = GGML_CPU_FP32_TO_FP16(-INFINITY);

    int64_t n_live = nek1;
    if (mp_cut) {
        n_live = 0;
        for (int64_t ic = 0; ic < nek1; ++ic) {
            n_live += (mp_cut[ic] != neg_inf16);
        }
    }
    const int64_t per     = (n_live + nth - 1) / nth;
    const int64_t lo_rank = (int64_t) ith * per;
    const int64_t hi_rank = std::min(lo_rank + per, n_live);

    ixbuf.clear();
    if (lo_rank < hi_rank) {
        ixbuf.reserve((size_t) (hi_rank - lo_rank));
        int64_t r = 0;
        for (int64_t ic = 0; ic < nek1 && r < hi_rank; ++ic) {
            if (mp_cut && mp_cut[ic] == neg_inf16) {
                continue;
            }
            if (r >= lo_rank) {
                ixbuf.push_back((int32_t) ic);
            }
            r++;
        }
    }
    const int64_t n = (int64_t) ixbuf.size();

    wbuf.resize((size_t) std::max<int64_t>(1, H * n));
    tbuf.resize((size_t) std::max<int64_t>(1, n));
    qbuf.resize((size_t) (H * DK));

    float Mloc[G4A_MAX_H];
    float slopes[G4A_MAX_H];
    for (int64_t h = 0; h < H; ++h) {
        Mloc[h]   = -INFINITY;
        slopes[h] = (max_bias > 0.0f) ? ((uint32_t) h < n_head_log2 ? powf(m0, h + 1) : powf(m1, 2*(h - n_head_log2) + 1)) : 1.0f;
        const float * pq = (const float *) ((const char *) q->data + h*nbq2);   // iq1 = 0, iq3 = 0
        q_to_vec_dot(pq, (ggml_fp16_t *) (qbuf.data() + h*DK), DK);
    }

    // ---- phase A: exact scores, key blocks outside, query heads inside (K comes from DRAM once)
    const int64_t BLK = g4a_blk();
    for (int64_t j0 = 0; j0 < n; j0 += BLK) {
        const int64_t j1 = std::min(j0 + BLK, n);
        for (int64_t h = 0; h < H; ++h) {
            const float   slope = slopes[h];
            const void *  Q_q   = (const void *) (qbuf.data() + h*DK);
            const int64_t ik2   = h / rk2;
            const ggml_fp16_t * mp = mask ? (const ggml_fp16_t *) ((const char *) mask->data + (h % mask->ne[2])*mask->nb[2]) : nullptr;
            float * w  = wbuf.data() + h*n;
            float   bm = Mloc[h];
            for (int64_t j = j0; j < j1; ++j) {
                const int64_t ic = ixbuf[(size_t) j];
                const float   mv = mp ? slope*GGML_CPU_FP16_TO_FP32(mp[ic]) : 0.0f;
                if (mv == -INFINITY) {
                    w[j] = -INFINITY;
                    continue;
                }
                float s;
                const char * k_data = (const char *) k->data + (ic*nbk1 + ik2*nbk2);
                kq_vec_dot(DK, &s, 0, k_data, 0, Q_q, 0, 1);
                s = s*scale;
                if (logit_softcap != 0.0f) {
                    s = logit_softcap*tanhf(s);
                }
                s += mv;
                w[j] = s;
                if (s > bm) {
                    bm = s;
                }
            }
            Mloc[h] = bm;
        }
    }
    for (int64_t h = 0; h < H; ++h) {
        my_partials[h * partial_stride] = Mloc[h];
    }

    ggml_barrier(params->threadpool);

    // ---- phase B1: the maximum over all slices, weights of every key, mass histogram
    float Mg[G4A_MAX_H];
    for (int64_t h = 0; h < H; ++h) {
        float m = -INFINITY;
        for (int t = 0; t < nth; ++t) {
            const float mt = (partials_base + t * partial_size + h * partial_stride)[0];
            if (mt > m) {
                m = mt;
            }
        }
        Mg[h] = m;

        float * hist = g4a_hist[ith][h];
        memset(hist, 0, sizeof(float) * G4A_BINS);

        float * w = wbuf.data() + h*n;
        float   S = 0.0f;
        if (n > 0 && m > -INFINITY) {
            S = (float) ggml_vec_soft_max_f32((int) n, tbuf.data(), w, m);
            for (int64_t j = 0; j < n; ++j) {
                const float d = (m - w[j]) * 2.0f;
                const int   b = d >= (float) (G4A_BINS - 1) ? G4A_BINS - 1 : (int) d;
                hist[b] += tbuf[(size_t) j];
            }
            memcpy(w, tbuf.data(), (size_t) n * sizeof(float));
        } else if (n > 0) {
            memset(w, 0, (size_t) n * sizeof(float));
        }
        (my_partials + h * partial_stride)[1] = S;
    }

    ggml_barrier(params->threadpool);

    // ---- phase B2: per head, the weight under which keys are dropped (same value in every thread)
    const float delta = g4a_delta();
    float wthr[G4A_MAX_H];
    for (int64_t h = 0; h < H; ++h) {
        float S = 0.0f;
        for (int t = 0; t < nth; ++t) {
            S += (partials_base + t * partial_size + h * partial_stride)[1];
        }
        int cut = G4A_BINS;   // bins >= cut are dropped
        if (delta > 0.0f && S > 0.0f) {
            const float budget = delta * S;
            float acc = 0.0f;
            for (int b = G4A_BINS - 1; b >= 1; --b) {
                float hb = 0.0f;
                for (int t = 0; t < nth; ++t) {
                    hb += g4a_hist[t][h][b];
                }
                if (acc + hb > budget) {
                    break;
                }
                acc += hb;
                cut = b;
            }
        }
        wthr[h] = cut >= G4A_BINS ? 0.0f : expf(-0.5f * (float) cut);
    }

    // ---- V rows of the keys that are kept; heads of one key head share the row while it is in L1
    for (int64_t h = 0; h < H; ++h) {
        float * P = my_partials + h * partial_stride;
        P[0] = Mg[h];
        memset(P + 2, 0, (size_t) DV * sizeof(float));
    }
    int64_t n_rows = 0;
    for (int64_t j = 0; j < n; ++j) {
        const int64_t ic = ixbuf[(size_t) j];
        for (int64_t h = 0; h < H; ++h) {
            const float wj = wbuf[(size_t) (h*n + j)];
            if (!(wj > wthr[h])) {
                continue;
            }
            const int64_t iv2    = h / rv2;
            const char *  v_data = (const char *) v->data + (ic*nbv1 + iv2*nbv2);
            float *       VKQ32  = my_partials + h * partial_stride + 2;
            if (v->type == GGML_TYPE_F16) {
                ggml_vec_mad_f16_f32(DV, VKQ32, (const ggml_fp16_t *) v_data, wj);
            } else {
                ggml_vec_mad_f32(DV, VKQ32, (const float *) v_data, wj);
            }
            n_rows++;
        }
    }

    if (g4a_mass_stat()) {
        g4a_n_mass_visit.fetch_add(n * H, std::memory_order_relaxed);
        g4a_n_mass_vrows.fetch_add(n_rows, std::memory_order_relaxed);
    }

    ggml_barrier(params->threadpool);

    // every slice carries the same M, so the reducer's rescale factors are exactly 1 and it sums S and VKQ
    ggml_flash_attn_ext_reduce_partials(params, dst, nth, 1);
}

