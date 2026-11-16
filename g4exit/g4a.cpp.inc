// ---- G4A (research, 2026-11-16): decode attention that reads the key cache once ---------------------------
// Inserted into ggml/src/ggml-cpu/ops.cpp by patch_g4s.py, in front of ggml_compute_forward_flash_attn_ext_f16.
//
// Upstream's split-KV decode path gives each thread a slice of the key cache and walks that slice once per
// QUERY head. With grouped-query attention (Gemma 4 E4B: 8 query heads on 2 key heads) every K and V row is
// pulled from DRAM 4 times per layer, and the slices are cut over the cache size, not over the live keys.
// Measured on the A54 at 17.5k keys: 271.5 ms per token in the 7 global-attention calls (JOURNAL 11-16 17:25).
//
// Here the slice is walked in blocks of G4A_BLK keys and every query head consumes a block before the next
// block is touched: K and V come from DRAM once and from L1/L2 for the other heads.
//
//   G4A=0            upstream path
//   G4A=1            loop interchange only. Same float operations in the same per-head order as upstream:
//                    bit-identical output.
//   G4A=2            1 + slices cut by LIVE key count (deterministic; the order partials merge in changes).
//   G4A=3 (default)  2 + two-phase blocks: scores of a block first, one rescale per block, vector exp.
//                    Same mathematics, different rounding.
//   G4A_BLK=<keys>   block size (default 32: K + V of both key heads = 128 KiB, inside L2)
//   G4A_EPS=<nats>   skip the V row of a key whose score is more than <nats> under the running maximum
//                    (0 = off, the default). At 18 nats a skipped key weighs under 1.6e-8 of the top key.
//   G4A_STAT=1       print visit / skip counters at exit

static int g4a_env_int(const char * name, int def) {
    const char * e = getenv(name);
    return e ? atoi(e) : def;
}

static int g4a_mode(void) {
    static const int v = g4a_env_int("G4A", 3);
    return v;
}

static int g4a_blk(void) {
    static const int v = std::min(256, std::max(1, g4a_env_int("G4A_BLK", 32)));
    return v;
}

static float g4a_eps(void) {
    static const float v = [] {
        const char * e = getenv("G4A_EPS");
        return e ? (float) atof(e) : 0.0f;
    }();
    return v;
}

static std::atomic<int64_t> g4a_n_calls{0};
static std::atomic<int64_t> g4a_n_visit{0};
static std::atomic<int64_t> g4a_n_skip{0};

static void g4a_stat_print(void) {
    const long long c = (long long) g4a_n_calls.load();
    const long long n = (long long) g4a_n_visit.load();
    const long long s = (long long) g4a_n_skip.load();
    fprintf(stderr, "G4A mode %d blk %d eps %.1f | calls %lld | head-key visits %lld | V rows skipped %lld (%.1f %%)\n",
            g4a_mode(), g4a_blk(), (double) g4a_eps(), c, n, s, n > 0 ? 100.0 * (double) s / (double) n : 0.0);
}

static bool g4a_stat(void) {
    static const bool v = [] {
        const bool on = g4a_env_int("G4A_STAT", 0) > 0;
        if (on) {
            atexit(g4a_stat_print);
        }
        return on;
    }();
    return v;
}

static void g4a_decode(const ggml_compute_params * params, ggml_tensor * dst) {
    const ggml_tensor * q     = dst->src[0];
    const ggml_tensor * k     = dst->src[1];
    const ggml_tensor * v     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    GGML_TENSOR_LOCALS(int64_t, neq, q, ne)
    GGML_TENSOR_LOCALS(size_t,  nbq, q, nb)
    GGML_TENSOR_LOCALS(int64_t, nek, k, ne)
    GGML_TENSOR_LOCALS(size_t,  nbk, k, nb)
    GGML_TENSOR_LOCALS(int64_t, nev, v, ne)
    GGML_TENSOR_LOCALS(size_t,  nbv, v, nb)

    const int64_t DK = nek0;
    const int64_t DV = nev0;
    const int64_t H  = neq2;

    const int ith  = params->ith;
    const int nth  = params->nth;
    const int mode = g4a_mode();

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
    GGML_ASSERT((v->type == GGML_TYPE_F16 || v->type == GGML_TYPE_F32) && "g4a: unsupported V-type");

    // partials in upstream's layout, so that its reducer can be reused: [q_head][slice][M, S, VKQ[DV]]
    const int64_t partial_size   = 2 + DV;
    float *       partials_base  = (float *) params->wdata + nth * (DK + 2*DV + CACHE_LINE_SIZE_F32);
    const int64_t partial_stride = nth * partial_size;
    float *       my_partials    = partials_base + ith * partial_size;

    // ---- this thread's slice of the key cache
    int64_t ic_start = 0;
    int64_t ic_end   = 0;

    // decode: one query row, one stream. The row of head 0 decides the cut; every head still tests its own mask.
    const ggml_fp16_t * mp_cut = mask ? (const ggml_fp16_t *) mask->data : nullptr;
    const ggml_fp16_t   neg_inf16 = GGML_CPU_FP32_TO_FP16(-INFINITY);

    if (mode >= 2 && mp_cut) {
        int64_t n_live = 0;
        for (int64_t ic = 0; ic < nek1; ++ic) {
            n_live += (mp_cut[ic] != neg_inf16);
        }
        const int64_t per     = (n_live + nth - 1) / nth;
        const int64_t lo_rank = (int64_t) ith * per;
        const int64_t hi_rank = std::min(lo_rank + per, n_live);
        if (lo_rank < hi_rank) {
            int64_t r = 0;
            for (int64_t ic = 0; ic < nek1; ++ic) {
                if (mp_cut[ic] == neg_inf16) {
                    continue;
                }
                if (r == lo_rank) {
                    ic_start = ic;
                }
                r++;
                if (r == hi_rank) {
                    ic_end = ic + 1;
                    break;
                }
            }
        }
    } else {
        const int64_t chunk_size = (nek1 + nth - 1) / nth;
        ic_start = (int64_t) ith * chunk_size;
        ic_end   = std::min(ic_start + chunk_size, nek1);
        if (ic_start >= nek1) {
            ic_start = ic_end = 0;
        }
    }

    // ---- per-head set-up: Q in the dot type, slope, empty partial
    static thread_local std::vector<float> qbuf;     // H x DK, sized for the widest dot type
    static thread_local std::vector<float> slopes;
    qbuf.resize((size_t) (H * DK));
    slopes.resize((size_t) H);

    for (int64_t h = 0; h < H; ++h) {
        slopes[(size_t) h] = (max_bias > 0.0f) ? ((uint32_t) h < n_head_log2 ? powf(m0, h + 1) : powf(m1, 2*(h - n_head_log2) + 1)) : 1.0f;

        const float * pq = (const float *) ((const char *) q->data + h*nbq2);   // iq1 = 0, iq3 = 0
        q_to_vec_dot(pq, (ggml_fp16_t *) (qbuf.data() + h*DK), DK);

        float * P = my_partials + h * partial_stride;
        P[0] = -INFINITY;
        P[1] = 0.0f;
        memset(P + 2, 0, (size_t) DV * sizeof(float));
    }

    const float   eps = g4a_eps();
    const int64_t BLK = g4a_blk();

    int64_t n_visit = 0;
    int64_t n_skip  = 0;

    float   sc[256];
    float   wt[256];
    int64_t ix[256];

    for (int64_t b0 = ic_start; b0 < ic_end; b0 += BLK) {
        const int64_t b1 = std::min(b0 + BLK, ic_end);

        for (int64_t h = 0; h < H; ++h) {
            float * P     = my_partials + h * partial_stride;
            float   M     = P[0];
            float   S     = P[1];
            float * VKQ32 = P + 2;

            const float slope = slopes[(size_t) h];
            const void * Q_q  = (const void *) (qbuf.data() + h*DK);

            const ggml_fp16_t * mp = mask ? (const ggml_fp16_t *) ((const char *) mask->data + (h % mask->ne[2])*mask->nb[2]) : nullptr;

            const int64_t ik2 = h / rk2;
            const int64_t iv2 = h / rv2;

            if (mode <= 2) {
                // upstream's online softmax, key by key
                for (int64_t ic = b0; ic < b1; ++ic) {
                    const float mv = mp ? slope*GGML_CPU_FP16_TO_FP32(mp[ic]) : 0.0f;
                    if (mv == -INFINITY) {
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

                    n_visit++;
                    if (eps > 0.0f && s < M - eps) {
                        n_skip++;
                        continue;
                    }

                    const float Mold = M;
                    float ms = 1.0f;
                    float vs = 1.0f;

                    const char * v_data = (const char *) v->data + (ic*nbv1 + iv2*nbv2);

                    if (s > M) {
                        M  = s;
                        ms = expf(Mold - M);
                        ggml_vec_scale_f32(DV, VKQ32, ms);
                    } else {
                        vs = expf(s - M);
                    }

                    if (v->type == GGML_TYPE_F16) {
                        ggml_vec_mad_f16_f32(DV, VKQ32, (const ggml_fp16_t *) v_data, vs);
                    } else {
                        ggml_vec_mad_f32(DV, VKQ32, (const float *) v_data, vs);
                    }

                    S = S*ms + vs;
                }
            } else {
                // two-phase block: all scores, one rescale, vector exp, then the V rows
                int   n  = 0;
                float bm = -INFINITY;
                for (int64_t ic = b0; ic < b1; ++ic) {
                    const float mv = mp ? slope*GGML_CPU_FP16_TO_FP32(mp[ic]) : 0.0f;
                    if (mv == -INFINITY) {
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

                    sc[n] = s;
                    ix[n] = ic;
                    n++;
                    if (s > bm) {
                        bm = s;
                    }
                }
                if (n > 0) {
                    n_visit += n;
                    if (bm > M) {
                        if (S != 0.0f) {
                            const float ms = expf(M - bm);
                            ggml_vec_scale_f32(DV, VKQ32, ms);
                            S *= ms;
                        }
                        M = bm;
                    }
                    if (eps > 0.0f && bm < M - eps) {
                        // the whole block is under the threshold: no exp, no V rows
                        n_skip += n;
                    } else {
                        S += (float) ggml_vec_soft_max_f32(n, wt, sc, M);
                        for (int i = 0; i < n; ++i) {
                            if (eps > 0.0f && sc[i] < M - eps) {
                                n_skip++;
                                continue;
                            }
                            const char * v_data = (const char *) v->data + (ix[i]*nbv1 + iv2*nbv2);
                            if (v->type == GGML_TYPE_F16) {
                                ggml_vec_mad_f16_f32(DV, VKQ32, (const ggml_fp16_t *) v_data, wt[i]);
                            } else {
                                ggml_vec_mad_f32(DV, VKQ32, (const float *) v_data, wt[i]);
                            }
                        }
                    }
                }
            }

            P[0] = M;
            P[1] = S;
        }
    }

    // sinks: once, on the first slice (upstream: the slice that starts at key 0)
    if (sinks && ith == 0) {
        for (int64_t h = 0; h < H; ++h) {
            float * P     = my_partials + h * partial_stride;
            float   M     = P[0];
            float   S     = P[1];
            float * VKQ32 = P + 2;

            const float s = ((const float *) ((const char *) sinks->data))[h];

            float ms = 1.0f;
            float vs = 1.0f;

            if (s > M) {
                ms = expf(M - s);
                M  = s;
                ggml_vec_scale_f32(DV, VKQ32, ms);
            } else {
                vs = expf(s - M);
            }

            S = S*ms + vs;

            P[0] = M;
            P[1] = S;
        }
    }

    if (g4a_stat()) {
        if (ith == 0) {
            g4a_n_calls.fetch_add(1, std::memory_order_relaxed);
        }
        g4a_n_visit.fetch_add(n_visit, std::memory_order_relaxed);
        g4a_n_skip.fetch_add(n_skip, std::memory_order_relaxed);
    }

    ggml_barrier(params->threadpool);

    // chunk_size 1: the reducer skips slices by "chunk_idx*chunk_size >= nek1" and by S == 0; empty slices have S == 0
    ggml_flash_attn_ext_reduce_partials(params, dst, nth, 1);
}

