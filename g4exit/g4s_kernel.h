// g4s_kernel.h -- the "down" half of the gate-first sparse FFN.
//
// y += sum_g h[g] * W_down[:, j_g]   for a group of up to 16 selected neurons.
//
// W_down is stored NEURON-MAJOR: record j holds column j of W_down, quantised as an ordinary ggml Q4_K row (so its
// quality is that of Q4_K and ggml's own dequantiser is the reference). A dense matvec cannot skip columns of a
// row-major quantised matrix; this layout can.
//
// Q4_K: 256 weights per 144-byte block = 8 sub-blocks of 32, each with a 6-bit scale sc and a 6-bit min m:
//     w = d * sc * q  -  dmin * m        (q = 4-bit code; d, dmin = fp16 per block)
// For a group of neurons and one sub-block the kernel needs  sum_g (h_g d_g sc_g) q_g[i]  -  sum_g h_g dmin_g m_g.
// The multipliers (h_g d_g sc_g) are quantised to int8 against the group's largest one, so 16 neurons accumulate in
// int16 without overflow (16 * 15 * 127 = 30480), then one float flush per sub-block. The second sum is one float.
//
// Record layout (G4S_REC bytes for n = 2560):
//   [ Q4_K row, 10 blocks, 1440 B ][ sc: u8 x 80 ][ m: u8 x 80 ][ d: f32 x 10 ][ dmin: f32 x 10 ]
// The unpacked scale tables sit next to the row so that one neuron is one contiguous read.
#pragma once
#include <math.h>
#include <stdint.h>
#include <string.h>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#define G4S_BLOCK 144 // bytes per Q4_K block of 256 weights
#define G4S_QS 16     // offset of the 128 code bytes inside a block (2 x fp16 + 12 scale bytes)
#define G4S_GROUP 16

static inline int g4s_rec_size(int n) { // n weights per row, n % 256 == 0
    const int nb = n / 256;
    return nb * G4S_BLOCK + nb * 8 * 2 + nb * 4 * 2;
}

static inline void g4s_scale_min(int j, const uint8_t * q, uint8_t * d, uint8_t * m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (uint8_t) ((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
        *m = (uint8_t) ((q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4));
    }
}

// build one record from a Q4_K row of n weights
static inline void g4s_build_record(const uint8_t * row, int n, float (*fp16_to_f32)(uint16_t), uint8_t * rec) {
    const int nb = n / 256;
    memcpy(rec, row, (size_t) nb * G4S_BLOCK);
    uint8_t * sc = rec + (size_t) nb * G4S_BLOCK;
    uint8_t * mm = sc + nb * 8;
    uint8_t * fd = mm + nb * 8;
    uint8_t * fm = fd + nb * 4;
    for (int b = 0; b < nb; ++b) {
        const uint8_t * blk = row + (size_t) b * G4S_BLOCK;
        uint16_t        d16, m16;
        memcpy(&d16, blk, 2);
        memcpy(&m16, blk + 2, 2);
        const float d = fp16_to_f32(d16), dmin = fp16_to_f32(m16);
        memcpy(fd + 4 * b, &d, 4);
        memcpy(fm + 4 * b, &dmin, 4);
        for (int j = 0; j < 8; ++j) {
            g4s_scale_min(j, blk + 4, &sc[b * 8 + j], &mm[b * 8 + j]);
        }
    }
}

// recs[g]: record of neuron g; h[g]: its activation; ng <= 16; n = weights per row; y[n] is accumulated into
static inline void g4s_axpy_group(const uint8_t * const * recs, const float * h, int ng, int n, float * y) {
    const int nb = n / 256;
    const size_t o_sc = (size_t) nb * G4S_BLOCK, o_mm = o_sc + nb * 8, o_d = o_mm + nb * 8, o_m = o_d + nb * 4;
    float tb[G4S_GROUP], ub[G4S_GROUP]; // h*d and h*dmin for the current block
    for (int g = ng; g < G4S_GROUP; ++g) {
        tb[g] = 0.0f;
        ub[g] = 0.0f;
    }
    for (int c = 0; c < n; c += 64) {
        const int    blk = c >> 8, pair = (c & 255) >> 6, sb = blk * 8 + 2 * pair;
        const size_t qo  = (size_t) blk * G4S_BLOCK + G4S_QS + (size_t) pair * 32;
        if (pair == 0) {
            for (int g = 0; g < ng; ++g) {
                float d, m;
                memcpy(&d, recs[g] + o_d + 4 * blk, 4);
                memcpy(&m, recs[g] + o_m + 4 * blk, 4);
                tb[g] = h[g] * d;
                ub[g] = h[g] * m;
            }
        }
        uint8_t t0[G4S_GROUP] = { 0 }, t1[G4S_GROUP] = { 0 }, n0[G4S_GROUP] = { 0 }, n1[G4S_GROUP] = { 0 };
        for (int g = 0; g < ng; ++g) {
            const uint8_t * r = recs[g];
            t0[g] = r[o_sc + sb];
            t1[g] = r[o_sc + sb + 1];
            n0[g] = r[o_mm + sb];
            n1[g] = r[o_mm + sb + 1];
        }
        int8_t s0[G4S_GROUP], s1[G4S_GROUP];
        float  q0, q1, off0, off1;
#if defined(__ARM_NEON)
        {
            const float32x4_t T[4] = { vld1q_f32(tb), vld1q_f32(tb + 4), vld1q_f32(tb + 8), vld1q_f32(tb + 12) };
            const float32x4_t U[4] = { vld1q_f32(ub), vld1q_f32(ub + 4), vld1q_f32(ub + 8), vld1q_f32(ub + 12) };
            const uint8x16_t  in[4] = { vld1q_u8(t0), vld1q_u8(t1), vld1q_u8(n0), vld1q_u8(n1) };
            float32x4_t       F[4][4];
            for (int k = 0; k < 4; ++k) {
                const uint16x8_t lo = vmovl_u8(vget_low_u8(in[k])), hi = vmovl_high_u8(in[k]);
                F[k][0] = vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo)));
                F[k][1] = vcvtq_f32_u32(vmovl_high_u16(lo));
                F[k][2] = vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi)));
                F[k][3] = vcvtq_f32_u32(vmovl_high_u16(hi));
            }
            float32x4_t M0[4], M1[4], mx0 = vdupq_n_f32(1e-30f), mx1 = mx0, a0 = vdupq_n_f32(0.0f), a1 = a0;
            for (int i = 0; i < 4; ++i) {
                M0[i] = vmulq_f32(T[i], F[0][i]);
                M1[i] = vmulq_f32(T[i], F[1][i]);
                mx0   = vmaxq_f32(mx0, vabsq_f32(M0[i]));
                mx1   = vmaxq_f32(mx1, vabsq_f32(M1[i]));
                a0    = vfmaq_f32(a0, U[i], F[2][i]);
                a1    = vfmaq_f32(a1, U[i], F[3][i]);
            }
            const float m0 = vmaxvq_f32(mx0), m1 = vmaxvq_f32(mx1);
            const float r0 = 127.0f / m0, r1 = 127.0f / m1;
            q0   = m0 / 127.0f;
            q1   = m1 / 127.0f;
            off0 = vaddvq_f32(a0);
            off1 = vaddvq_f32(a1);
            for (int i = 0; i < 4; ++i) {
                const int32x4_t i0 = vcvtnq_s32_f32(vmulq_n_f32(M0[i], r0)), i1 = vcvtnq_s32_f32(vmulq_n_f32(M1[i], r1));
                const int16x4_t h0 = vmovn_s32(i0), h1 = vmovn_s32(i1);
                int16_t         w0[4], w1[4];
                vst1_s16(w0, h0);
                vst1_s16(w1, h1);
                for (int l = 0; l < 4; ++l) {
                    s0[4 * i + l] = (int8_t) w0[l];
                    s1[4 * i + l] = (int8_t) w1[l];
                }
            }
        }
        const uint8x16_t mk = vdupq_n_u8(0x0F);
        int16x8_t        aL0 = vdupq_n_s16(0), aL1 = aL0, aL2 = aL0, aL3 = aL0, aH0 = aL0, aH1 = aL0, aH2 = aL0, aH3 = aL0;
        for (int g = 0; g < ng; ++g) {
            const uint8_t *  q   = recs[g] + qo;
            const uint8x16_t v0  = vld1q_u8(q), v1 = vld1q_u8(q + 16);
            const int8x16_t  l0  = vreinterpretq_s8_u8(vandq_u8(v0, mk)), l1 = vreinterpretq_s8_u8(vandq_u8(v1, mk));
            const int8x16_t  h0  = vreinterpretq_s8_u8(vshrq_n_u8(v0, 4)), h1 = vreinterpretq_s8_u8(vshrq_n_u8(v1, 4));
            const int8x8_t   sv0 = vdup_n_s8(s0[g]), sv1 = vdup_n_s8(s1[g]);
            aL0 = vmlal_s8(aL0, vget_low_s8(l0), sv0);
            aL1 = vmlal_s8(aL1, vget_high_s8(l0), sv0);
            aL2 = vmlal_s8(aL2, vget_low_s8(l1), sv0);
            aL3 = vmlal_s8(aL3, vget_high_s8(l1), sv0);
            aH0 = vmlal_s8(aH0, vget_low_s8(h0), sv1);
            aH1 = vmlal_s8(aH1, vget_high_s8(h0), sv1);
            aH2 = vmlal_s8(aH2, vget_low_s8(h1), sv1);
            aH3 = vmlal_s8(aH3, vget_high_s8(h1), sv1);
        }
        const int16x8_t   acc[8] = { aL0, aL1, aL2, aL3, aH0, aH1, aH2, aH3 };
        const float32x4_t vo0 = vdupq_n_f32(off0), vo1 = vdupq_n_f32(off1);
        for (int k = 0; k < 8; ++k) {
            float *           yy = y + c + 8 * k;
            const float       qs = k < 4 ? q0 : q1;
            const float32x4_t vo = k < 4 ? vo0 : vo1;
            const float32x4_t lo = vcvtq_f32_s32(vmovl_s16(vget_low_s16(acc[k])));
            const float32x4_t hi = vcvtq_f32_s32(vmovl_high_s16(acc[k]));
            vst1q_f32(yy, vsubq_f32(vfmaq_n_f32(vld1q_f32(yy), lo, qs), vo));
            vst1q_f32(yy + 4, vsubq_f32(vfmaq_n_f32(vld1q_f32(yy + 4), hi, qs), vo));
        }
#else
        {
            float m0[G4S_GROUP], m1[G4S_GROUP], mx0 = 1e-30f, mx1 = 1e-30f;
            off0 = 0.0f;
            off1 = 0.0f;
            for (int g = 0; g < ng; ++g) {
                m0[g] = tb[g] * t0[g];
                m1[g] = tb[g] * t1[g];
                mx0   = fabsf(m0[g]) > mx0 ? fabsf(m0[g]) : mx0;
                mx1   = fabsf(m1[g]) > mx1 ? fabsf(m1[g]) : mx1;
                off0 += ub[g] * n0[g];
                off1 += ub[g] * n1[g];
            }
            for (int g = 0; g < ng; ++g) {
                s0[g] = (int8_t) lrintf(m0[g] * (127.0f / mx0));
                s1[g] = (int8_t) lrintf(m1[g] * (127.0f / mx1));
            }
            q0 = mx0 / 127.0f;
            q1 = mx1 / 127.0f;
        }
        int16_t lo[32], hi[32];
        memset(lo, 0, sizeof(lo));
        memset(hi, 0, sizeof(hi));
        for (int g = 0; g < ng; ++g) {
            const uint8_t * q = recs[g] + qo;
            const int16_t   a = s0[g], b = s1[g];
            for (int l = 0; l < 32; ++l) {
                lo[l] = (int16_t) (lo[l] + (int16_t) (q[l] & 0xF) * a);
                hi[l] = (int16_t) (hi[l] + (int16_t) (q[l] >> 4) * b);
            }
        }
        for (int l = 0; l < 32; ++l) {
            y[c + l] += lo[l] * q0 - off0;
            y[c + 32 + l] += hi[l] * q1 - off1;
        }
#endif
    }
}
