// [CB 0084] Shared embedding: find the INT2 LM head inside tf_lite_prefill_decode and decode its
// rows exactly as Gemma 4's tf_lite_embedder graph does. Self-contained (no schema header, only
// <cstdint>/<cstring>/<cmath>): inserted verbatim by patches/0084_sharedemb.py AND compiled by the
// device test research/e4b-a1b/phone/sharedemb_test.cc, so the code proven on the phone is the
// code that ships.
//
// TFLite schema field ids used (tflite schema.fbs; fixed by the format):
//   Model{2 subgraphs, 4 buffers}  SubGraph{0 tensors}  Tensor{0 shape, 1 type, 2 buffer,
//   4 quantization}  QuantizationParameters{2 scale, 3 zero_point, 6 quantized_dimension}
//   Buffer{0 data, 1 offset, 2 size}  TensorType INT2 = 19.
// Buffer offset/size are relative to the start of the model (external weights), as in
// tf_lite_prefill_decode; the embedder section stores its table inline (data vector).
namespace cb_sharedemb {

struct Fb {
  const uint8_t* p;
  uint64_t n;
  bool Ok(uint64_t off, uint64_t len) const { return off <= n && len <= n - off; }
  uint16_t U16(uint64_t off) const {
    uint16_t v = 0;
    if (Ok(off, 2)) std::memcpy(&v, p + off, 2);
    return v;
  }
  uint32_t U32(uint64_t off) const {
    uint32_t v = 0;
    if (Ok(off, 4)) std::memcpy(&v, p + off, 4);
    return v;
  }
  uint64_t U64(uint64_t off) const {
    uint64_t v = 0;
    if (Ok(off, 8)) std::memcpy(&v, p + off, 8);
    return v;
  }
  // Absolute position of field f of the table at t; 0 when the field is absent.
  uint64_t Field(uint64_t t, int f) const {
    if (t == 0 || !Ok(t, 4)) return 0;
    const int64_t vt = static_cast<int64_t>(t) - static_cast<int32_t>(U32(t));
    if (vt < 0 || !Ok(static_cast<uint64_t>(vt), 4)) return 0;
    const uint64_t fo = 4 + 2 * static_cast<uint64_t>(f);
    if (fo + 2 > U16(static_cast<uint64_t>(vt))) return 0;
    const uint16_t o = U16(static_cast<uint64_t>(vt) + fo);
    return o != 0 ? t + o : 0;
  }
  // Follows the uoffset stored at pos; 0 when out of range.
  uint64_t Deref(uint64_t pos) const {
    if (pos == 0 || !Ok(pos, 4)) return 0;
    const uint64_t r = pos + U32(pos);
    return Ok(r, 4) ? r : 0;
  }
  // The vector held by field f of table t: element count and position of element 0.
  bool Vec(uint64_t t, int f, uint64_t* len, uint64_t* data) const {
    const uint64_t v = Deref(Field(t, f));
    if (v == 0) return false;
    *len = U32(v);
    *data = v + 4;
    return true;
  }
};

struct Head {
  const uint8_t* codes = nullptr;   // vocab rows of dim/4 bytes, four 2-bit codes per byte
  const uint8_t* scales = nullptr;  // vocab float32 row scales (not necessarily aligned)
  int vocab = 0;
  int dim = 0;
  int refs = 0;                     // graph tensors that point at this table
};

// The largest rank-2 INT2 [vocab, dim] tensor with one scale per row, quantized dimension 0 and
// all-zero zero points. False (with *why) when there is none, or when two different tables tie.
inline bool FindInt2Head(const uint8_t* base, uint64_t size, Head* out, const char** why) {
  const Fb fb{base, size};
  const uint64_t model = fb.Deref(0);
  uint64_t nsg = 0, sg0 = 0, nbuf = 0, buf0 = 0;
  if (model == 0 || !fb.Vec(model, 2, &nsg, &sg0) || !fb.Vec(model, 4, &nbuf, &buf0)) {
    *why = "not a TFLite flatbuffer";
    return false;
  }
  Head best;
  for (uint64_t s = 0; s < nsg; ++s) {
    const uint64_t sg = fb.Deref(sg0 + 4 * s);
    uint64_t nt = 0, t0 = 0;
    if (sg == 0 || !fb.Vec(sg, 0, &nt, &t0)) continue;
    for (uint64_t i = 0; i < nt; ++i) {
      const uint64_t t = fb.Deref(t0 + 4 * i);
      const uint64_t tp = fb.Field(t, 1);
      if (tp == 0 || !fb.Ok(tp, 1) || base[tp] != 19) continue;  // TensorType INT2
      uint64_t nd = 0, d0 = 0;
      if (!fb.Vec(t, 0, &nd, &d0) || nd != 2 || !fb.Ok(d0, 8)) continue;
      const int32_t vocab = static_cast<int32_t>(fb.U32(d0));
      const int32_t dim = static_cast<int32_t>(fb.U32(d0 + 4));
      if (vocab < 1024 || dim <= 0 || dim % 4 != 0) continue;
      const uint64_t q = fb.Deref(fb.Field(t, 4));
      uint64_t ns = 0, s0 = 0;
      if (q == 0 || !fb.Vec(q, 2, &ns, &s0) || ns != static_cast<uint64_t>(vocab) ||
          !fb.Ok(s0, 4 * ns)) {
        continue;
      }
      const uint64_t qd = fb.Field(q, 6);
      if (qd != 0 && fb.U32(qd) != 0) continue;
      uint64_t nz = 0, z0 = 0;
      if (fb.Vec(q, 3, &nz, &z0)) {
        if (!fb.Ok(z0, 8 * nz)) continue;
        bool zero = true;
        for (uint64_t z = 0; z < nz && zero; ++z) zero = fb.U64(z0 + 8 * z) == 0;
        if (!zero) continue;
      }
      const uint64_t bp = fb.Field(t, 2);
      const uint64_t b = bp != 0 ? fb.U32(bp) : 0;
      if (b >= nbuf) continue;
      const uint64_t bu = fb.Deref(buf0 + 4 * b);
      if (bu == 0) continue;
      const uint64_t op = fb.Field(bu, 1), sp = fb.Field(bu, 2);
      const uint64_t off = op != 0 ? fb.U64(op) : 0, len = sp != 0 ? fb.U64(sp) : 0;
      uint64_t at = 0, bytes = 0;
      if (len > 0 && off > 1) {
        at = off;
        bytes = len;
      } else if (!fb.Vec(bu, 0, &bytes, &at)) {
        continue;
      }
      const uint64_t want = static_cast<uint64_t>(vocab) * static_cast<uint64_t>(dim / 4);
      if (bytes != want || !fb.Ok(at, bytes)) continue;
      if (vocab > best.vocab) {
        best.codes = base + at;
        best.scales = base + s0;
        best.vocab = vocab;
        best.dim = dim;
        best.refs = 1;
      } else if (vocab == best.vocab) {
        if (base + at != best.codes || dim != best.dim) {
          *why = "two different INT2 tables of the same vocabulary";
          return false;
        }
        ++best.refs;
      }
    }
  }
  if (best.codes == nullptr) {
    *why = "no INT2 [vocab, dim] tensor with per-row scales";
    return false;
  }
  *out = best;
  return true;
}

// Gemma's embedding normaliser sqrt(dim), rounded to float32 -- for dim 2560 this is 0x424a62c2,
// bit-identical to the MUL constant in the E4B embedder graph.
inline float Normaliser(int dim) {
  return static_cast<float>(std::sqrt(static_cast<double>(dim)));
}

// One row with the embedder graph's arithmetic: (float(code) * row_scale) * mul. Codes are 2-bit
// two's complement, element 4j+k in bits 2k..2k+1 of byte j. Ids outside [0, vocab) give row 0
// (the graph's GREATER_EQUAL/LESS/SELECT_V2 clamp). out receives dim float32 values; it need
// not be aligned.
inline void DecodeRow(const uint8_t* codes, const float* scales, int vocab, int dim, float mul,
                      int token, uint8_t* out) {
  const uint64_t row = (token >= 0 && token < vocab) ? static_cast<uint64_t>(token) : 0;
  const uint8_t* src = codes + row * static_cast<uint64_t>(dim / 4);
  const float s = scales[row];
  float quad[4];
  for (int j = 0; j < dim / 4; ++j) {
    const unsigned b = src[j];
    for (int k = 0; k < 4; ++k) {
      int c = static_cast<int>((b >> (2 * k)) & 3u);
      if (c >= 2) c -= 4;
      quad[k] = (static_cast<float>(c) * s) * mul;
    }
    std::memcpy(out + 16 * static_cast<uint64_t>(j), quad, sizeof(quad));
  }
}

}  // namespace cb_sharedemb
