// C ABI around cb_sharedemb.inc (the exact file an on-device engine patch compiles), so a Python
// script can drive it with ctypes. Build: g++ -O2 -std=c++17 -shared -fPIC -o libsharedemb.so sharedemb_lib.cc
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "cb_sharedemb.inc"

extern "C" {

// 1 on success. Offsets are relative to base.
int se_find_head(const uint8_t* base, uint64_t size, int* vocab, int* dim, int* refs,
                 uint64_t* codes_off, uint64_t* scales_off, char* why, int why_len) {
  cb_sharedemb::Head head;
  const char* w = "";
  if (!cb_sharedemb::FindInt2Head(base, size, &head, &w)) {
    std::strncpy(why, w, static_cast<size_t>(why_len) - 1);
    why[why_len - 1] = 0;
    return 0;
  }
  *vocab = head.vocab;
  *dim = head.dim;
  *refs = head.refs;
  *codes_off = static_cast<uint64_t>(head.codes - base);
  *scales_off = static_cast<uint64_t>(head.scales - base);
  return 1;
}

float se_normaliser(int dim) { return cb_sharedemb::Normaliser(dim); }

// n rows of dim float32 each into out.
void se_decode_rows(const uint8_t* codes, const float* scales, int vocab, int dim, float mul,
                    const int* tokens, int n, uint8_t* out) {
  for (int i = 0; i < n; ++i) {
    cb_sharedemb::DecodeRow(codes, scales, vocab, dim, mul, tokens[i],
                            out + static_cast<uint64_t>(i) * static_cast<uint64_t>(dim) * 4);
  }
}

// FNV-1a 64 over the float32 bytes of every row 0..vocab-1 (the phone test prints the same hash).
uint64_t se_fnv_all(const uint8_t* codes, const float* scales, int vocab, int dim, float mul) {
  std::vector<uint8_t> row(static_cast<size_t>(dim) * 4);
  uint64_t h = 1469598103934665603ULL;
  for (int r = 0; r < vocab; ++r) {
    cb_sharedemb::DecodeRow(codes, scales, vocab, dim, mul, r, row.data());
    for (uint8_t b : row) {
      h ^= b;
      h *= 1099511628211ULL;
    }
  }
  return h;
}

}  // extern "C"
