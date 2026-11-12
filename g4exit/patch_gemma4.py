#!/usr/bin/env python3
"""Add the G4_SKIP knob to llama.cpp's src/models/gemma4.cpp (b11235).

G4_SKIP="a:b,c:d" skips layers a <= il < b (and c <= il < d). Only KV-less layers (the top stack that reuses
the K/V of earlier layers) may be skipped; anything else aborts, because it would leave a hole in the cache.
"""
import pathlib
import sys

p = pathlib.Path(sys.argv[1]) / "src" / "models" / "gemma4.cpp"
s = p.read_text(encoding="utf-8")

inc = '#include "models.h"\n'
assert s.count(inc) == 1, "include anchor"
s = s.replace(inc, inc + "\n#include <cstdlib>\n#include <string>\n#include <utility>\n#include <vector>\n")

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
p.write_text(s, encoding="utf-8")
print("patched", p)
