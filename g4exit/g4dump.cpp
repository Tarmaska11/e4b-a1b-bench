// g4dump -- capture the residual stream after chosen layers (tensor "l_out-<il>") for a list of prompts.
//
// env: G4_LAYERS="23,26,41"  layers to capture
//      G4_OUT=path           binary output
//      G4_LAST=1             keep only the last token's vector per record (else every token)
// args: the usual llama.cpp common args; records come from -f <file>, separated by "\n<<<G4SEP>>>\n",
//       terminated by a record "<<<END>>>". Always pass --no-escape.
//
// output (little endian): int32 magic 'G4D1', n_embd, last_only, n_layers, layers[n_layers]
//   then per record: int32 rec_idx, n_tok (negative = skipped, too long), n_vec,
//                    int32 tokens[n_tok], then for each layer: float32[n_vec * n_embd]

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <clocale>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

struct g4cap {
    std::vector<int>                  layers;
    bool                              last_only = false;
    int                               n_embd    = 0;
    std::map<int, std::vector<float>> data;
    std::map<int, int>                ncols;
};

static bool g4_cb(struct ggml_tensor * t, bool ask, void * ud) {
    g4cap * c = (g4cap *) ud;
    if (strncmp(t->name, "l_out-", 6) != 0) {
        return !ask;
    }
    const int  il   = atoi(t->name + 6);
    const bool want = std::find(c->layers.begin(), c->layers.end(), il) != c->layers.end();
    if (ask) {
        return want;
    }
    if (!want) {
        return true;
    }
    if (t->type != GGML_TYPE_F32) {
        fprintf(stderr, "g4dump: %s is not F32\n", t->name);
        exit(2);
    }
    const size_t       nb = ggml_nbytes(t);
    std::vector<float> buf(nb / sizeof(float));
    ggml_backend_tensor_get(t, buf.data(), 0, nb);
    const int n_embd = (int) t->ne[0];
    const int n_col  = (int) t->ne[1];
    c->n_embd        = n_embd;
    auto & v         = c->data[il];
    if (c->last_only) {
        v.assign(buf.begin() + (size_t) (n_col - 1) * n_embd, buf.begin() + (size_t) n_col * n_embd);
        c->ncols[il] = 1;
    } else {
        v.insert(v.end(), buf.begin(), buf.end());
        c->ncols[il] += n_col;
    }
    return true;
}

static void w32(FILE * f, int32_t v) {
    fwrite(&v, sizeof(v), 1, f);
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    const char * e_layers = getenv("G4_LAYERS");
    const char * e_out    = getenv("G4_OUT");
    const char * e_last   = getenv("G4_LAST");
    if (!e_layers || !e_out) {
        fprintf(stderr, "g4dump: set G4_LAYERS and G4_OUT\n");
        return 1;
    }

    g4cap cap;
    {
        std::string s = e_layers;
        size_t      p = 0;
        while (p < s.size()) {
            size_t q = s.find(',', p);
            if (q == std::string::npos) {
                q = s.size();
            }
            cap.layers.push_back(atoi(s.substr(p, q - p).c_str()));
            p = q + 1;
        }
    }
    cap.last_only = e_last && atoi(e_last) != 0;

    llama_backend_init();
    llama_numa_init(params.numa);

    params.cb_eval           = g4_cb;
    params.cb_eval_user_data = &cap;
    params.warmup            = false;

    auto   llama_init = common_init_from_params(params);
    auto * model      = llama_init->model();
    auto * ctx        = llama_init->context();
    if (model == nullptr || ctx == nullptr) {
        fprintf(stderr, "g4dump: failed to init\n");
        return 1;
    }

    const std::string        sep = "\n<<<G4SEP>>>\n";
    std::vector<std::string> recs;
    {
        size_t start = 0;
        while (true) {
            size_t p = params.prompt.find(sep, start);
            if (p == std::string::npos) {
                recs.push_back(params.prompt.substr(start));
                break;
            }
            recs.push_back(params.prompt.substr(start, p - start));
            start = p + sep.size();
        }
    }

    FILE * fo = fopen(e_out, "wb");
    if (!fo) {
        fprintf(stderr, "g4dump: cannot open %s\n", e_out);
        return 1;
    }
    const int n_embd_model = llama_model_n_embd(model);
    w32(fo, 0x31443447);
    w32(fo, n_embd_model);
    w32(fo, cap.last_only ? 1 : 0);
    w32(fo, (int32_t) cap.layers.size());
    for (int l : cap.layers) {
        w32(fo, l);
    }

    const int   n_batch = params.n_batch;
    const int   n_ctx   = llama_n_ctx(ctx);
    llama_batch batch   = llama_batch_init(n_batch, 0, 1);

    int n_done = 0;
    for (size_t r = 0; r < recs.size(); ++r) {
        const std::string & text = recs[r];
        if (text.rfind("<<<END>>>", 0) == 0) {
            break;
        }
        std::vector<llama_token> tokens = common_tokenize(ctx, text, true, true);
        const int                n_tok  = (int) tokens.size();
        if (n_tok == 0 || n_tok > n_batch || n_tok > n_ctx) {
            fprintf(stderr, "g4dump: record %zu skipped (%d tokens)\n", r, n_tok);
            w32(fo, (int32_t) r);
            w32(fo, -n_tok - 1);
            w32(fo, 0);
            continue;
        }

        llama_memory_clear(llama_get_memory(ctx), true);
        common_batch_clear(batch);
        for (int i = 0; i < n_tok; ++i) {
            common_batch_add(batch, tokens[i], i, { 0 }, cap.last_only ? (i == n_tok - 1) : true);
        }
        cap.data.clear();
        cap.ncols.clear();
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "g4dump: llama_decode failed on record %zu\n", r);
            return 3;
        }

        const int n_vec = cap.last_only ? 1 : n_tok;
        w32(fo, (int32_t) r);
        w32(fo, n_tok);
        w32(fo, n_vec);
        fwrite(tokens.data(), sizeof(int32_t), n_tok, fo);
        for (int l : cap.layers) {
            auto it = cap.data.find(l);
            if (it == cap.data.end() || cap.ncols[l] != n_vec || (int) it->second.size() != n_vec * cap.n_embd) {
                fprintf(stderr, "g4dump: layer %d not captured correctly on record %zu (cols %d, want %d)\n", l, r,
                        it == cap.data.end() ? -1 : cap.ncols[l], n_vec);
                return 4;
            }
            fwrite(it->second.data(), sizeof(float), it->second.size(), fo);
        }
        fflush(fo);
        if (++n_done % 20 == 0) {
            fprintf(stderr, "g4dump: %d records done\n", n_done);
        }
    }
    fclose(fo);
    fprintf(stderr, "g4dump: finished, %d records\n", n_done);

    llama_batch_free(batch);
    llama_backend_free();
    return 0;
}
