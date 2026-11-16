
// ====================================================================================================================
// G4PROF=1: wall time per KIND of graph node (operation, storage type and shape of its first input), measured on
// thread 0 from the start of the node to the end of the barrier that follows it. g4prof_mark() is called once per
// decoded token by the tick op in gemma4.cpp; the first mark forgets everything before it (prompt pass, first
// token), so the table is "per decoded token". Printed at exit, largest first.
// ====================================================================================================================
#define G4PROF_MAX 384
struct g4prof_row { char key[112]; double us; long long n; };
static struct g4prof_row g4prof_tab[G4PROF_MAX];
static int       g4prof_n      = 0;
static int       g4prof_state  = -1; // -1 = not read, 0 = off, 1 = on
static long long g4prof_tokens = 0;

static int g4prof_enabled(void) {
    if (g4prof_state < 0) {
        const char * e = getenv("G4PROF");
        g4prof_state   = (e && e[0] && e[0] != '0') ? 1 : 0;
    }
    return g4prof_state;
}

static int g4prof_cmp(const void * a, const void * b) {
    const double x = ((const struct g4prof_row *) a)->us;
    const double y = ((const struct g4prof_row *) b)->us;
    return x < y ? 1 : (x > y ? -1 : 0);
}

static void g4prof_report(void) {
    if (g4prof_tokens < 3 || g4prof_n == 0) {
        return;
    }
    const double tk = (double) (g4prof_tokens - 1);
    double       total = 0.0;
    for (int i = 0; i < g4prof_n; ++i) {
        total += g4prof_tab[i].us;
    }
    qsort(g4prof_tab, (size_t) g4prof_n, sizeof(g4prof_tab[0]), g4prof_cmp);
    fprintf(stderr, "G4PROF per decoded token (%lld tokens): %.2f ms in graph nodes\n", g4prof_tokens - 1, total / tk / 1000.0);
    for (int i = 0; i < g4prof_n; ++i) {
        if (g4prof_tab[i].us / tk < 50.0 && i > 40) {
            break;
        }
        fprintf(stderr, "G4PROF %8.3f ms %6.1f calls  %s\n", g4prof_tab[i].us / tk / 1000.0, (double) g4prof_tab[i].n / tk, g4prof_tab[i].key);
    }
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((visibility("default")))
#endif
void g4prof_mark(void) {
    if (g4prof_enabled() != 1) {
        return;
    }
    if (g4prof_tokens == 0) {
        g4prof_n = 0;
        atexit(g4prof_report);
    }
    g4prof_tokens += 1;
}

static void g4prof_add(const struct ggml_tensor * node, int64_t us) {
    char                       key[112];
    const struct ggml_tensor * a = node->src[0];
    if (a) {
        snprintf(key, sizeof(key), "%-12s in0 %-5s [%lld x %lld] %.1f MiB -> out [%lld x %lld]", ggml_op_name(node->op), ggml_type_name(a->type),
                 (long long) a->ne[0], (long long) a->ne[1], ggml_nbytes(a) / 1048576.0, (long long) node->ne[0], (long long) node->ne[1]);
    } else {
        snprintf(key, sizeof(key), "%-12s -> out [%lld x %lld]", ggml_op_name(node->op), (long long) node->ne[0], (long long) node->ne[1]);
    }
    for (int i = 0; i < g4prof_n; ++i) {
        if (strcmp(g4prof_tab[i].key, key) == 0) {
            g4prof_tab[i].us += (double) us;
            g4prof_tab[i].n += 1;
            return;
        }
    }
    if (g4prof_n < G4PROF_MAX) {
        snprintf(g4prof_tab[g4prof_n].key, sizeof(g4prof_tab[g4prof_n].key), "%s", key);
        g4prof_tab[g4prof_n].us = (double) us;
        g4prof_tab[g4prof_n].n  = 1;
        g4prof_n += 1;
    }
}

