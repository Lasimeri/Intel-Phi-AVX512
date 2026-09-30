/* expert-placement.c: which experts of a mixture the cards should hold
 * whole, from what a run routed to (the backend's PHI_GGML_IDS lines:
 * "ggml-phi: ids layer N tokens T used U: id id ...", one per layer's gate
 * and up request). Three uses:
 *
 *   tcc -run tools/expert-placement.c stats LOG
 *       per layer, averaged: the share of selections on the most used
 *       12.5, 25, 37.5 and 50 percent of experts, prefill lines (T > 1)
 *       and generation lines (T == 1) apart, beside what uniform routing
 *       gives the same sample by chance
 *   tcc -run tools/expert-placement.c rank LOG K > placement.txt
 *       every layer's experts in descending order of use (prefill and
 *       generation together), the first K only: the file PHI_GGML_EXPERTS
 *       names ("layer N: e e e ...")
 *   tcc -run tools/expert-placement.c cover LOG placement.txt
 *       the share of LOG's selections that fall on the placed experts:
 *       what a placement made from one text is worth on another
 *
 * See expert-placement.md. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXL 128
#define MAXE 1024

struct layer { int n; long cnt[2][MAXE]; long tot[2]; };
static struct layer L[MAXL];
static int experts;

static struct layer *lay(int n)
{
    if (n < 0 || n >= MAXL) return NULL;
    L[n].n = n + 1;   /* seen */
    return &L[n];
}

static void load(const char *path)
{
    FILE *f = fopen(path, "r");
    if (!f) { perror(path); exit(1); }
    size_t cap = 1 << 22;
    char *line = malloc(cap);
    if (!line) exit(1);
    while (fgets(line, cap, f)) {
        int n, at;
        long t, u;
        if (sscanf(line, "ggml-phi: ids layer %d tokens %ld used %ld:%n", &n, &t, &u, &at) != 3) continue;
        struct layer *l = lay(n);
        if (!l) continue;
        int cls = t > 1 ? 0 : 1;
        const char *p = line + at;
        long id;
        int adv;
        while (sscanf(p, "%ld%n", &id, &adv) == 1) {
            p += adv;
            if (id < 0 || id >= MAXE) continue;
            l->cnt[cls][id]++;
            l->tot[cls]++;
            if (id + 1 > experts) experts = id + 1;
        }
    }
    fclose(f);
    free(line);
}

static long *sortkey;
static int bycount(const void *a, const void *b)
{
    long x = sortkey[*(const int *)a], y = sortkey[*(const int *)b];
    return x < y ? 1 : x > y ? -1 : 0;
}

/* the experts of a layer in descending order of use, both classes together */
static void ranked(const struct layer *l, int *idx)
{
    static long both[MAXE];
    for (int e = 0; e < experts; e++) both[e] = l->cnt[0][e] + l->cnt[1][e];
    for (int e = 0; e < experts; e++) idx[e] = e;
    sortkey = both;
    qsort(idx, experts, sizeof *idx, bycount);
}

static unsigned long long rng = 0x9e3779b97f4a7c15ull;
static unsigned long long next_rand(void)
{
    rng ^= rng << 13;
    rng ^= rng >> 7;
    rng ^= rng << 17;
    return rng;
}

static int cmpdesc(const void *a, const void *b)
{
    long x = *(const long *)a, y = *(const long *)b;
    return x < y ? 1 : x > y ? -1 : 0;
}

/* the share of `tot` draws on the top quarter-eighths of `cnt` sorted */
static void quantiles(long *cnt, long tot, double share[4])
{
    qsort(cnt, experts, sizeof *cnt, cmpdesc);
    long acc = 0;
    for (int e = 0, q = 0; e < experts; e++) {
        acc += cnt[e];
        while (q < 4 && e + 1 == experts * (q + 1) / 8) share[q++] = (double)acc / tot;
    }
}

static void stats(void)
{
    for (int cls = 0; cls < 2; cls++) {
        double share[4] = { 0 }, uni[4] = { 0 };
        int used = 0;
        long samples = 0;
        for (int i = 0; i < MAXL; i++) {
            if (!L[i].n || !L[i].tot[cls]) continue;
            used++;
            samples += L[i].tot[cls];
            long s[MAXE];
            memcpy(s, L[i].cnt[cls], sizeof s);
            double q[4];
            quantiles(s, L[i].tot[cls], q);
            for (int k = 0; k < 4; k++) share[k] += q[k];
            long r[MAXE];
            memset(r, 0, sizeof r);
            for (long d = 0; d < L[i].tot[cls]; d++) r[next_rand() % (unsigned)experts]++;
            quantiles(r, L[i].tot[cls], q);
            for (int k = 0; k < 4; k++) uni[k] += q[k];
        }
        if (!used) continue;
        printf("%s: %d layers, %ld selections per layer on average, %d experts\n", cls == 0 ? "prefill" : "generation", used, samples / used, experts);
        printf("  most used experts   share of selections   uniform routing would give\n");
        for (int k = 0; k < 4; k++) printf("  %5.1f%%              %5.1f%%                 %5.1f%%\n", 12.5 * (k + 1), 100 * share[k] / used, 100 * uni[k] / used);
    }
}

static void rank(int k)
{
    for (int i = 0; i < MAXL; i++) {
        if (!L[i].n) continue;
        int idx[MAXE];
        ranked(&L[i], idx);
        printf("layer %d:", i);
        for (int e = 0; e < k && e < experts; e++) printf(" %d", idx[e]);
        printf("\n");
    }
}

static void cover(const char *placement)
{
    static char placed[MAXL][MAXE];
    FILE *f = fopen(placement, "r");
    if (!f) { perror(placement); exit(1); }
    char line[16384];
    while (fgets(line, sizeof line, f)) {
        int li, at;
        if (sscanf(line, "layer %d:%n", &li, &at) != 1 || li < 0 || li >= MAXL) continue;
        const char *p = line + at;
        int e, adv;
        while (sscanf(p, "%d%n", &e, &adv) == 1) {
            p += adv;
            if (e >= 0 && e < MAXE) placed[li][e] = 1;
        }
    }
    fclose(f);
    for (int cls = 0; cls < 2; cls++) {
        double sum = 0, lo = 1, hi = 0;
        int used = 0;
        for (int i = 0; i < MAXL; i++) {
            if (!L[i].n || !L[i].tot[cls]) continue;
            long c = 0;
            for (int e = 0; e < experts; e++) if (placed[i][e]) c += L[i].cnt[cls][e];
            double s = (double)c / L[i].tot[cls];
            sum += s;
            used++;
            if (s < lo) lo = s;
            if (s > hi) hi = s;
        }
        if (used) printf("%s: the placed experts take %.1f%% of the selections (layers from %.1f%% to %.1f%%), %d layers\n",
                         cls ? "generation" : "prefill", 100 * sum / used, 100 * lo, 100 * hi, used);
    }
}

int main(int argc, char **argv)
{
    if (argc >= 3 && !strcmp(argv[1], "stats")) { load(argv[2]); stats(); return 0; }
    if (argc >= 4 && !strcmp(argv[1], "rank")) { load(argv[2]); rank(atoi(argv[3])); return 0; }
    if (argc >= 4 && !strcmp(argv[1], "cover")) { load(argv[2]); cover(argv[3]); return 0; }
    fprintf(stderr, "usage: expert-placement stats LOG | rank LOG K | cover LOG PLACEMENT\n");
    return 2;
}
