/* pmedian_core.c — C mirror of pbp_pmedian/core.py's hot kernels.
 *
 * House rule (inherited from tmc-pbp): core.py is canonical; any semantic
 * change lands there first and is mirrored here. The Python test suite
 * cross-checks every function against the canonical implementation.
 *
 * Convention: C is row-major m x n (facilities x clients); S holds p row
 * indices. Tie-breaking replicates the canonical NumPy semantics: first
 * minimum in ascending scan order.
 */

#include <stdlib.h>

static int cmp_int(const void *a, const void *b)
{
    return (*(const int *)a) - (*(const int *)b);
}

/* f(S) = sum over columns of the cheapest open row. */
double pm_cost(const double *C, int m, int n, const int *S, int p)
{
    (void)m;
    double total = 0.0;
    for (int j = 0; j < n; j++) {
        double best = C[(long)S[0] * n + j];
        for (int k = 1; k < p; k++) {
            double v = C[(long)S[k] * n + j];
            if (v < best)
                best = v;
        }
        total += best;
    }
    return total;
}

/* Per-column top-2 structure over the rows of S (S in given order):
 * serve1 = min, amin = position in S attaining it (first), serve2 = second
 * minimum (>= serve1). serve2 is +inf when p == 1 (caller passes huge). */
static void top2(const double *C, int n, const int *S, int p,
                 double *serve1, double *serve2, int *amin)
{
    for (int j = 0; j < n; j++) {
        double b1 = 1e308, b2 = 1e308;
        int k1 = 0;
        for (int k = 0; k < p; k++) {
            double v = C[(long)S[k] * n + j];
            if (v < b1) {
                b2 = b1;
                b1 = v;
                k1 = k;
            } else if (v < b2) {
                b2 = v;
            }
        }
        serve1[j] = b1;
        serve2[j] = b2;
        amin[j] = k1;
    }
}

/* Best 1-swap upper bound on min_{S' != S} f_{C2}(S') - f_star over the
 * 1-swap neighbourhood, clamped at 0. Returns -1.0 on allocation failure. */
double pm_swap_t1(const double *C2, int m, int n, const int *S, int p,
                  double f_star)
{
    double *serve1 = malloc((size_t)n * sizeof(double));
    double *serve2 = malloc((size_t)n * sizeof(double));
    int *amin = malloc((size_t)n * sizeof(int));
    char *in_S = calloc((size_t)m, 1);
    if (!serve1 || !serve2 || !amin || !in_S) {
        free(serve1); free(serve2); free(amin); free(in_S);
        return -1.0;
    }
    for (int k = 0; k < p; k++)
        in_S[S[k]] = 1;
    top2(C2, n, S, p, serve1, serve2, amin);
    double best = 1e308;
    for (int k = 0; k < p; k++) {
        for (int i = 0; i < m; i++) {
            if (in_S[i])
                continue;
            const double *row = C2 + (long)i * n;
            double f = 0.0;
            for (int j = 0; j < n; j++) {
                double cw = (amin[j] == k) ? serve2[j] : serve1[j];
                f += (row[j] < cw) ? row[j] : cw;
            }
            if (f < best)
                best = f;
        }
    }
    free(serve1); free(serve2); free(amin); free(in_S);
    double d = best - f_star;
    return d > 0.0 ? d : 0.0;
}

/* Best-improvement 1-swap descent. S (length p) is modified in place and
 * sorted ascending on exit. Returns the final cost; iters_out gets the
 * number of applied swaps. Returns -1.0 on allocation failure. */
double pm_swap_descent(const double *C, int m, int n, int p, int *S,
                       int max_iters, int *iters_out)
{
    double *serve1 = malloc((size_t)n * sizeof(double));
    double *serve2 = malloc((size_t)n * sizeof(double));
    int *amin = malloc((size_t)n * sizeof(int));
    char *in_S = calloc((size_t)m, 1);
    if (!serve1 || !serve2 || !amin || !in_S) {
        free(serve1); free(serve2); free(amin); free(in_S);
        return -1.0;
    }
    qsort(S, (size_t)p, sizeof(int), cmp_int);
    double f = pm_cost(C, m, n, S, p);
    int it = 0;
    for (; it < max_iters; it++) {
        for (int i = 0; i < m; i++)
            in_S[i] = 0;
        for (int k = 0; k < p; k++)
            in_S[S[k]] = 1;
        top2(C, n, S, p, serve1, serve2, amin);
        double best_delta = 0.0;
        int best_k = -1, best_i = -1;
        for (int k = 0; k < p; k++) {
            for (int i = 0; i < m; i++) {
                if (in_S[i])
                    continue;
                const double *row = C + (long)i * n;
                double fn = 0.0;
                for (int j = 0; j < n; j++) {
                    double cw = (amin[j] == k) ? serve2[j] : serve1[j];
                    fn += (row[j] < cw) ? row[j] : cw;
                }
                if (fn - f < best_delta - 1e-9) {
                    best_delta = fn - f;
                    best_k = k;
                    best_i = i;
                }
            }
        }
        if (best_k < 0)
            break;
        S[best_k] = best_i;
        qsort(S, (size_t)p, sizeof(int), cmp_int);
        f += best_delta;
    }
    free(serve1); free(serve2); free(amin); free(in_S);
    if (iters_out)
        *iters_out = it;
    return f;
}
