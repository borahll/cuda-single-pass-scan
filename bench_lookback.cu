/**
 * Task 4 driver: validates and benchmarks the look-back variants of
 *   the single-pass scan (`scanBlockLevel` / `scanBlockKernel`).
 *
 * Usage: ./bench_lookback <bytes> <block-size> [timed-runs=50] [validated-runs=3] [type]
 *   type in {u8,u16,u32,u64,f32}; all if omitted. N = bytes / sizeof(T).
 *
 * Per type: cudaMemcpy D2D and CUB InclusiveScan as references, then each
 *   variant: `validated-runs` launches compared with a sequential CPU scan
 *   (exact for integers, relative tolerance for f32; output poisoned before
 *   each launch, I8), then
 *   `timed-runs` launches; the median (cudaEvent) is reported.
 * Every launch resets block counter + flags first (I6, one memset inside
 *   `scanBlockLevel`); it is inside the timed region, just as CUB's timing
 *   includes its tile-state init kernel.
 * Variant names: look-back (serial, warp_read, warp_par, warp_par4, skip),
 *   `+packed` = packed descriptors (4.3a), `+local` = 4.4 local phase
 *   (otherwise the Task 3 local phase). Last CSV column: resident blocks/SM.
 * Unsigned types: wrap-around is well defined, so equality is exact.
 * Output: one CSV line per measurement. -DLOOKBACK_DEBUG builds also print
 *   look-back statistics (4.3c), averaged over the validated launches.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <type_traits>
#include <cub/cub.cuh>
#include <cuda/std/functional>
#include "host_skel.cuh"

static int RUNS   = 50;
static int CHECKS = 3;

// median over RUNS timed calls of `launch`, in ms; the caller has already warmed up
template<class F>
float medianMs(F launch) {
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    ::std::vector<float> ms(RUNS);
    for (int r = 0; r < RUNS; r++) {
        cudaEventRecord(t0);
        launch();
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);
        cudaEventElapsedTime(&ms[r], t0, t1);
    }
    CUDASSERT(cudaPeekAtLastError());
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    ::std::sort(ms.begin(), ms.end());
    return ms[RUNS / 2];
}

// float: GPU and CPU sum in different orders, so compare with a relative tolerance (I8).
//   Inputs are positive and < 1/64, so a block sums to ~50 and losing the fractional
//   part of an offset (e.g. a numeric float->int conversion) is a ~1% error, far above it.
static const double FLOAT_REL_TOL = 1e-4;

// poison d_out, launch once, compare with h_ref (exact for integers); exits on mismatch
template<class T, class F>
void validate(F launch, size_t N, T* d_out, const T* h_ref, T* h_out, const char* tname, const char* name) {
    CUDASSERT(cudaMemset(d_out, 0xA5, N * sizeof(T)));
    launch();
    CUDASSERT(cudaDeviceSynchronize());
    CUDASSERT(cudaMemcpy(h_out, d_out, N * sizeof(T), cudaMemcpyDeviceToHost));
    if constexpr (::std::is_floating_point_v<T>) {
        double max_err = 0.0;
        for (size_t i = 0; i < N; i++) {
            double err = fabs((double)h_out[i] - (double)h_ref[i]) / (double)h_ref[i];
            if (!(err <= FLOAT_REL_TOL)) {  // also catches NaN from the poison pattern
                printf("INVALID %s %s at index %zu: GPU=%.8g CPU=%.8g rel.err=%.2e\n",
                       tname, name, i, (double)h_out[i], (double)h_ref[i], err);
                exit(1);
            }
            max_err = fmax(max_err, err);
        }
        printf("# %s %s max rel.err %.2e\n", tname, name, max_err);
        return;
    }
    if (memcmp(h_out, h_ref, N * sizeof(T)) == 0) return;
    size_t i = 0;
    while (h_out[i] == h_ref[i]) i++;
    printf("INVALID %s %s at index %zu: GPU=%llu CPU=%llu\n", tname, name, i,
           (unsigned long long)h_out[i], (unsigned long long)h_ref[i]);
    exit(1);
}

template<class T>
struct Bench {
    const char* tname;
    size_t N;
    uint32_t B;
    T *d_in, *d_out, *h_ref, *h_out;
    void* ws;                                    // counter + flags, see scanWorkspaceBytes
    T *Ps, *As;
    float memcpyMs, cubMs;

    // per_sm: resident blocks per SM of this variant's kernel (0 for the references);
    //   rows are printed only after validation passed (validate() exits otherwise)
    void report(const char* name, float ms, int per_sm, const char* check = "VALID") {
        double gbs = 2.0 * N * sizeof(T) / (ms * 1e6);
        printf("%s,%zu,%zu,%s,%.2f,%.1f,%.1f,%.1f,%d,%s\n", tname, N * sizeof(T), N, name,
               ms * 1e3, gbs, 100.0 * memcpyMs / ms, 100.0 * cubMs / ms, per_sm, check);
    }

    template<int LB, bool PACKED, bool LOCAL>
    void variant(const char* name) {
        auto launch = [&] {
            scanBlockLevel<Add<T>, LB, PACKED, LOCAL>(B, N, d_out, d_in, ws, Ps, As);
        };
        // residency bounds how many blocks are in flight, hence the look-back distance
        int per_sm = 0;
        CUDASSERT(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &per_sm, scanBlockKernel<Add<T>, scanChunk<Add<T>>(), LB, PACKED, LOCAL>,
            B, scanShmemBytes<Add<T>, LOCAL>(B)));
#ifdef LOOKBACK_DEBUG
        unsigned long long st[LBS_COUNT] = {0};
        CUDASSERT(cudaMemcpyToSymbol(g_lbStats, st, sizeof(st)));
#endif
        for (int k = 0; k < CHECKS; k++)  // first launch doubles as warm-up
            validate(launch, N, d_out, h_ref, h_out, tname, name);
#ifdef LOOKBACK_DEBUG
        CUDASSERT(cudaMemcpyFromSymbol(st, g_lbStats, sizeof(st)));
        double nb = st[LBS_BLOCKS] > 0 ? (double)st[LBS_BLOCKS] : 1.0;
        printf("stats,%s,%zu,%s,%llu,%.3f,%.3f,%.2f,%.3f,%llu", tname, N * sizeof(T), name,
               st[LBS_BLOCKS] / CHECKS, st[LBS_TRIPS] / nb, st[LBS_X_READS] / nb,
               100.0 * st[LBS_X_BLOCKS] / nb, st[LBS_DIST_SUM] / nb, st[LBS_DIST_MAX]);
        for (int h = 0; h < 9; h++) printf(",%.2f", 100.0 * st[LBS_HIST + h] / nb);
        printf("\n");
#endif
        report(name, medianMs(launch), per_sm);
    }

    void run() {
        // references: device-to-device copy of the same size, CUB device scan
        auto mcpy = [&] { cudaMemcpy(d_out, d_in, N * sizeof(T), cudaMemcpyDeviceToDevice); };
        mcpy();
        memcpyMs = medianMs(mcpy);

        void* d_tmp = nullptr;
        size_t tmp_bytes = 0;
        auto plus = ::cuda::std::plus<T>();
        cub::DeviceScan::InclusiveScan(d_tmp, tmp_bytes, d_in, d_out, plus, (int)N);
        CUDASSERT(cudaMalloc(&d_tmp, tmp_bytes));
        auto cub_scan = [&] { cub::DeviceScan::InclusiveScan(d_tmp, tmp_bytes, d_in, d_out, plus, (int)N); };
        validate(cub_scan, N, d_out, h_ref, h_out, tname, "cub");
        cubMs = medianMs(cub_scan);
        cudaFree(d_tmp);
        report("memcpy", memcpyMs, 0, "-");
        report("cub", cubMs, 0);

        // Task 3 local phase
        variant<LB_SERIAL,    false, false>("serial");     // baseline
        variant<LB_WARP_READ, false, false>("warp_read");  // 4.1
        variant<LB_WARP_PAR,  false, false>("warp_par");   // 4.2
        variant<LB_SKIP,      false, false>("skip");       // 4.3b
        // 4.3a packed descriptors (64-bit elements: two-word descriptor)
        variant<LB_SERIAL,    true, false>("serial+packed");
        variant<LB_WARP_READ, true, false>("warp_read+packed");
        variant<LB_WARP_PAR,  true, false>("warp_par+packed");
        variant<LB_SKIP,      true, false>("skip+packed");
        // 4.4 local phase (+local); 4.5 window of 128 (warp_par4); 4.6 keep window (_keep)
        variant<LB_SERIAL,         false, true>("serial+local");
        variant<LB_WARP_READ,      false, true>("warp_read+local");
        variant<LB_WARP_PAR,       false, true>("warp_par+local");
        variant<LB_WARP_PAR4,      false, true>("warp_par4+local");
        variant<LB_WARP_PAR_KEEP,  false, true>("warp_par_keep+local");
        variant<LB_WARP_PAR4_KEEP, false, true>("warp_par4_keep+local");
        variant<LB_SKIP,           false, true>("skip+local");
        variant<LB_SERIAL,         true, true>("serial+packed+local");
        variant<LB_WARP_READ,      true, true>("warp_read+packed+local");
        variant<LB_WARP_PAR,       true, true>("warp_par+packed+local");
        variant<LB_WARP_PAR4,      true, true>("warp_par4+packed+local");
        variant<LB_WARP_PAR_KEEP,  true, true>("warp_par_keep+packed+local");
        variant<LB_WARP_PAR4_KEEP, true, true>("warp_par4_keep+packed+local");
        variant<LB_SKIP,           true, true>("skip+packed+local");
    }
};

template<class T>
void benchType(const char* tname, size_t bytes, uint32_t B) {
    Bench<T> b;
    b.tname = tname;
    b.B = B;
    b.N = bytes / sizeof(T);
    if (b.N == 0 || b.N > INT_MAX) {      // CUB takes the length as int
        fprintf(stderr, "%s: need 0 < N <= INT_MAX elements\n", tname);
        exit(1);
    }
    const uint32_t epb = elemsPerBlock<Add<T>>(B);
    const size_t nb = (b.N + epb - 1) / epb;

    printf("# %s: N=%zu, %u elems/block, %zu blocks, %d SMs\n",
           tname, b.N, epb, nb, prop.multiProcessorCount);

    T* h_in = (T*)malloc(b.N * sizeof(T));
    b.h_ref = (T*)malloc(b.N * sizeof(T));
    b.h_out = (T*)malloc(b.N * sizeof(T));
    if constexpr (::std::is_floating_point_v<T>) {
        double acc = 0.0;              // reference accumulated in double
        for (size_t i = 0; i < b.N; i++) {
            h_in[i] = (T)((rand() % 1023 + 1) / 65536.0);  // in (0, 1/64), see FLOAT_REL_TOL
            acc += h_in[i];
            b.h_ref[i] = (T)acc;
        }
    } else {
        T acc = 0;
        for (size_t i = 0; i < b.N; i++) {
            h_in[i] = (T)rand();
            acc = (T)(acc + h_in[i]);
            b.h_ref[i] = acc;
        }
    }

    CUDASSERT(cudaMalloc(&b.d_in, b.N * sizeof(T)));
    CUDASSERT(cudaMalloc(&b.d_out, b.N * sizeof(T)));
    CUDASSERT(cudaMalloc(&b.ws, scanWorkspaceBytes(nb)));
    CUDASSERT(cudaMalloc(&b.Ps, nb * sizeof(T)));
    CUDASSERT(cudaMalloc(&b.As, nb * sizeof(T)));
    CUDASSERT(cudaMemcpy(b.d_in, h_in, b.N * sizeof(T), cudaMemcpyHostToDevice));

    b.run();

    free(h_in); free(b.h_ref); free(b.h_out);
    cudaFree(b.d_in); cudaFree(b.d_out); cudaFree(b.ws); cudaFree(b.Ps); cudaFree(b.As);
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        printf("Usage: %s <bytes> <block-size> [timed-runs=50] [validated-runs=3] [u8|u16|u32|u64|f32]\n", argv[0]);
        exit(1);
    }
    const size_t bytes = strtoull(argv[1], NULL, 10);
    const uint32_t B = atoi(argv[2]);
    if (argc > 3) RUNS = atoi(argv[3]);
    if (argc > 4) CHECKS = atoi(argv[4]);
    if (RUNS < 1 || CHECKS < 1) {   // CHECKS >= 1: every variant is validated before timing (I8)
        printf("timed-runs and validated-runs must be >= 1\n");
        exit(1);
    }
    if (B == 0 || B % 32 != 0 || B > 1024) {
        printf("block-size must be a nonzero multiple of 32 <= 1024\n");
        exit(1);
    }
    const char* only = (argc > 5) ? argv[5] : NULL;

    initHwd();
    printf("type,bytes,N,variant,us,GBps,pct_memcpy,pct_cub,blocks_per_sm,check\n");
#ifdef LOOKBACK_DEBUG
    printf("stats,type,bytes,variant,blocks,trips_per_blk,xreads_per_blk,pct_blk_met_x,"
           "mean_dist,max_dist,d1,d2,d3-4,d5-8,d9-16,d17-32,d33-64,d65-128,d129+\n");
#endif
    if (!only || !strcmp(only, "u8"))  benchType<uint8_t >("u8",  bytes, B);
    if (!only || !strcmp(only, "u16")) benchType<uint16_t>("u16", bytes, B);
    if (!only || !strcmp(only, "f32")) benchType<float   >("f32", bytes, B);
    if (!only || !strcmp(only, "u32")) benchType<uint32_t>("u32", bytes, B);
    if (!only || !strcmp(only, "u64")) benchType<uint64_t>("u64", bytes, B);
    return 0;
}
