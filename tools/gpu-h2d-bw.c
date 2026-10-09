/* gpu-h2d-bw.c: host-to-device DMA bandwidth from pinned host memory,
 * each GPU alone and then all of them at once (the aggregate the host's
 * memory and the PCIe root complex sustain), on the CUDA driver API.
 *
 *   gcc -O2 -I/opt/cuda/include -o gpu-h2d-bw gpu-h2d-bw.c -lcuda
 *   ./gpu-h2d-bw [MiB per copy, 128] [repeats, 20]
 *
 * Written 2026-10-09 to size an expert cache's miss path (a decode
 * step's experts streamed from host memory into VRAM) against the CPU's
 * own read rate (about 60 GB/s measured through llama.cpp's kernels). */
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define CK(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { const char *s_ = "?"; cuGetErrorString(r_, &s_); fprintf(stderr, "%s: %s\n", #x, s_); exit(1); } } while (0)

static double now(void)
{
	struct timespec t;
	clock_gettime(CLOCK_MONOTONIC, &t);
	return t.tv_sec + t.tv_nsec * 1e-9;
}

int main(int argc, char **argv)
{
	size_t mib = argc > 1 ? (size_t)atol(argv[1]) : 128;
	int reps = argc > 2 ? atoi(argv[2]) : 20;
	size_t bytes = mib << 20;
	CK(cuInit(0));
	int n;
	CK(cuDeviceGetCount(&n));
	if (n > 8) n = 8;
	CUcontext ctx[8];
	CUdeviceptr dev[8];
	void *host[8];
	CUstream st[8];
	for (int i = 0; i < n; i++) {
		CUdevice d;
		CK(cuDeviceGet(&d, i));
		CK(cuDevicePrimaryCtxRetain(&ctx[i], d));
		CK(cuCtxSetCurrent(ctx[i]));
		CK(cuMemAlloc(&dev[i], bytes));
		CK(cuMemHostAlloc(&host[i], bytes, CU_MEMHOSTALLOC_PORTABLE));
		memset(host[i], i + 1, bytes);
		CK(cuStreamCreate(&st[i], CU_STREAM_NON_BLOCKING));
	}
	for (int i = 0; i < n; i++) {
		CK(cuCtxSetCurrent(ctx[i]));
		CK(cuMemcpyHtoDAsync(dev[i], host[i], bytes, st[i]));
		CK(cuStreamSynchronize(st[i]));
		double t0 = now();
		for (int r = 0; r < reps; r++)
			CK(cuMemcpyHtoDAsync(dev[i], host[i], bytes, st[i]));
		CK(cuStreamSynchronize(st[i]));
		double t = now() - t0;
		printf("GPU %d alone: host to device %.1f GB/s (%zu MiB x %d)\n", i, (double)bytes * reps / t / 1e9, mib, reps);
	}
	double t0 = now();
	for (int r = 0; r < reps; r++)
		for (int i = 0; i < n; i++) {
			CK(cuCtxSetCurrent(ctx[i]));
			CK(cuMemcpyHtoDAsync(dev[i], host[i], bytes, st[i]));
		}
	for (int i = 0; i < n; i++) {
		CK(cuCtxSetCurrent(ctx[i]));
		CK(cuStreamSynchronize(st[i]));
	}
	double t = now() - t0;
	printf("all %d at once: host to device %.1f GB/s aggregate\n", n, (double)bytes * reps * n / t / 1e9);
	/* device to host too: a result's way back, and what a GPU-side
	 * prefill hands the host */
	t0 = now();
	for (int r = 0; r < reps; r++)
		for (int i = 0; i < n; i++) {
			CK(cuCtxSetCurrent(ctx[i]));
			CK(cuMemcpyDtoHAsync(host[i], dev[i], bytes, st[i]));
		}
	for (int i = 0; i < n; i++) {
		CK(cuCtxSetCurrent(ctx[i]));
		CK(cuStreamSynchronize(st[i]));
	}
	t = now() - t0;
	printf("all %d at once: device to host %.1f GB/s aggregate\n", n, (double)bytes * reps * n / t / 1e9);
	return 0;
}
