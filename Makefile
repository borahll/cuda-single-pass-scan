# nvcc from PATH, else the default CUDA install (the futhark nodes do not put it on PATH)
COMPILER?=$(shell command -v nvcc 2>/dev/null || echo /usr/local/cuda/bin/nvcc)
OPT_FLAGS?=-O3 -std=c++17
# GPU architecture: native = the GPU of the build machine; A100: make ARCH=sm_80
ARCH?=native

PBB_MAIN=test_pbb
BENCH=bench_lookback

.PHONY: clean all run bench bench_stats test_lookback sanitize FORCE

ELEMS_PER_THREAD=24
N=100003565
B=256

# `make`: build, validate every variant against the CPU reference, then time it (CSV on stdout)
default: bench

all: $(PBB_MAIN) $(BENCH) $(BENCH)_debug $(BENCH)_nop

$(PBB_MAIN): pbb_main.cu host_skel.cuh pbb_kernels.cuh constants.cuh FORCE
	$(COMPILER) $(OPT_FLAGS) -arch=$(ARCH) -DELEMS_PER_THREAD=$(ELEMS_PER_THREAD) -o $(PBB_MAIN) $<

run: $(PBB_MAIN)
	./$(PBB_MAIN) $(N) $(B)

test: $(PBB_MAIN)
	./$(PBB_MAIN) 100 256
	./$(PBB_MAIN) 6144 256
	./$(PBB_MAIN) 6145 256
	./$(PBB_MAIN) 100000 256
	./$(PBB_MAIN) 100003565 256

# Task 4: look-back variants
SIZES=98304 262144 1048576 4194304 16777216 67108864 268435456 536870912
DEPS=bench_lookback.cu host_skel.cuh pbb_kernels.cuh constants.cuh

$(BENCH): $(DEPS) FORCE
	$(COMPILER) $(OPT_FLAGS) -arch=$(ARCH) -DELEMS_PER_THREAD=$(ELEMS_PER_THREAD) -o $@ $<

# 4.3c statistics + bounded spin (timings from this binary are not representative)
$(BENCH)_debug: $(DEPS) FORCE
	$(COMPILER) $(OPT_FLAGS) -arch=$(ARCH) -DELEMS_PER_THREAD=$(ELEMS_PER_THREAD) -DLOOKBACK_DEBUG -o $@ $<

# only block 0 publishes P: every look-back walks to block 0 (window slides, k < 32, 4.3b forward pass)
$(BENCH)_nop: $(DEPS) FORCE
	$(COMPILER) $(OPT_FLAGS) -arch=$(ARCH) -DELEMS_PER_THREAD=$(ELEMS_PER_THREAD) -DLOOKBACK_DEBUG -DLOOKBACK_NO_P -o $@ $<

bench: $(BENCH)
	for s in $(SIZES); do ./$(BENCH) $$s $(B) 100 3 || exit 1; done

bench_stats: $(BENCH)_debug
	for s in $(SIZES); do ./$(BENCH)_debug $$s $(B) 3 3 || exit 1; done

test_lookback: $(BENCH)_debug $(BENCH)_nop
	for s in 100 98304 98305 1000000 25165824; do ./$(BENCH)_nop $$s $(B) 1 3 || exit 1; done
	for s in 100 98305 16777216 268435456; do ./$(BENCH)_debug $$s $(B) 1 20 || exit 1; done

# shared-memory races, barrier misuse, out-of-bounds tail accesses (slow: small sizes only)
sanitize: $(BENCH)
	for t in racecheck synccheck memcheck; do for s in 98305 1000003; do \
	  compute-sanitizer --tool $$t --error-exitcode 1 ./$(BENCH) $$s $(B) 1 1 > /dev/null || exit 1; done; done

# Rebuild so changes to command-line compiler settings always take effect.
FORCE:

clean:
	rm -f $(PBB_MAIN) $(BENCH) $(BENCH)_debug $(BENCH)_nop
