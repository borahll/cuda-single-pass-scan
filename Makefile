COMPILER?=nvcc
OPT_FLAGS?=-O3

PBB_MAIN=test_pbb

.PHONY: clean all run

ELEMS_PER_THREAD=24
N=100003565
B=256

default: $(PBB_MAIN)

all: $(PBB_MAIN)

$(PBB_MAIN): pbb_main.cu host_skel.cuh pbb_kernels.cuh constants.cuh
	$(COMPILER) $(OPT_FLAGS) -DELEMS_PER_THREAD=$(ELEMS_PER_THREAD) -o $(PBB_MAIN) $<

run: $(PBB_MAIN)
	./$(PBB_MAIN) $(N) $(B)

clean:
	rm -f $(PBB_MAIN)
