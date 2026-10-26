COMPILER?=nvcc
OPT_FLAGS?=-O3

PBB_MAIN=test_pbb
SPMV_MAIN=test_spmv

.PHONY: clean all run

default: compile run

compile: $(PBB_MAIN) $(SPMV_MAIN)

ELEMS_PER_THREAD=24

$(PBB_MAIN): pbb_main.cu host_skel.cuh pbb_kernels.cuh constants.cuh
	$(COMPILER) $(OPT_FLAGS) -DELEMS_PER_THREAD=$(ELEMS_PER_THREAD) -o $(PBB_MAIN) $<

$(SPMV_MAIN): spmv_mul_main.cu spmv_mul_kernels.cuh
	$(COMPILER) $(OPT_FLAGS) -o $(SPMV_MAIN) $<

all: $(PBB_MAIN) $(SPMV_MAIN)

run: $(PBB_MAIN) $(SPMV_MAIN)
	./$(PBB_MAIN) 100003565 256
	./$(SPMV_MAIN) 11033 2076 256

run_pbb: $(PBB_MAIN)
	./$(PBB_MAIN) 100003565 256

run_spmv: $(SPMV_MAIN)
	./$(SPMV_MAIN) 11033 2076 256

clean:
	rm -f $(PBB_MAIN) $(SPMV_MAIN)
