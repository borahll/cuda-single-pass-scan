#ifndef PBB_KERNELS
#define PBB_KERNELS

#include <cuda_runtime.h>
#include <cuda/atomic>
#include <cuda/std/type_traits>
using namespace cuda;
/**
 * Naive memcpy kernel, for the purpose of comparing with
 * a more "realistic" bandwidth number.
 */
__global__ void naiveMemcpy(int* d_out, int* d_inp, const uint32_t N) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid < N) {
        d_out[gid] = d_inp[gid];
    }
}

/**
 * Generic Add operator that can be instantiated over
 *  numeric-basic types, such as int32_t, int64_t,
 *  float, double, etc.
 */
template<class T>
class Add {
  public:
    typedef T InpElTp;
    typedef T RedElTp;
    static const bool commutative = true;
    static __device__ __host__ inline T identInp()                    { return (T)0;    }
    static __device__ __host__ inline T mapFun(const T& el)           { return el;      }
    static __device__ __host__ inline T identity()                    { return (T)0;    }
    static __device__ __host__ inline T apply(const T t1, const T t2) { return t1 + t2; }

    static __device__ __host__ inline bool equals(const T t1, const T t2) { return (t1 == t2); }
    static __device__ __host__ inline T remVolatile(volatile T& t)    { T res = t; return res; }
};

/***************************************************/
/*** Generic Value-Flag Tuple for Segmented Scan ***/
/***************************************************/

/**
 * Generic data-type that semantically tuples template
 * `T` with a flag, which is represented as a char.
 */
template<class T>
class ValFlg {
  public:
    T    v;
    char f;
    __device__ __host__ inline ValFlg() { f = 0; }
    __device__ __host__ inline ValFlg(const char& f1, const T& v1) { v = v1; f = f1; }
    __device__ __host__ inline ValFlg(const ValFlg& vf) { v = vf.v; f = vf.f; }
    __device__ __host__ inline void operator=(const ValFlg& vf) volatile { v = vf.v; f = vf.f; }
};

/**
 * Generic segmented-scan operator, which lifts a generic
 * associative binary operator, given by generic type `OP`
 * to the corresponding segmented operator that works over
 * flag-value tuples.
 */
template<class OP>
class LiftOP {
  public:
    typedef ValFlg<typename OP::RedElTp> RedElTp;
    static __device__ __host__ inline RedElTp identity() {
        return RedElTp( (char)0, OP::identity());
    }

    static __device__ __host__ inline RedElTp
    apply(const RedElTp t1, const RedElTp t2) {
        typename OP::RedElTp v;
        char f = t1.f | t2.f;
        if (t2.f != 0) v = t2.v;
        else v = OP::apply(t1.v, t2.v);
        return RedElTp(f, v);
    }

    static __device__ __host__ inline bool
    equals(const RedElTp t1, const RedElTp t2) {
        return ( (t1.f == t2.f) && OP::equals(t1.v, t2.v) );
    }
};

/*****************************************/
/*** MSSP Data-Structure and Operators ***/
/*****************************************/

/**
 * A class representing a quad-tuple of ints, to be used
 * for MSSP problem. The reason for this is that we need
 * constructors (e.g., copy constructor) that makes it
 * work with the generic skeletons of reduce, scan, etc.
 */
class MyInt4 {
  public:
    int x, y, z, w;

    __device__ __host__ inline MyInt4() {
        x = 0; y = 0; z = 0; w = 0;
    }

    __device__ __host__ inline MyInt4(const int& a, const int& b, const int& c, const int& d) {
        x = a; y = b; z = c; w = d;
    }

    __device__ __host__ inline MyInt4(const MyInt4& i4) {
        x = i4.x; y = i4.y; z = i4.z; w = i4.w;
    }

    __device__ __host__ inline void operator=(const MyInt4& i4) volatile {
        x = i4.x; y = i4.y; z = i4.z; w = i4.w;
    }
};

/**
 * Representation of the MSSP operator, so that it can work the generically
 * defined skeletons of reduce, scan, etc.
 */
class Mssp {
  public:
    typedef int InpElTp;
    typedef MyInt4 RedElTp;
    static const bool commutative = false;
    static __device__ __host__ inline InpElTp identInp() { return 0; }
    static __device__ __host__ inline MyInt4 mapFun(const InpElTp& el) {
        InpElTp x = max(0, el);
        return MyInt4(x, x, x, el);
    }
    static __device__ __host__ inline MyInt4 identity() { return MyInt4(0,0,0,0); }
    static __device__ __host__ inline MyInt4 apply(volatile MyInt4& t1, volatile MyInt4& t2) {
        InpElTp mss = max( t1.z+t2.y, max(t1.x, t2.x) );
        InpElTp mis = max( t1.y, t1.w + t2.y);
        InpElTp mcs = max( t2.z, t2.w + t1.z);
        InpElTp t   = t1.w + t2.w;
        return MyInt4(mss, mis, mcs, t);
    }

    static __device__ __host__ inline MyInt4 remVolatile(volatile MyInt4& t) {
        MyInt4 res; res.x = t.x; res.y = t.y; res.z = t.z; res.w = t.w; return res;
    }

    static __device__ __host__ inline bool equals(MyInt4& t1, MyInt4& t2) {
        return (t1.x == t2.x && t1.y == t2.y && t1.z == t2.z && t1.w == t2.w);
    }
};

/***************************************/
/*** Scan Inclusive Helpers & Kernel ***/
/***************************************/

/**
 * A warp of threads cooperatively scan with generic-binop `OP` a
 *   number of warp elements stored in shared memory (`ptr`).
 * No synchronization is needed because the threads in a warp execute
 *   in lockstep.
 * `idx` is the local thread index within a cuda block (threadIdx.x)
 * Each thread returns the corresponding scanned element of type
 *   `typename OP::RedElTp`
 ********************************
 * Weekly Assignment 2, Task 2: *
 ********************************
 *   The provided dummy implementation works correctly, but it is
 *     very slow because the warp reduction is performed sequentially
 *     by the first thread of each warp, so it takes WARP-1=31 steps
 *     to complete, while the other 31 threads of the WARP are iddle.
 *   Your task is to write a warp-level scan implementation in which
 *     the threads in the same WARP cooperate such that the depth of
 *     this implementation is 5 steps ( WARP==32, and lg(32)=5 ).
 *     The algorithm that you need to implement is shown in the
 *     slides of Lab2.
 *   The implementation does not need any synchronization, i.e.,
 *     please do NOT use "__syncthreads();" and the like in here,
 *     especially because it will break the whole thing (because
 *     this function is conditionally called sometimes, so not
 *     all threads will reach the barrier, resulting in incorrect
 *     results.)
 */
template<class OP>
__device__ inline typename OP::RedElTp
scanIncWarp( volatile typename OP::RedElTp* ptr, const uint32_t idx ) {
    const uint32_t lane = idx & (WARP-1);

#if 1
    // new. Lanes of a warp are not lockstep (independent thread scheduling,
    //   sm_70+), so each step reads, syncs, writes, syncs; `volatile` alone
    //   orders nothing between lanes. Must be called by all 32 lanes.
    #pragma unroll
    for(uint32_t d=0; d<lgWARP; d++) {
        const uint32_t h = (1<<d);
        typename OP::RedElTp prev;
        if(lane >= h) prev = OP::remVolatile(ptr[idx-h]);
        __syncwarp();  // every lane has read step d's operand before any lane overwrites it
        if(lane >= h) {
            ptr[idx] = OP::apply(prev, ptr[idx]);
        }
        __syncwarp();  // step d's writes visible before step d+1 reads them
    }
#else
    // original
    if(lane==0) {
        #pragma unroll
        for(int i=1; i<WARP; i++) {
            ptr[idx+i] = OP::apply(ptr[idx+i-1], ptr[idx+i]);
        }
    }
#endif

    return OP::remVolatile(ptr[idx]);
}

/**
 * A CUDA-block of threads cooperatively scan with generic-binop `OP`
 *   a CUDA-block number of elements stored in shared memory (`ptr`).
 * `idx` is the local thread index within a cuda block (threadIdx.x)
 * Each thread returns the corresponding scanned element of type
 *   `typename OP::RedElTp`. Note that this is NOT published to shared memory!
 *
 *******************************************************
 * Weekly Assignment 2, Task 3:
 *******************************************************
 * Find and fix the bug (race condition) that manifests
 *  only when the CUDA block size is set to 1024.
 */
template<class OP>
__device__ inline typename OP::RedElTp
scanIncBlock(volatile typename OP::RedElTp* ptr, const uint32_t idx) {
    const uint32_t lane   = idx & (WARP-1);
    const uint32_t warpid = idx >> lgWARP;

    // 1. perform scan at warp level. `scanIncWarp` computes its result in-place
    //    and also returns the per-thread result.
    typename OP::RedElTp res = scanIncWarp<OP>(ptr,idx);
    __syncthreads();

    // 2. place the end-of-warp results in
    //   the first warp. This works because
    //   warp size = 32, and
    //   max block size = 32^2 = 1024
    typename OP::RedElTp tmp;
    if (lane == (WARP-1)) { tmp = OP::remVolatile(ptr[idx]); }
    __syncthreads();

    if (lane == (WARP-1)) { ptr[warpid] = tmp; }
    __syncthreads();

    // 3. scan again the first warp.
    if (warpid == 0) scanIncWarp<OP>(ptr, idx);
    __syncthreads();

    // 4. accumulate results from previous step.
    if (warpid > 0) {
        res = OP::apply(ptr[warpid-1], res);
    }
    return res;
}

/********************************************/
/*** Naive (silly) Kernels, just for demo ***/
/********************************************/

/**
 * Kernel to implement the naive reduce, which uses neither
 *   efficient sequentialization, nor fast, shared memory.
 */
template<class OP>
__global__ void
redNaiveKernel1( typename OP::RedElTp* d_out
               , typename OP::InpElTp* d_in
               , const uint32_t N
               , const uint32_t T
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    typename OP::InpElTp el = OP::identInp();
    uint32_t ind = 2*gid;
    if(ind < N) el = d_in[ind];
    typename OP::RedElTp el_red1 = OP::mapFun(el);
    el = OP::identInp();
    ind = ind + 1;
    if(ind < N) el = d_in[ind];
    typename OP::RedElTp el_red2 = OP::mapFun(el);
    el_red1 = OP::apply(el_red1, el_red2);
    if(gid < T) d_out[gid] = el_red1;
}

/**
 * Kernel to implement the naive reduce, which uses neither
 *   efficient sequentialization, nor fast, shared memory.
 */
template<class OP>
__global__ void
redNaiveKernel2( typename OP::RedElTp* d_out
               , const uint32_t offs_inp
               , const uint32_t offs_out
               , const uint32_t N
               , const uint32_t T
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    typename OP::RedElTp el1 = OP::identity();
    uint32_t ind = 2*gid;
    if(ind < N) el1 = d_out[offs_inp + ind];

    typename OP::RedElTp el2 = OP::identity();
    ind = ind + 1;
    if(ind < N) el2 = d_out[offs_inp + ind];
    el1 = OP::apply(el1, el2);

    if(T==1) { if (threadIdx.x == 0) d_out[0] = el1; }
    else {
        if (gid < T) d_out[offs_out + gid] = el1;
    }
}


/***********************************************************/
/*** Reduce Commutative & Non-Commutative Helper Kernels ***/
/***********************************************************/

/**
 * Kernel for reducing up to CUDA-block number of elements
 *   with CUDA-block number of threads. The generic associative
 *   binary operator does not need to be commutative.
 * This is a helper kernel for implementing the second stage of
 *   a reduction.
 */
template<class OP>
__global__ void
redAssoc1Block( typename OP::RedElTp* d_inout   // operates in place; the reduction
              , uint32_t N                      // result is written in position 0
) {
    extern __shared__ char sh_mem[];
    volatile typename OP::RedElTp* shmem_red = (typename OP::RedElTp*)sh_mem;
    typename OP::RedElTp elm = OP::identity();
    if(threadIdx.x < N) {
        elm = d_inout[threadIdx.x];
    }
    shmem_red[threadIdx.x] = elm;
    __syncthreads();
    elm = scanIncBlock<OP>(shmem_red, threadIdx.x);
    if (threadIdx.x == blockDim.x-1) {
        d_inout[0] = elm;
    }
}

/**
 * This kernel assumes that the generic-associative binary operator
 *   `OP` is also commutative. It implements the first stage of the
 *   reduction.
 * `N` is the length of the input array
 * `T` is the total number of CUDA threads spawned.
 * `d_tmp` is the result array, having number-of-blocks elements.
 * `d_in` is the input array of length `N`.
 *
 * The number of spawned blocks is <= 1024, so that the result
 *   array (having one element per block) can be reduced within
 *   one block with kernel `redAssoc1Block`.
 */
template<class OP>
__global__ void
redCommuKernel( typename OP::RedElTp* d_tmp
              , typename OP::InpElTp* d_in
              , uint32_t N
              , uint32_t T
) {
    extern __shared__ char sh_mem[];
    // shared memory holding the to-be-reduced elements.
    // The length of `shmem_red` array is the CUDA block size.
    volatile typename OP::RedElTp* shmem_red = (typename OP::RedElTp*)sh_mem;
    uint32_t gid = blockIdx.x*blockDim.x + threadIdx.x;

    // The loop efficiently sequentializes the computation by making
    // each thread iterate through the input with a stride `T` until
    // all array elements have been processed. The stride `T`
    // optimizes spatial locality: it results in coalesced
    // access to global memory (32 threads access consecutive
    // words in memory). If each thread would have processed
    // adjacent array elements (i.e., stride 1), then the penalty
    // would be severe 5-10x slower!
    typename OP::RedElTp acc = OP::identity();
    for(uint32_t ind=gid; ind < N; ind+=T) {
        // read input element
        typename OP::InpElTp elm = d_in[ind];
        // apply the mapped function and accumulate the per-thread
        // result in `acc`.
        typename OP::RedElTp red = OP::mapFun(elm);
        acc = OP::apply(acc, red);
    }

    // the per-thread results are then placed in shared memory
    // and reduced in parallel within the current CUDA-block.
    shmem_red[threadIdx.x] = acc;
    __syncthreads();
    acc = scanIncBlock<OP>(shmem_red, threadIdx.x);

    // the result of the current CUDA block is placed
    // in global memory; the position is thus given by
    // the index of the current block.
    if (threadIdx.x == blockDim.x-1) {
        d_tmp[blockIdx.x] = acc;
    }
}

/**
 * Helper function that copies `CHUNK` input elements per thread from
 *   global to shared memory, in a way that optimizes spatial locality,
 *   i.e., (32) consecutive threads read/write consecutive input elements
 *   from/to global memory in the same SIMD instruction.
 *   This leads to "coalesced" access in the case when the element size
 *   is a word (or less). Coalesced access means that (groups of 32)
 *   consecutive threads access consecutive memory words.
 *
 * `glb_offs` is the offset in global-memory array `d_inp`
 *    from where elements should be read.
 * `d_inp` is the input array stored in global memory
 * `N` is the length of `d_inp`
 * `ne` is the neutral element of `T` (think zero). In case
 *    the index of the element to be read by the current thread
 *    is out of range, then place `ne` in shared memory instead.
 * `shmem_inp` is the shared memory. It has size
 *     `blockDim.x*CHUNK*sizeof(T)`, where `blockDim.x` is the
 *     size of the CUDA block. `shmem_inp` should be filled from
 *     index `0` to index `blockDim.x*CHUNK - 1`.
 *
 * As such, a CUDA-block B of threads executing this function would
 *   read `CHUNK*B` elements from global memory and store them to
 *   (fast) shared memory, in the same order in which they appear
 *   in global memory, but making sure that consecutive threads
 *   read consecutive elements of `d_inp` in a SIMD instruction.
 *
 ********************************
 * Weekly Assignment 2, Task 1: *
 ********************************
 * The current implementations of functions `copyFromGlb2ShrMem`
 *   and `copyFromShr2GlbMem` are broken because they feature
 *   (very) uncoalesced access to global memory arrays `d_inp`
 *   and `d_out` (see above). For example, `d_inp[glb_ind]`
 *   can be expanded to `d_inp[glb_offs + threadIdx.x*CHUNK + i]`,
 *   where `threadIdx.x` denotes the local thread-id in the
 *   current block. Assuming `T` is 32-bit int, it follows that
 *   two consecutive threads are going to access in the same SIMD
 *   instruction memory locations that are CHUNK words appart
 *   (`i` being the same for both threads since they execute in
 *   lockstep).
 *  Your task is to rewrite in both functions the line
 *      `uint32_t loc_ind = threadIdx.x*CHUNK + i;`
 *    such that the result is the same---i.e., the same elements
 *    are ultimately placed at the same position---but with the
 *    new formula for computing `loc_ind`, two consecutive threads
 *    will access consecutive memory words in the same SIMD instruction.
 */
template<class T, uint32_t CHUNK>
__device__ inline void
copyFromGlb2ShrMem( const uint32_t glb_offs
                  , const uint32_t N
                  , const T& ne
                  , T* d_inp
                  , volatile T* shmem_inp
) {
    #pragma unroll
    for(uint32_t i=0; i<CHUNK; i++) {
        #if 1
        uint32_t loc_ind = threadIdx.x + i * blockDim.x;
        #else
        uint32_t loc_ind = threadIdx.x * CHUNK + i;
        #endif
        uint32_t glb_ind = glb_offs + loc_ind;
        T elm = ne;
        if(glb_ind < N) { elm = d_inp[glb_ind]; }
        shmem_inp[loc_ind] = elm;
    }
    __syncthreads(); // leave this here at the end!
}

/**
 * This is very similar with `copyFromGlb2ShrMem` except
 * that you need to copy from shared to global memory, so
 * that consecutive threads write consecutive indices in
 * global memory in the same SIMD instruction.
 * `glb_offs` is the offset in global-memory array `d_out`
 *    where elements should be written.
 * `d_out` is the global-memory array
 * `N` is the length of `d_out`
 * `shmem_red` is the shared-memory of size
 *    `blockDim.x*CHUNK*sizeof(T)`
 */
template<class T, uint32_t CHUNK>
__device__ inline void
copyFromShr2GlbMem( const uint32_t glb_offs
                  , const uint32_t N
                  , T* d_out
                  , volatile T* shmem_red
) {
    #pragma unroll
    for (uint32_t i = 0; i < CHUNK; i++) {
        #if 1
        uint32_t loc_ind = threadIdx.x + i * blockDim.x;
        #else
        uint32_t loc_ind = threadIdx.x * CHUNK + i;
        #endif
        uint32_t glb_ind = glb_offs + loc_ind;
        if (glb_ind < N) {
            T elm = const_cast<const T&>(shmem_red[loc_ind]);
            d_out[glb_ind] = elm;
        }
    }
    __syncthreads(); // leave this here at the end!
}


/**
 * This kernel assumes that the generic-associative binary operator
 *   `OP` is NOT commutative. It implements the first stage of the
 *   reduction.
 * `N` is the length of the input array
 * `CHUNK` (the template parameter) is the number of elements to
 *    be processed sequentially by a thread in one go.
 * `num_seq_chunks` is used to sequentialize even more computation,
 *    such that the number of blocks is <= 1024, hence the result
 *    array (one element per block) can be reduced within one
 *    block with kernel `redAssoc1Block`.
 * `d_tmp` is the result array, having number-of-blocks elements,
 * `d_in` is the input array of length `N`.
 */
template<class OP, int CHUNK>
__global__ void
redAssocKernel( typename OP::RedElTp* d_tmp
              , typename OP::InpElTp* d_in
              , uint32_t N
              , uint32_t num_seq_chunks
) {
    extern __shared__ char sh_mem[];
    // shared memory for the input-element and reduce-element type;
    // the two shared memories overlap, since they are not used in
    // the same time.
    volatile typename OP::InpElTp* shmem_inp = (typename OP::InpElTp*)sh_mem;
    volatile typename OP::RedElTp* shmem_red = (typename OP::RedElTp*)sh_mem;

    // initialization for the per-block result
    typename OP::RedElTp res = OP::identity();

    uint32_t num_elems_per_block = num_seq_chunks * CHUNK * blockDim.x;
    uint32_t inp_block_offs = num_elems_per_block * blockIdx.x;
    uint32_t num_elems_per_iter  = CHUNK * blockDim.x;

    // virtualization loop of count `num_seq_chunks`. Each iteration processes
    //   `blockDim.x * CHUNK` elements, i.e., `CHUNK` elements per thread.
    // `num_seq_chunks` is chosen such that it covers all N input elements
    for(int seq=0; seq<num_elems_per_block; seq+=num_elems_per_iter) {

        // 1. copy `CHUNK` input elements per thread from global to shared memory
        //    in a coalesced fashion (for global memory)
        copyFromGlb2ShrMem<typename OP::InpElTp,CHUNK>
                ( inp_block_offs + seq, N, OP::identInp(), d_in, shmem_inp );

        // 2. each thread sequentially reads its `CHUNK` elements from shared
        //     memory, applies the map function and reduces them.
        typename OP::RedElTp acc = OP::identity();
        uint32_t shmem_offset = threadIdx.x * CHUNK;
        #pragma unroll
        for (uint32_t i = 0; i < CHUNK; i++) {
            typename OP::InpElTp elm = shmem_inp[shmem_offset + i];
            typename OP::RedElTp red = OP::mapFun(elm);
            acc = OP::apply(acc, red);
        }
        __syncthreads();

        // 3. each thread publishes the previous result in shared memory
        shmem_red[threadIdx.x] = acc;
        __syncthreads();

        // 4. perform an intra-block reduction with the per-thread result
        //    from step 2; the last thread updates the per-block result `res`
        acc = scanIncBlock<OP>(shmem_red, threadIdx.x);
        if (threadIdx.x == blockDim.x-1) {
            res = OP::apply(res, acc);
        }
        __syncthreads();
        // rinse and repeat until all elements have been processed.
    }

    // 4. last thread publishes the per-block reduction result
    //    in global memory
    if (threadIdx.x == blockDim.x-1) {
        d_tmp[blockIdx.x] = res;
    }
}

/********************/
/*** Scan Kernels ***/
/********************/

/**
 * Kernel for scanning up to CUDA-block elements using
 *    CUDA-block threads.
 * `N` number of elements to be scanned (N < CUDA-block size)
 * `d_input` is the value array stored in shared memory
 *  This kernel operates in-place, i.e., the input is
 *  overwritten with the result.
 */
template<class OP>
__global__ void
scan1Block( typename OP::RedElTp* d_inout, uint32_t N ) {
    extern __shared__ char sh_mem[];
    volatile typename OP::RedElTp* shmem_red = (typename OP::RedElTp*)sh_mem;
    typename OP::RedElTp elm = OP::identity();
    if(threadIdx.x < N) {
        elm = d_inout[threadIdx.x];
    }
    shmem_red[threadIdx.x] = elm;
    __syncthreads();
    elm = scanIncBlock<OP>(shmem_red, threadIdx.x);
    if (threadIdx.x < N) {
        d_inout[threadIdx.x] = elm;
    }
}

/**
 * This kernel assumes that the generic-associative binary operator
 *   `OP` is NOT-necessarily commutative. It implements the third
 *   stage of the scan (parallel prefix sum), which scans within
 *   a block.  (The first stage is a per block reduction with the
 *   `redAssocKernel` kernel, and the second one is the `scan1Block`
 *    kernel that scans the reduced elements of each CUDA block.)
 *
 * `N` is the length of the input array
 * `CHUNK` (the template parameter) is the number of elements to
 *    be processed sequentially by a thread in one go.
 * `num_seq_chunks` is used to sequentialize even more computation,
 *    such that the number of blocks is <= 1024.
 * `d_out` is the result array of length `N`
 * `d_in`  is the input  array of length `N`
 * `d_tmp` is the array holding the per-block scanned results.
 *         it has number-of-CUDA-blocks elements, i.e., element
 *         `d_tmp[i-1]` is the scanned prefix that needs to be
 *         accumulated to each of the scanned elements corresponding
 *         to block `i`.
 * This kernels scans the elements corresponding to the current block
 *   `i`---in number of num_seq_chunks*CHUNK*blockDim.x---and then it
 *   accumulates to each of them the prefix of the previous block `i-1`,
 *   which is stored in `d_tmp[i-1]`.
 */
template<class OP, int CHUNK>
__global__ void
scan3rdKernel ( typename OP::RedElTp* d_out
              , typename OP::InpElTp* d_in
              , typename OP::RedElTp* d_tmp
              , uint32_t N
              , uint32_t num_seq_chunks
) {
    extern __shared__ char sh_mem[];
    // shared memory for the input elements (types)
    volatile typename OP::InpElTp* shmem_inp = (typename OP::InpElTp*)sh_mem;

    // shared memory for the reduce-element type; it overlaps with the
    //   `shmem_inp` since they are not going to be used in the same time.
    volatile typename OP::RedElTp* shmem_red = (typename OP::RedElTp*)sh_mem;

    // number of elements to be processed by each block
    uint32_t num_elems_per_block = num_seq_chunks * CHUNK * blockDim.x;

    // the current block start processing input elements from this offset:
    uint32_t inp_block_offs = num_elems_per_block * blockIdx.x;

    // number of elments to be processed by an iteration of the
    // "virtualization" loop
    uint32_t num_elems_per_iter  = CHUNK * blockDim.x;

    // accumulator updated at each iteration of the "virtualization"
    //   loop so we remember the prefix for the current elements.
    typename OP::RedElTp accum = (blockIdx.x == 0) ? OP::identity() : d_tmp[blockIdx.x-1];

    // register memory for storing the scanned elements.
    typename OP::RedElTp chunk[CHUNK];

    // virtualization loop of count `num_seq_chunks`. Each iteration processes
    //   `blockDim.x * CHUNK` elements, i.e., `CHUNK` elements per thread.
    for(int seq=0; seq<num_elems_per_block; seq+=num_elems_per_iter) {
        // 1. copy `CHUNK` input elements per thread from global to shared memory
        //    in coalesced fashion (for global memory)
        copyFromGlb2ShrMem<typename OP::InpElTp, CHUNK>
                  (inp_block_offs+seq, N, OP::identInp(), d_in, shmem_inp);

        // 2. each thread sequentially scans its `CHUNK` elements
        //    and stores the result in the `chunk` array. The reduced
        //    result is stored in `tmp`.
        typename OP::RedElTp tmp = OP::identity();
        uint32_t shmem_offset = threadIdx.x * CHUNK;
        #pragma unroll
        for (uint32_t i = 0; i < CHUNK; i++) {
            typename OP::InpElTp elm = shmem_inp[shmem_offset + i];
            typename OP::RedElTp red = OP::mapFun(elm);
            tmp = OP::apply(tmp, red);
            chunk[i] = tmp;
        }
        __syncthreads();

        // 3. Each thread publishes in shared memory the reduced result of its
        //    `CHUNK` elements
        shmem_red[threadIdx.x] = tmp;
        __syncthreads();

        // 4. perform an intra-CUDA-block scan
        tmp = scanIncBlock<OP>(shmem_red, threadIdx.x);
        __syncthreads();

        // 5. write the scan result back to shared memory
        shmem_red[threadIdx.x] = tmp;
        __syncthreads();

        // 6. the previous element is read from shared memory in `tmp`:
        //       it is the prefix of the previous threads in the current block.
        tmp   = OP::identity();
        if (threadIdx.x > 0)
            tmp = OP::remVolatile(shmem_red[threadIdx.x-1]);
        // 7. the prefix of the previous blocks (and iterations) is hold
        //    in `accum` and is accumulated to `tmp`, which now holds the
        //    global prefix for the `CHUNK` elements processed by the current thread.
        tmp   = OP::apply(accum, tmp);

        // 8. `accum` is also updated with the reduced result of the current
        //    iteration, i.e., of the last thread in the block: `shmem_red[blockDim.x-1]`
        accum = OP::apply(accum, shmem_red[blockDim.x-1]);
        __syncthreads();

        // 9. the `tmp` prefix is accumulated to all the `CHUNK` elements
        //      locally processed by the current thread (i.e., the ones
        //      in `chunk` array hold in registers).
        #pragma unroll
        for (uint32_t i = 0; i < CHUNK; i++) {
            shmem_red[threadIdx.x*CHUNK + i] = OP::apply(tmp, chunk[i]);
        }
        __syncthreads();

        // 5. write back from shared to global memory in coalesced fashion.
        copyFromShr2GlbMem<typename OP::RedElTp, CHUNK>
                  (inp_block_offs+seq, N, d_out, shmem_red);
    }
}

/*************************************************/
/*** Single-Pass Scan: decoupled look-back     ***/
/*************************************************/

/**
 * Look-back variants (Task 4), selected by the `LB` template
 *   parameter of `scanBlockKernel`:
 *   LB_SERIAL    : baseline, thread 0 walks back one descriptor at
 *                  a time and busy-waits on X;
 *   LB_WARP_READ : 4.1, warp 0 reads 32 descriptors in one coalesced
 *                  access into shared memory, thread 0 walks the copy;
 *   LB_WARP_PAR  : 4.2, same window, reduced with ballot + shuffles;
 *   LB_SKIP      : 4.3b, thread 0 skips X on the way back to the first
 *                  P, then revisits the skipped interval forwards;
 *   LB_WARP_PAR4 : 4.5, as 4.2 with 4 descriptors per lane (window 128),
 *                  for GPUs whose first P is usually > 32 blocks back;
 *   LB_WARP_PAR_KEEP, LB_WARP_PAR4_KEEP : 4.6, as 4.2 / 4.5, but after an X
 *                  the window is kept and only the blocking X lanes re-read.
 */
enum LookBack { LB_SERIAL = 0, LB_WARP_READ = 1, LB_WARP_PAR = 2, LB_SKIP = 3, LB_WARP_PAR4 = 4,
                LB_WARP_PAR_KEEP = 5, LB_WARP_PAR4_KEEP = 6 };

/**
 * 4.3c: debug-only instrumentation (compile with -DLOOKBACK_DEBUG).
 *   Counters are summed over all blocks of one launch; the host
 *   zeroes/reads `g_lbStats` with cudaMemcpy{To,From}Symbol.
 *   The spin guard turns a hang into a kernel error.
 */
enum LBStat { LBS_BLOCKS     // blocks that ran a look-back (bid > 0)
            , LBS_TRIPS      // global round trips (descriptor or window reads)
            , LBS_X_READS    // reads that met X where it blocked progress
            , LBS_X_BLOCKS   // blocks that met such an X at least once
            , LBS_DIST_SUM   // sum over blocks of (bid - index of the P that stopped the walk)
            , LBS_DIST_MAX
            , LBS_HIST       // 9 buckets of that distance: 1,2,3-4,5-8,9-16,17-32,33-64,65-128,>128
            , LBS_COUNT = LBS_HIST + 9 };
#ifdef LOOKBACK_DEBUG
__device__ unsigned long long g_lbStats[LBS_COUNT];
#define LB_STAT(i, x) atomicAdd(&g_lbStats[i], (unsigned long long)(x))
#define LB_SPIN_GUARD(n) if (++(n) > (1u << 26)) { printf("look-back hang: block %u\n", bid); __trap(); }
#else
#define LB_STAT(i, x)    ((void)0)
#define LB_SPIN_GUARD(n) ((void)0)
#endif

#ifdef LOOKBACK_DEBUG
__device__ inline void lbStatDistance(uint32_t d, uint32_t xreads) {
    LB_STAT(LBS_BLOCKS, 1);
    LB_STAT(LBS_X_READS, xreads);
    LB_STAT(LBS_X_BLOCKS, xreads > 0);
    LB_STAT(LBS_DIST_SUM, d);
    atomicMax(&g_lbStats[LBS_DIST_MAX], (unsigned long long)d);
    uint32_t b = (d <= 1) ? 0 : min(32 - __clz(d - 1), 8);  // ceil(log2 d), capped
    LB_STAT(LBS_HIST + b, 1);
}
#define LB_STAT_DIST(d, xreads) lbStatDistance((d), (xreads))
#else
#define LB_STAT_DIST(d, xreads) ((void)0)
#endif

/**
 * Descriptor of block j, in one of two layouts:
 *   - separate arrays (default): flag `Fs[j]`, aggregate `As[j]`,
 *     inclusive prefix `Ps[j]`;
 *   - packed (4.3a), elements of at most 32 bits: one 64-bit word
 *     `Ds[j] = flag << 32 | value bits` (bit copy, so floats survive
 *     unchanged);
 *   - packed, 64-bit elements: two 64-bit words, each carrying its own
 *     copy of the flag, `Ds[2j] = flag << 32 | low 32 value bits`,
 *     `Ds[2j+1] = flag << 32 | high 32 value bits`. Each word is
 *     written once per flag (I4), so two words with equal flags come
 *     from the same publish; unequal flags read as X (retry, I3).
 *   X encodes as 0, so a zero memset of `Ds` resets every flag (I6).
 * These two functions are the only places that touch descriptors,
 *   so the I5 ordering is written once.
 */
template<class T, bool PACKED>
__device__ inline void
publishDesc(uint32_t j, Status f, T v, Status* Fs, T* Ps, T* As, unsigned long long* Ds) {
    if constexpr (PACKED && sizeof(T) == 8) {
        unsigned long long bits;
        memcpy(&bits, &v, 8);
        // each half carries the flag: no order needed between the two stores
        //   (a reader that sees only one of them gets unequal flags)
        atomic_ref<unsigned long long, thread_scope_device>(Ds[2 * j])
            .store(((unsigned long long)f << 32) | (uint32_t)bits, memory_order_relaxed);
        atomic_ref<unsigned long long, thread_scope_device>(Ds[2 * j + 1])
            .store(((unsigned long long)f << 32) | (bits >> 32), memory_order_relaxed);
    } else if constexpr (PACKED) {
        uint32_t bits = 0;
        memcpy(&bits, &v, sizeof(T));     // copy the bits: a numeric cast would truncate floats
        // flag and value travel in one 64-bit store: nothing to order between them
        atomic_ref<unsigned long long, thread_scope_device>(Ds[j])
            .store(((unsigned long long)f << 32) | bits, memory_order_relaxed);
    } else {
        if (f == P) Ps[j] = v; else As[j] = v;
        // release: the value store above becomes visible no later than the flag (I5 writer)
        atomic_ref<Status, thread_scope_device>(Fs[j]).store(f, memory_order_release);
    }
}

// Returns the flag of block j; `v` is its aggregate (A) or inclusive prefix (P), undefined for X.
template<class T, bool PACKED>
__device__ inline Status
loadDesc(int j, T& v, Status* Fs, T* Ps, T* As, unsigned long long* Ds) {
    if constexpr (PACKED && sizeof(T) == 8) {
        // two independent loads, each with its own flag: nothing to order between them
        unsigned long long lo = atomic_ref<unsigned long long, thread_scope_device>(Ds[2 * j])
                                    .load(memory_order_relaxed);
        unsigned long long hi = atomic_ref<unsigned long long, thread_scope_device>(Ds[2 * j + 1])
                                    .load(memory_order_relaxed);
        if ((lo >> 32) != (hi >> 32)) return X;  // halves of different publishes: not ready yet
        unsigned long long bits = (hi << 32) | (uint32_t)lo;
        memcpy(&v, &bits, 8);
        return (Status)(lo >> 32);
    } else if constexpr (PACKED) {
        // one 64-bit load returns flag and value together: nothing to order between them
        unsigned long long w = atomic_ref<unsigned long long, thread_scope_device>(Ds[j])
                                   .load(memory_order_relaxed);
        uint32_t bits = (uint32_t)w;
        memcpy(&v, &bits, sizeof(T));     // low sizeof(T) bytes (little-endian), inverse of publishDesc
        return (Status)(w >> 32);
    } else {
        // acquire: the value load below cannot be performed before the flag load (I5 reader)
        Status f = atomic_ref<Status, thread_scope_device>(Fs[j]).load(memory_order_acquire);
        if (f == P) v = Ps[j];
        else if (f == A) v = As[j];
        return f;
    }
}

/**
 * 4.4 local-phase tile geometry. The tile holds one row per thread,
 *   CHUNK elements = 4*ELEMS_PER_THREAD bytes, stored in units U:
 *   4-byte words (8 bytes for 64-bit elements), E elements per unit,
 *   UPT units per row. Global copies move whole units (one 32/64-bit
 *   access instead of up to 4 narrow ones). The row pitch is odd in
 *   units, so 32 threads reading the same column of their own rows hit
 *   32 different banks (an even pitch, e.g. 24 words, gives 8-way
 *   conflicts).
 */
template<class T, int CHUNK>
struct TileGeom {
    using U = typename ::cuda::std::conditional<(sizeof(T) > 4), unsigned long long, uint32_t>::type;
    static constexpr uint32_t E     = sizeof(U) / sizeof(T);
    static constexpr uint32_t UPT   = CHUNK / E;
    static constexpr uint32_t PITCH = UPT | 1;
    static_assert(CHUNK % E == 0, "a thread's elements must fill whole units");
};

/**
 *******************************************************
 * Single-Pass Scan (Merrill & Garland, decoupled look-back)
 *******************************************************
 * Each CUDA block scans ONE chunk of CHUNK*blockDim.x elements and
 *   obtains its exclusive offset from its predecessors' descriptors.
 *   Chunk order is the DYNAMIC block index `bid` (atomic counter),
 *   never `blockIdx.x` (I7).
 * `LB` selects the look-back variant (see `LookBack`); `PACKED`
 *   selects the packed descriptor layout (4.3a; two words per
 *   descriptor for 64-bit elements, see `publishDesc`).
 * `LOCAL` selects the per-block (local) phase:
 *   false: Task 3 local phase (copy helpers + `scanIncBlock`, 12 barriers);
 *   true : 4.4 local phase, see `TileGeom` and the steps below (4 barriers).
 * The host must reset `*g_block_counter` and the flags (`Fs`, or
 *   `Ds` if packed) to 0 before every launch (I6).
 */
template<class OP, int CHUNK, int LB, bool PACKED, bool LOCAL>
__global__ void
scanBlockKernel ( typename OP::RedElTp* d_out
                , typename OP::InpElTp* d_in
                , uint32_t N
                , Status* Fs
                , typename OP::RedElTp* Ps
                , typename OP::RedElTp* As
                , unsigned long long* Ds
                , uint32_t *g_block_counter
) {
    using T = typename OP::RedElTp;
    constexpr bool PK = PACKED;


    // step 2, global block counter
    __shared__ uint32_t bid;

    // only increment the block counter when the first thread spawns
    if (threadIdx.x == 0) {
        // atomic add also prevents race conditions
        bid = atomicAdd(g_block_counter, 1);
    }

    // ensure that the threads gets the bid before
    __syncthreads();


    extern __shared__ char sh_mem[];
    // shared memory for the input elements (types)
    volatile typename OP::InpElTp* shmem_inp = (typename OP::InpElTp*)sh_mem;

    // shared memory for the reduce-element type; it overlaps with the
    //   `shmem_inp` since they are not going to be used in the same time.
    volatile typename OP::RedElTp* shmem_red = (typename OP::RedElTp*)sh_mem;

    // the current block start processing input elements from this offset:
    uint32_t inp_block_offs = CHUNK * blockDim.x * bid;

    // register memory for storing the scanned elements (Task 3 local phase only).
    typename OP::RedElTp chunk[LOCAL ? 1 : CHUNK];
    typename OP::RedElTp tmp = OP::identity();

    // 4.4 local phase: tile in shared memory as one row per thread, in units
    //   of U (4-byte words, or 8 bytes for 64-bit elements); see `TileGeom`.
    using G = TileGeom<T, CHUNK>;
    using U = typename G::U;
    U* tile = (U*)sh_mem;
    const uint32_t row = threadIdx.x * G::PITCH;          // this thread's row
    const size_t   blk_unit = (size_t)bid * blockDim.x * G::UPT;
    const size_t   full_units = N / G::E;                 // units entirely inside the array
    const bool     full_tile  = blk_unit + (size_t)blockDim.x * G::UPT <= full_units;  // block-uniform
    T thrExcl = OP::identity();                           // block-local exclusive prefix of this thread

    T blockAggregate;                     // valid in thread 0 only

    if constexpr (!LOCAL) {
        // 1. copy `CHUNK` input elements per thread from global to shared memory
        //    in coalesced fashion (for global memory)
        copyFromGlb2ShrMem<typename OP::InpElTp, CHUNK>
                  (inp_block_offs, N, OP::identInp(), d_in, shmem_inp);

        // 2. each thread sequentially scans its `CHUNK` elements
        //    and stores the result in the `chunk` array. The reduced
        //    result is stored in `tmp`.
        uint32_t shmem_offset = threadIdx.x * CHUNK;
        #pragma unroll
        for (uint32_t i = 0; i < CHUNK; i++) {
            typename OP::InpElTp elm = shmem_inp[shmem_offset + i];
            typename OP::RedElTp red = OP::mapFun(elm);
            tmp = OP::apply(tmp, red);
            chunk[i] = tmp;
        }
        __syncthreads();  // all reads of shmem_inp done before shmem_red (same bytes) is written

        // 3. Each thread publishes in shared memory the reduced result of its
        //    `CHUNK` elements
        shmem_red[threadIdx.x] = tmp;
        __syncthreads();  // per-thread sums visible to the block-level scan

        // 4. perform an intra-CUDA-block scan
        tmp = scanIncBlock<OP>(shmem_red, threadIdx.x);
        __syncthreads();  // scan reads of shmem_red done before it is overwritten below

        // 5. write the scan result back to shared memory
        shmem_red[threadIdx.x] = tmp;
        __syncthreads();  // scanned sums visible: thread 0 reads the aggregate, threads read [tid-1]

        if(threadIdx.x == 0){
            blockAggregate = OP::remVolatile(shmem_red[blockDim.x - 1]);
        }
    } else {
        static_assert(sizeof(typename OP::InpElTp) == sizeof(T), "4.4 local phase: map must preserve size");
        // L1. coalesced load in units: unit w = tid + k*B of the block goes to
        //     row w/UPT, column w%UPT. Full tiles (all blocks but the last) take
        //     the unchecked path; in the last tile a unit straddling N is
        //     assembled element by element with the identity beyond N.
        const U* in_u = (const U*)d_in + blk_unit;      // this block's first unit
        if (full_tile) {
            #pragma unroll
            for (uint32_t k = 0; k < G::UPT; k++) {
                const uint32_t w = threadIdx.x + k * blockDim.x;
                tile[(w / G::UPT) * G::PITCH + w % G::UPT] = in_u[w];
            }
        } else {
            for (uint32_t k = 0; k < G::UPT; k++) {
                const uint32_t w  = threadIdx.x + k * blockDim.x;
                const size_t   gu = blk_unit + w;
                U u;
                if (gu < full_units) {
                    u = in_u[w];
                } else {
                    T e[G::E];
                    #pragma unroll
                    for (uint32_t q = 0; q < G::E; q++) {
                        const size_t gi = gu * G::E + q;
                        e[q] = (gi < N) ? (T)d_in[gi] : OP::identInp();
                    }
                    memcpy(&u, e, sizeof(U));
                }
                tile[(w / G::UPT) * G::PITCH + w % G::UPT] = u;
            }
        }
        __syncthreads();  // whole tile in shared memory before each thread reads its own row

        // L2. thread total over its own row (conflict-free: odd row pitch)
        T total = OP::identity();
        #pragma unroll
        for (uint32_t k = 0; k < G::UPT; k++) {
            U u = tile[row + k];
            T e[G::E];
            memcpy(e, &u, sizeof(U));
            #pragma unroll
            for (uint32_t q = 0; q < G::E; q++) total = OP::apply(total, OP::mapFun(e[q]));
        }

        // L3. block scan of thread totals in registers: warp scan with shuffles,
        //     then the (at most 32) warp totals through shared memory
        const uint32_t lane = threadIdx.x & (WARP - 1), warp = threadIdx.x >> lgWARP;
        T x = total;
        #pragma unroll
        for (uint32_t d = 1; d < WARP; d <<= 1) {
            T y = __shfl_up_sync(0xFFFFFFFF, x, d);
            if (lane >= d) x = OP::apply(y, x);
        }
        T warpExcl = __shfl_up_sync(0xFFFFFFFF, x, 1);
        if (lane == 0) warpExcl = OP::identity();
        __shared__ T warpTot[WARP];
        if (lane == WARP - 1) warpTot[warp] = x;
        __syncthreads();  // warp totals visible to every thread
        T before = OP::identity();        // reduction of the warps before this one
        for (uint32_t w = 0; w < warp; w++) before = OP::apply(before, warpTot[w]);
        thrExcl = OP::apply(before, warpExcl);
        if (threadIdx.x == 0) {
            blockAggregate = OP::identity();
            for (uint32_t w = 0; w < blockDim.x / WARP; w++) blockAggregate = OP::apply(blockAggregate, warpTot[w]);
        }
    }

    // decoupled look-back: publish A, find the exclusive offset, publish P.
    //    Prefixes are combined as apply(earlier, later), so OP need not commute.
    __shared__ T blockOffset;

    if (bid == 0) {
        if (threadIdx.x == 0) {
            blockOffset = OP::identity();
            publishDesc<T, PK>(0, P, blockAggregate, Fs, Ps, As, Ds);
        }
    } else if (threadIdx.x < WARP) {      // warp 0 runs the look-back (only thread 0 in serial variants)
        const uint32_t lane = threadIdx.x;
        // A before the look-back, so successors need not wait for our P
        if (lane == 0) publishDesc<T, PK>(bid, A, blockAggregate, Fs, Ps, As, Ds);
        T prefix = OP::identity();        // offset accumulated so far; valid in lane 0
        [[maybe_unused]] uint32_t xreads = 0;  // reads that met a blocking X (stats + spin guard)

        if constexpr (LB == LB_SERIAL) {
            // baseline: walk back one descriptor per round trip, busy-wait on X
            if (lane == 0) {
                int j = (int)bid - 1;
                while (true) {            // terminates: block 0 always ends with P
                    T v;
                    Status f = loadDesc<T, PK>(j, v, Fs, Ps, As, Ds);
                    LB_STAT(LBS_TRIPS, 1);
                    if (f == X) { LB_SPIN_GUARD(xreads); continue; }  // re-read, never skip (I3)
                    prefix = OP::apply(v, prefix);
                    if (f == P) break;    // P covers 0..j (I2)
                    j--;
                }
                LB_STAT_DIST(bid - j, xreads);
            }
        } else if constexpr (LB == LB_SKIP) {
            // 4.3b: backwards to the first P without waiting on X, then
            //   forwards over the skipped interval, waiting where still X.
            //   Aggregates read before the first X (blocks hi+1..bid-1) are
            //   kept in `suffix` and not re-read (valid by I4).
            if (lane == 0) {
                [[maybe_unused]] uint32_t skipped = 0;        // number of X blocks skipped (stats)
                int j = (int)bid - 1;
                int hi = -1;              // highest block seen as X; -1 = none so far
                T suffix = OP::identity();
                T v;
                Status f;
                while (true) {
                    f = loadDesc<T, PK>(j, v, Fs, Ps, As, Ds);
                    LB_STAT(LBS_TRIPS, 1);
                    if (f == P) break;
                    if (f == X) {
                        if (j == 0) { LB_SPIN_GUARD(xreads); continue; }  // nothing before block 0 to skip to
                        skipped++;
                        if (hi < 0) hi = j;
                    } else if (hi < 0) {
                        suffix = OP::apply(v, suffix);
                    }
                    j--;
                }
                prefix = v;               // Ps[j] covers 0..j
                for (int i = j + 1; i <= hi; ) {
                    f = loadDesc<T, PK>(i, v, Fs, Ps, As, Ds);
                    LB_STAT(LBS_TRIPS, 1);
                    if (f == X) { LB_SPIN_GUARD(xreads); continue; }  // wait: skipping again would leave a hole (I1, I3)
                    prefix = (f == P) ? v : OP::apply(prefix, v);     // P covers 0..i: replace
                    i++;
                }
                prefix = OP::apply(prefix, suffix);
                LB_STAT_DIST(bid - j, xreads + skipped);
            }
        } else if constexpr (LB == LB_WARP_READ) {
            // 4.1: lane i reads block base-i (one coalesced access); lane 0 walks the copy
            __shared__ Status winF[WARP];
            __shared__ T      winV[WARP];
            int base = (int)bid - 1;
            while (true) {
                int j = base - (int)lane;
                T v = OP::identity();
                Status f = X;             // lanes before block 0 read nothing; the walk stops at block 0
                if (j >= 0) f = loadDesc<T, PK>(j, v, Fs, Ps, As, Ds);
                winF[lane] = f;
                winV[lane] = v;
                if (lane == 0) LB_STAT(LBS_TRIPS, 1);
                __syncwarp();             // window writes visible to lane 0
                int state = 0;            // 0: X before the first P, re-read; 1: no P, slide; 2: done
                if (lane == 0) {
                    T acc = OP::identity();
                    int i = 0;
                    state = 1;
                    for (; i < WARP; i++) {
                        Status fi = winF[i];
                        if (fi == X) { state = 0; break; }  // never skip X (I3)
                        acc = OP::apply(winV[i], acc);
                        if (fi == P) { state = 2; break; }  // I2
                    }
                    if (state != 0) prefix = OP::apply(acc, prefix);
                    else LB_SPIN_GUARD(xreads);
                    if (state == 2) LB_STAT_DIST(bid - (base - i), xreads);
                }
                state = __shfl_sync(0xFFFFFFFF, state, 0);  // warp-uniform loop decision
                __syncwarp();             // lane 0 is done reading the window before it is overwritten
                if (state == 2) break;
                if (state == 1) base -= WARP;
            }
        } else if constexpr (LB == LB_WARP_PAR || LB == LB_WARP_PAR4 ||
                             LB == LB_WARP_PAR_KEEP || LB == LB_WARP_PAR4_KEEP) {
            // 4.2: same window as 4.1, combined in parallel by the warp.
            //   LB_WARP_PAR4 (4.5): K = 4 descriptors per lane, window of 128.
            //   Round r: lane l reads rank l + 32r, i.e. block base - rank
            //   (one coalesced access per round, all K rounds in flight together).
            //   *_KEEP (4.6): after a blocking X the window is kept; the next pass
            //   re-reads only the X lanes before the closest P. Best effort: an A
            //   already read is reused (I4) even if its block has since published P.
            constexpr int K = (LB == LB_WARP_PAR4 || LB == LB_WARP_PAR4_KEEP) ? 4 : 1;
            constexpr bool KEEP = (LB == LB_WARP_PAR_KEEP || LB == LB_WARP_PAR4_KEEP);
            int base = (int)bid - 1;
            T v[K];
            Status f[K];
            int pr = K;                   // round holding the closest P; K = none
            unsigned upto = 0xFFFFFFFF;   // in round pr: lanes 0..(closest P) inclusive
            #pragma unroll
            for (int r = 0; r < K; r++) f[r] = X;  // nothing read yet: every lane reads
            while (true) {
                #pragma unroll
                for (int r = 0; r < K; r++) {
                    bool blocking = f[r] == X && (r < pr || (r == pr && ((upto >> lane) & 1)));
                    if (!KEEP || blocking) {
                        int j = base - (int)(lane + WARP * r);
                        v[r] = OP::identity();
                        f[r] = A;         // ranks before block 0: identity, never counted as X
                        if (j >= 0) f[r] = loadDesc<T, PK>(j, v[r], Fs, Ps, As, Ds);
                    }
                }
                if (lane == 0) LB_STAT(LBS_TRIPS, 1);
                // walk rounds in rank order: closest P = lowest lane of the first round
                //   that has one; an X at a smaller rank blocks the walk at its round (I3)
                pr = K;
                upto = 0xFFFFFFFF;
                int xr = K;               // round of the first blocking X; rounds before it are all A
                #pragma unroll
                for (int r = 0; r < K; r++) {
                    unsigned pmask = __ballot_sync(0xFFFFFFFF, f[r] == P);
                    unsigned xmask = __ballot_sync(0xFFFFFFFF, f[r] == X);
                    if (pr == K) {        // still at ranks before the closest P
                        unsigned up = (pmask & (0u - pmask)) * 2 - 1;  // all lanes if no P
                        if (xr == K && (xmask & up)) xr = r;
                        if (pmask) { pr = r; upto = up; }
                    }
                }
                if (KEEP && xr < K) {     // window kept, next pass re-reads the blocking X lanes (I3)
                    if (lane == 0) LB_SPIN_GUARD(xreads);
                    continue;
                }
                // rounds 0..pr (or the all-A rounds 0..xr-1), each an ordered tree
                //   reduction (lane i+d holds blocks earlier than lane i); round r is
                //   earlier than round r-1
                #pragma unroll
                for (int r = 0; r < K; r++) {
                    if (r > pr || r >= xr) break;  // warp-uniform
                    T x = (r < pr || ((upto >> lane) & 1)) ? v[r] : OP::identity();
                    #pragma unroll
                    for (int d = 1; d < WARP; d <<= 1) {
                        T y = __shfl_down_sync(0xFFFFFFFF, x, d);
                        if (lane + d < WARP) x = OP::apply(y, x);
                    }
                    if (lane == 0) prefix = OP::apply(x, prefix);
                }
                if (xr < K) {             // X before the closest P: keep the all-A rounds
                    if (lane == 0) LB_SPIN_GUARD(xreads);  //   (I4), re-read from round xr (I3)
                    base -= WARP * xr;
                    continue;
                }
                if (pr < K) {             // I2
                    if (lane == 0) LB_STAT_DIST(bid - (base - (WARP * pr + __popc(upto) - 1)), xreads);
                    break;
                }
                base -= WARP * K;
                #pragma unroll
                for (int r = 0; r < K; r++) f[r] = X;  // new window: values of the old one must not be reused (I1)
            }
        }

        if (lane == 0) {
            blockOffset = prefix;
#ifndef LOOKBACK_NO_P
            publishDesc<T, PK>(bid, P, OP::apply(prefix, blockAggregate), Fs, Ps, As, Ds);
#endif
        }
    }
    __syncthreads();                      // blockOffset visible to the whole block

    if constexpr (!LOCAL) {
        // 6. the previous element is read from shared memory in `tmp`:
        //       it is the prefix of the previous threads in the current block.
        tmp   = OP::identity();
        if (threadIdx.x > 0)
            tmp = OP::remVolatile(shmem_red[threadIdx.x-1]);
        tmp = OP::apply(blockOffset, tmp);
        // KEEP this barrier: it separates the reads of `shmem_red[threadIdx.x-1]`
        //   above from the writes to `shmem_red[threadIdx.x*CHUNK + i]` below,
        //   which overlap. (In `scan3rdKernel` it sat after the `accum` update.)
        __syncthreads();

        // 9. the `tmp` prefix is accumulated to all the `CHUNK` elements
        //      locally processed by the current thread (i.e., the ones
        //      in `chunk` array hold in registers).
        #pragma unroll
        for (uint32_t i = 0; i < CHUNK; i++) {
            shmem_red[threadIdx.x*CHUNK + i] = OP::apply(tmp, chunk[i]);
        }
        __syncthreads();  // results in shared memory before the coalesced (transposed) copy-out

        // 5. write back from shared to global memory in coalesced fashion.
        copyFromShr2GlbMem<typename OP::RedElTp, CHUNK>
                  (inp_block_offs, N, d_out, shmem_red);
    } else {
        // L4. re-read the own row and overwrite it in place with the final
        //     inclusive prefixes (no per-element registers kept across the look-back)
        T run = OP::apply(blockOffset, thrExcl);
        #pragma unroll
        for (uint32_t k = 0; k < G::UPT; k++) {
            U u = tile[row + k];
            T e[G::E];
            memcpy(e, &u, sizeof(U));
            #pragma unroll
            for (uint32_t q = 0; q < G::E; q++) { run = OP::apply(run, OP::mapFun(e[q])); e[q] = run; }
            memcpy(&u, e, sizeof(U));
            tile[row + k] = u;
        }
        __syncthreads();  // all rows final before the striped copy-out reads other threads' rows

        // L5. coalesced store in units, mirror of L1; elements beyond N are not written
        U* out_u = (U*)d_out + blk_unit;
        if (full_tile) {
            #pragma unroll
            for (uint32_t k = 0; k < G::UPT; k++) {
                const uint32_t w = threadIdx.x + k * blockDim.x;
                out_u[w] = tile[(w / G::UPT) * G::PITCH + w % G::UPT];
            }
        } else {
            for (uint32_t k = 0; k < G::UPT; k++) {
                const uint32_t w  = threadIdx.x + k * blockDim.x;
                const size_t   gu = blk_unit + w;
                U u = tile[(w / G::UPT) * G::PITCH + w % G::UPT];
                if (gu < full_units) {
                    out_u[w] = u;
                } else {
                    T e[G::E];
                    memcpy(e, &u, sizeof(U));
                    #pragma unroll
                    for (uint32_t q = 0; q < G::E; q++) {
                        const size_t gi = gu * G::E + q;
                        if (gi < N) d_out[gi] = e[q];
                    }
                }
            }
        }
    }
}


/*************************************************/
/*************************************************/
/*** Segmented Inclusive Scan Helpers & Kernel ***/
/*************************************************/
/*************************************************/

/**
 * A warp of threads cooperatively segmented-scans with generic-binop
 *   `OP` a number of warp value-flag elements stored in shared memory
 *   arrays `ptr` and `flg`, respectively.
 * `F` template type is always `char`.
 * `OP` is the associative operator of segmented scan.
 * `idx` is the local thread index within a cuda block (threadIdx.x)
 * No synchronization is needed because the thread in a warp execute
 *   in lockstep.
 * Each thread returns the corresponding scanned element of type
 *   `ValFlg<typename OP::RedElTp>`
 */
template<class OP, class F>
__device__ inline ValFlg<typename OP::RedElTp>
sgmScanIncWarp(volatile typename OP::RedElTp* ptr, volatile F* flg, const uint32_t idx) {
    typedef ValFlg<typename OP::RedElTp> FVTup;
    const uint32_t lane = idx & (WARP-1);

    // no synchronization needed inside a WARP, i.e., SIMD execution
    #pragma unroll
    for(uint32_t i=0; i<lgWARP; i++) {
        const uint32_t p = (1<<i);
        if( lane >= p ) {
            if(flg[idx] == 0) { ptr[idx] = OP::apply(ptr[idx-p], ptr[idx]); }
            flg[idx] = flg[idx-p] | flg[idx];
        } // __syncwarp();
    }

    F f = flg[idx];
    typename OP::RedElTp v = OP::remVolatile(ptr[idx]);
    return FVTup( f, v );
}

/**
 * A CUDA-block of threads cooperatively perform a segmented scan with
 *   generic-binop `OP` on a CUDA-block number of elements whose values
 *   and flags are stored in shared-memory arrays `ptr` and `flg`, respectively.
 * `F` template type is always `char`
 * `idx` is the local thread index within a cuda block (threadIdx.x)
 * Each thread returns the corresponding scanned element of type
 *   `typename OP::RedElTp`; note that this is NOT published to shared memory!
 */
template<class OP, class F>
__device__ inline ValFlg<typename OP::RedElTp>
sgmScanIncBlock(volatile typename OP::RedElTp* ptr, volatile F* flg, const uint32_t idx) {
    typedef ValFlg<typename OP::RedElTp> FVTup;
    const uint32_t lane   = idx & (WARP-1);
    const uint32_t warpid = idx >> lgWARP;

    // 1. perform scan at warp level
    FVTup res = sgmScanIncWarp<OP,F>(ptr, flg, idx);
    __syncthreads();

    // 2. if last thread in a warp, record it at the beginning of sh_data
    if ( lane == (WARP-1) ) { flg[warpid] = res.f; ptr[warpid] = res.v; }
    __syncthreads();

    // 3. first warp scans the per warp results (again)
    if( warpid == 0 ) sgmScanIncWarp<OP,F>(ptr, flg, idx);
    __syncthreads();

    // 4. accumulate results from previous step;
    if (warpid > 0) {
        FVTup prev;
        prev.f = (char) flg[warpid-1];
        prev.v = OP::remVolatile(ptr[warpid-1]);
        res = LiftOP<OP>::apply( prev, res );
    }
    //__syncthreads();
    //flg[idx] = res.f;
    //ptr[idx] = res.v;
    //__syncthreads();
    return res;
}

////////////////////////////////////////////

/**
 * Kernel for scanning up to CUDA-block elements using
 *    CUDA-block threads.
 * `N` number of elements to be scanned (N < CUDA-block size)
 * `d_vals` is the value array stored in shared memory
 * `d_flag` is the flag  array stored in shared memory
 *  This kernel operates in-place, i.e., the input is
 *  overwritten with the result.
 */
template<class OP>
__global__ void
sgmScan1Block( typename OP::RedElTp* d_vals
             , char*                 d_flag
             , uint32_t N
) {
    extern __shared__ char sh_mem[];
    volatile typename OP::RedElTp* shmem_red = (volatile typename OP::RedElTp*)sh_mem;
    volatile char*                 shmem_flg = (volatile char*)(shmem_red + blockDim.x);
    typename OP::RedElTp elm = OP::identity();
    char flg = 0;
    if(threadIdx.x < N) {
        elm = d_vals[threadIdx.x];
        flg = d_flag[threadIdx.x];
    }
    shmem_red[threadIdx.x] = elm;
    shmem_flg[threadIdx.x] = flg;

    __syncthreads();

    ValFlg<typename OP::RedElTp> fv = //(blockDim.x == 32) ?
        //sgmScanIncWarp <OP,char>(shmem_red, shmem_flg, threadIdx.x);
        sgmScanIncBlock<OP,char>(shmem_red, shmem_flg, threadIdx.x) ;

    if (threadIdx.x < N) {
        d_vals[threadIdx.x] = fv.v;
        d_flag[threadIdx.x] = fv.f;
    }
}

/**
 * This kernel implements the first stage of a segmented scan.
 * In essence, it reduces the elements to tbe processed by the
 * current CUDA block (in number of `num_seq_chunks*CHUNK*blockDim.x`).
 *
 * The implementation is very similar to `redAssocKernel` except for
 * the input and result flag arrays `d_flag` and `d_tmp_flag`, and
 * for using the extended operator for segmented reduction---that
 * works on flag-value pairs---and is implemented by `LiftOP<OP>`
 */
template<class OP, int CHUNK>
__global__ void
redSgmScanKernel( char*                 d_tmp_flag
                , typename OP::RedElTp* d_tmp_vals
                , char*                 d_flag
                , typename OP::InpElTp* d_in
                , uint32_t N
                , uint32_t num_seq_chunks
) {
    typedef ValFlg<typename OP::RedElTp> FVTup;
    extern __shared__ char sh_mem[];
    volatile typename OP::InpElTp* shmem_inp = (typename OP::InpElTp*)sh_mem;
    volatile typename OP::RedElTp* shmem_red = (typename OP::RedElTp*)sh_mem;

    FVTup res = LiftOP<OP>::identity();

    uint32_t num_elems_per_block = num_seq_chunks * CHUNK * blockDim.x;
    uint32_t inp_block_offs = num_elems_per_block * blockIdx.x;
    uint32_t num_elems_per_iter = CHUNK * blockDim.x;

    // virtualization loop of count `num_seq_chunks`. Each iteration processes
    //   `blockDim.x * CHUNK` elements, i.e., `CHUNK` elements per thread.
    for(int seq=0; seq<num_elems_per_block; seq+=num_elems_per_iter) {
        volatile char* shmem_flg = (volatile char*)(shmem_inp + CHUNK*blockDim.x);

        // 1. copy `CHUNK` input elements per thread
        //    from global to shared memory (both values and flags)
        copyFromGlb2ShrMem<typename OP::InpElTp, CHUNK>
                  (inp_block_offs+seq, N, OP::identInp(), d_in, shmem_inp);
        copyFromGlb2ShrMem<char, CHUNK>
                  (inp_block_offs+seq, N, 0, d_flag, shmem_flg);

        // 2. each thread sequentially reduces its `CHUNK` elements
        FVTup acc = LiftOP<OP>::identity();
        uint32_t shmem_offset = threadIdx.x * CHUNK;
        #pragma unroll
        for (uint32_t i = 0; i < CHUNK; i++) {
            typename OP::InpElTp elm = shmem_inp[shmem_offset + i];
            typename OP::RedElTp red = OP::mapFun(elm);
            char flg = shmem_flg[shmem_offset + i];
            acc = LiftOP<OP>::apply( acc, FVTup(flg, red) );
        }
        __syncthreads();

        shmem_flg = (volatile char*)(shmem_red + blockDim.x);

        // 3. perform an intra-block reduction with the per-thread result
        //    from step 2. and the last thread updates the per-block result
        shmem_red[threadIdx.x] = acc.v;
        shmem_flg[threadIdx.x] = acc.f;
        __syncthreads();

        acc = sgmScanIncBlock<OP,char>(shmem_red, shmem_flg, threadIdx.x);

        if (threadIdx.x == blockDim.x-1) {
            res = LiftOP<OP>::apply(res, acc);
        }
        __syncthreads();
    }

    // 4. publish result in global memory
    if (threadIdx.x == blockDim.x-1) {
        d_tmp_flag[blockIdx.x] = res.f;
        d_tmp_vals[blockIdx.x] = res.v;
    }
}

/**
 * This kernel assumes that the generic-associative binary operator
 *   `OP` is NOT-necessarily commutative. It implements the third
 *   stage of the segmented scan (parallel prefix sum), which scans within
 *   a block.  (The first stage is a per block reduction with the
 *   `redSgmScanKernel` kernel, and the second one is the `sgmScan1Block`
 *   kernel that scan in one block the results of the first stage.)
 *
 * `N` is the length of the input array
 * `CHUNK` (the template parameter) is the number of elements to
 *    be processed sequentially by a thread in one go.
 * `num_seq_chunks` is used to sequentialize even more computation,
 *    such that the number of blocks is <= 1024
 * `d_out` is the result array of length `N`
 * `d_inp` is the input  array of length `N`
 * `d_tmp_vals` and `d_tmp_flag` are the arrays holding the per-block
 *     scanned results (of values and flags), and they have
 *     number-of-CUDA-blocks elements, i.e., element
 *     `(d_tmp_flag[i-1], d_tmp_vals[i-1])` is the scanned prefix that
 *     needs to be accumulated to each of the scanned elements
 *     corresponding to block `i`.
 * This kernels scans the elements corresponding to the current block
 *   `i`---in number of num_seq_chunks*CHUNK*blockDim.x---and then it
 *   accumulates to each of them the prefix of the previous block `i-1`.
 *
 * The implementation is very similar to `scan3rdKernel` kernel.
 */
template<class OP, int CHUNK>
__global__ void
sgmScan3rdKernel ( typename OP::RedElTp* d_out
                 , typename OP::InpElTp* d_inp
                 , char*                 d_flg
                 , typename OP::RedElTp* d_tmp_vals
                 , char*                 d_tmp_flag
                 , uint32_t N
                 , uint32_t num_seq_chunks
) {
    // datatype for a flag-value pair
    typedef ValFlg<typename OP::RedElTp> FVTup;

    // externally declared shared memory
    extern __shared__ char sh_mem[];

    // shared memory for the input and mapped elements; they overlap,
    // since they are not to be used in the same time.
    volatile typename OP::InpElTp* shmem_inp = (typename OP::InpElTp*)sh_mem;
    volatile typename OP::RedElTp* shmem_red = (typename OP::RedElTp*)sh_mem;

    uint32_t num_elems_per_block = num_seq_chunks * CHUNK * blockDim.x;
    uint32_t inp_block_offs = num_elems_per_block * blockIdx.x;
    uint32_t num_elems_per_iter  = CHUNK * blockDim.x;

    // everybody reads the flag-value prefix corresponding to
    //   the previous CUDA block if any, which is stored in `acum`.
    FVTup accum;
    if (blockIdx.x > 0) {
        accum.f = d_tmp_flag[blockIdx.x-1];
        accum.v = d_tmp_vals[blockIdx.x-1];
    } else {
        accum = LiftOP<OP>::identity();
    }

    // register-allocated array for holding the `CHUNK` elements that
    //   are to be processed sequentially (individually) by each thread.
    FVTup chunk[CHUNK];

    // virtualization loop of count `num_seq_chunks`. Each iteration processes
    //   `blockDim.x * CHUNK` elements, i.e., `CHUNK` elements per thread.
    for(int seq=0; seq<num_elems_per_block; seq+=num_elems_per_iter) {
        volatile char* shmem_flg = (volatile char*)(shmem_inp + CHUNK*blockDim.x);

        // 1. copy `CHUNK` input elements per thread from
        //    global to shared memory in coalesced fashion
        copyFromGlb2ShrMem<typename OP::InpElTp, CHUNK>
                  (inp_block_offs+seq, N, OP::identInp(), d_inp, shmem_inp);
        copyFromGlb2ShrMem<char, CHUNK>
                  (inp_block_offs+seq, N, 0, d_flg, shmem_flg);

        // 2. each thread sequentially reduces its `CHUNK` elements,
        //    the result is stored in `chunk` array (mapped to registers),
        //    and `tmp` denotes the reduced result.
        FVTup tmp = LiftOP<OP>::identity();
        uint32_t shmem_offset = threadIdx.x * CHUNK;
        #pragma unroll
        for (uint32_t i = 0; i < CHUNK; i++) {
            typename OP::InpElTp elm = shmem_inp[shmem_offset + i];
            char                 flg = shmem_flg[shmem_offset + i];
            FVTup red(flg, OP::mapFun(elm));
            tmp = LiftOP<OP>::apply(tmp, red);
            chunk[i] = tmp;
        }
        __syncthreads();

        // 3. publish in shared memory and perform intra-group scan
        shmem_flg = (volatile char*)(shmem_red + blockDim.x);
        shmem_red[threadIdx.x] = tmp.v;
        shmem_flg[threadIdx.x] = tmp.f;
        __syncthreads();
        tmp = sgmScanIncBlock<OP,char>(shmem_red, shmem_flg, threadIdx.x);
        __syncthreads();
        shmem_red[threadIdx.x] = tmp.v;
        shmem_flg[threadIdx.x] = tmp.f;
        __syncthreads();

        // 4. read the previous element and complete the scan in shared memory
        tmp   = LiftOP<OP>::identity();
        if (threadIdx.x > 0) {
            tmp.v = OP::remVolatile(shmem_red[threadIdx.x-1]);
            tmp.f = shmem_flg[threadIdx.x-1];
        }
        tmp   = LiftOP<OP>::apply(accum, tmp);
        for (uint32_t i = 0; i < CHUNK; i++) {
            chunk[i] = LiftOP<OP>::apply(tmp, chunk[i]);
        }
        tmp.f = shmem_flg[blockDim.x-1];
        tmp.v = OP::remVolatile(shmem_red[blockDim.x-1]);
        accum = LiftOP<OP>::apply(accum, tmp);
        __syncthreads();

        #pragma unroll
        for (uint32_t i = 0; i < CHUNK; i++) {
            shmem_red[threadIdx.x*CHUNK + i] = chunk[i].v;
        }
        __syncthreads();

        // 5. write back to global memory in coalesced form
        copyFromShr2GlbMem<typename OP::RedElTp, CHUNK>
                  (inp_block_offs+seq, N, d_out, shmem_red);
    }
}

#endif // PBB_KERNELS
