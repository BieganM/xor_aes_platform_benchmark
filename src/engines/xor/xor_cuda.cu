#include "xor_cuda.cuh"
#include <iostream>

#ifdef HAS_CUDA
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err)            \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;  \
            throw std::runtime_error(cudaGetErrorString(err));                \
        }                                                                     \
    } while (0)

__global__ void xorKernel(const uint8_t* input, uint8_t* output,
                          const uint8_t* key, size_t keyLen, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] ^ key[idx % keyLen];
    }
}

__global__ void xorKernelVec4(const uint32_t* input, uint32_t* output,
                               const uint8_t* key, size_t keyLen, size_t numWords) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numWords) {
        size_t byteIdx = idx * 4;
        uint32_t in = input[idx];
        uint32_t k = 0;
        for (int i = 0; i < 4; ++i) {
            k |= (static_cast<uint32_t>(key[(byteIdx + i) % keyLen]) << (i * 8));
        }
        output[idx] = in ^ k;
    }
}
#endif

namespace hpc_benchmark {

XorCudaEngine::XorCudaEngine() 
    : d_input_(nullptr), d_output_(nullptr), d_key_(nullptr),
      allocatedSize_(0), allocatedKeySize_(0), initialized_(false) {}

XorCudaEngine::~XorCudaEngine() {
    cleanup();
}

bool XorCudaEngine::isAvailable() const {
#ifdef HAS_CUDA
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    return (err == cudaSuccess && deviceCount > 0);
#else
    return false;
#endif
}

void XorCudaEngine::initialize() {
#ifdef HAS_CUDA
    if (initialized_) return;
    
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        throw std::runtime_error("No CUDA devices found");
    }
    
    CUDA_CHECK(cudaSetDevice(0));
    initialized_ = true;
#endif
}

void XorCudaEngine::cleanup() {
#ifdef HAS_CUDA
    if (d_input_) { cudaFree(d_input_); d_input_ = nullptr; }
    if (d_output_) { cudaFree(d_output_); d_output_ = nullptr; }
    if (d_key_) { cudaFree(d_key_); d_key_ = nullptr; }
    allocatedSize_ = 0;
    allocatedKeySize_ = 0;
    initialized_ = false;
#endif
}

void XorCudaEngine::encrypt(const uint8_t* input, uint8_t* output, 
                             size_t size, const uint8_t* key, size_t keyLen,
                             const uint8_t*) {
#ifdef HAS_CUDA
    if (!initialized_) {
        initialize();
    }
    
    if (size > allocatedSize_) {
        if (d_input_) cudaFree(d_input_);
        if (d_output_) cudaFree(d_output_);
        CUDA_CHECK(cudaMalloc(&d_input_, size));
        CUDA_CHECK(cudaMalloc(&d_output_, size));
        allocatedSize_ = size;
    }
    
    if (keyLen > allocatedKeySize_) {
        if (d_key_) cudaFree(d_key_);
        CUDA_CHECK(cudaMalloc(&d_key_, keyLen));
        allocatedKeySize_ = keyLen;
    }
    
    CUDA_CHECK(cudaMemcpy(d_input_, input, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_key_, key, keyLen, cudaMemcpyHostToDevice));
    
    constexpr int THREADS_PER_BLOCK = 256;
    int numBlocks = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    xorKernel<<<numBlocks, THREADS_PER_BLOCK>>>(d_input_, d_output_, d_key_, keyLen, size);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(output, d_output_, size, cudaMemcpyDeviceToHost));
#else
    for (size_t i = 0; i < size; ++i) {
        output[i] = input[i] ^ key[i % keyLen];
    }
#endif
}

void XorCudaEngine::decrypt(const uint8_t* input, uint8_t* output, 
                             size_t size, const uint8_t* key, size_t keyLen,
                             const uint8_t* iv) {
    encrypt(input, output, size, key, keyLen, iv);
}

}
