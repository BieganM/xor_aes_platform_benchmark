#include "aes_cuda.cuh"
#include "kernels/aes_tables.hpp"
#include <iostream>
#include <stdexcept>
#include <cstring>

#ifdef HAS_CUDA
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA Error: ") + cudaGetErrorString(err)); \
        } \
    } while (0)

__constant__ uint8_t d_SBOX[256] = {
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
};

__device__ uint8_t d_xtime(uint8_t x) {
    return ((x << 1) ^ (((x >> 7) & 1) * 0x1b));
}

__device__ void d_aes_round(uint8_t* state, const uint8_t* roundKey) {
    uint8_t tmp[16];
    
    for (int i = 0; i < 16; i++) state[i] = d_SBOX[state[i]];
    
    tmp[0] = state[0]; tmp[1] = state[5]; tmp[2] = state[10]; tmp[3] = state[15];
    tmp[4] = state[4]; tmp[5] = state[9]; tmp[6] = state[14]; tmp[7] = state[3];
    tmp[8] = state[8]; tmp[9] = state[13]; tmp[10] = state[2]; tmp[11] = state[7];
    tmp[12] = state[12]; tmp[13] = state[1]; tmp[14] = state[6]; tmp[15] = state[11];
    
    for (int i = 0; i < 16; i++) state[i] = tmp[i];
    
    for (int i = 0; i < 16; i += 4) {
        uint8_t a = state[i], b = state[i+1], c = state[i+2], d = state[i+3];
        uint8_t e = a ^ b ^ c ^ d;
        tmp[i] = a ^ e ^ d_xtime(a ^ b);
        tmp[i+1] = b ^ e ^ d_xtime(b ^ c);
        tmp[i+2] = c ^ e ^ d_xtime(c ^ d);
        tmp[i+3] = d ^ e ^ d_xtime(d ^ a);
    }
    
    for (int i = 0; i < 16; i++) state[i] = tmp[i] ^ roundKey[i];
}

__device__ void d_aes_final_round(uint8_t* state, const uint8_t* roundKey) {
    uint8_t tmp[16];
    
    for (int i = 0; i < 16; i++) state[i] = d_SBOX[state[i]];
    
    tmp[0] = state[0]; tmp[1] = state[5]; tmp[2] = state[10]; tmp[3] = state[15];
    tmp[4] = state[4]; tmp[5] = state[9]; tmp[6] = state[14]; tmp[7] = state[3];
    tmp[8] = state[8]; tmp[9] = state[13]; tmp[10] = state[2]; tmp[11] = state[7];
    tmp[12] = state[12]; tmp[13] = state[1]; tmp[14] = state[6]; tmp[15] = state[11];
    
    for (int i = 0; i < 16; i++) state[i] = tmp[i] ^ roundKey[i];
}

__device__ void d_aes256_encrypt_block(uint8_t* block, const uint32_t* roundKeys) {
    const uint8_t* rk = reinterpret_cast<const uint8_t*>(roundKeys);
    
    for (int i = 0; i < 16; i++) block[i] ^= rk[i];
    
    for (int round = 1; round < 14; round++) {
        d_aes_round(block, rk + round * 16);
    }
    
    d_aes_final_round(block, rk + 14 * 16);
}

__global__ void aesCtrEncryptKernel(const uint8_t* input, uint8_t* output,
                                     const uint32_t* roundKeys, const uint8_t* iv,
                                     size_t numBlocks) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numBlocks) return;
    
    uint8_t counter[16];
    for (int i = 0; i < 16; i++) counter[i] = iv[i];
    
    uint64_t ctr = idx;
    for (int i = 15; i >= 8 && ctr > 0; i--) {
        uint64_t sum = counter[i] + (ctr & 0xFF);
        counter[i] = static_cast<uint8_t>(sum & 0xFF);
        ctr = (ctr >> 8) + (sum >> 8);
    }
    
    d_aes256_encrypt_block(counter, roundKeys);
    
    size_t offset = idx * 16;
    for (int i = 0; i < 16; i++) {
        output[offset + i] = input[offset + i] ^ counter[i];
    }
}
#endif

namespace hpc_benchmark {

AesCudaEngine::AesCudaEngine() 
    : d_input_(nullptr), d_output_(nullptr), d_roundKeys_(nullptr), d_iv_(nullptr),
      allocatedSize_(0), initialized_(false) {
    for (int i = 0; i < 16; i++) defaultIV_[i] = static_cast<uint8_t>(i);
}

AesCudaEngine::~AesCudaEngine() {
    cleanup();
}

bool AesCudaEngine::isAvailable() const {
#ifdef HAS_CUDA
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    return (err == cudaSuccess && deviceCount > 0);
#else
    return false;
#endif
}

void AesCudaEngine::initialize() {
#ifdef HAS_CUDA
    if (initialized_) return;
    
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) throw std::runtime_error("No CUDA devices");
    
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaMalloc(&d_roundKeys_, 60 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_iv_, 16));
    
    initialized_ = true;
#endif
}

void AesCudaEngine::cleanup() {
#ifdef HAS_CUDA
    if (d_input_) { cudaFree(d_input_); d_input_ = nullptr; }
    if (d_output_) { cudaFree(d_output_); d_output_ = nullptr; }
    if (d_roundKeys_) { cudaFree(d_roundKeys_); d_roundKeys_ = nullptr; }
    if (d_iv_) { cudaFree(d_iv_); d_iv_ = nullptr; }
    allocatedSize_ = 0;
    initialized_ = false;
#endif
}

void AesCudaEngine::encrypt(const uint8_t* input, uint8_t* output, 
                             size_t size, const uint8_t* key, size_t keyLen,
                             const uint8_t* iv) {
#ifdef HAS_CUDA
    if (keyLen != 32) throw std::runtime_error("AES-256 requires 32-byte key");
    if (!initialized_) initialize();
    
    const uint8_t* actualIV = iv ? iv : defaultIV_.data();
    
    uint32_t roundKeys[60];
    aes::keyExpansion256(key, roundKeys);
    
    size_t paddedSize = ((size + 15) / 16) * 16;
    size_t numBlocks = paddedSize / 16;
    
    if (paddedSize > allocatedSize_) {
        if (d_input_) cudaFree(d_input_);
        if (d_output_) cudaFree(d_output_);
        CUDA_CHECK(cudaMalloc(&d_input_, paddedSize));
        CUDA_CHECK(cudaMalloc(&d_output_, paddedSize));
        allocatedSize_ = paddedSize;
    }
    
    CUDA_CHECK(cudaMemcpy(d_input_, input, size, cudaMemcpyHostToDevice));
    if (paddedSize > size) {
        CUDA_CHECK(cudaMemset(d_input_ + size, 0, paddedSize - size));
    }
    CUDA_CHECK(cudaMemcpy(d_roundKeys_, roundKeys, sizeof(roundKeys), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_iv_, actualIV, 16, cudaMemcpyHostToDevice));
    
    constexpr int THREADS_PER_BLOCK = 256;
    int numCudaBlocks = (numBlocks + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    aesCtrEncryptKernel<<<numCudaBlocks, THREADS_PER_BLOCK>>>(
        d_input_, d_output_, d_roundKeys_, d_iv_, numBlocks);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(output, d_output_, size, cudaMemcpyDeviceToHost));
#else
    throw std::runtime_error("CUDA not available");
#endif
}

void AesCudaEngine::decrypt(const uint8_t* input, uint8_t* output, 
                             size_t size, const uint8_t* key, size_t keyLen,
                             const uint8_t* iv) {
    encrypt(input, output, size, key, keyLen, iv);
}

}
