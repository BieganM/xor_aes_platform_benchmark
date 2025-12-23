#include "aes_opencl.hpp"
#include "kernels/aes_tables.hpp"
#include <iostream>
#include <stdexcept>
#include <cstring>
#include <array>

#ifdef HAS_OPENCL
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#endif

namespace hpc_benchmark {

#ifdef HAS_OPENCL
struct AesOpenCLEngine::Impl {
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_program program = nullptr;
    cl_kernel kernel = nullptr;
    cl_device_id device = nullptr;
    std::array<uint8_t, 16> defaultIV;
    
    ~Impl() {
        if (kernel) clReleaseKernel(kernel);
        if (program) clReleaseProgram(program);
        if (queue) clReleaseCommandQueue(queue);
        if (context) clReleaseContext(context);
    }
};

static const char* AES_CTR_KERNEL_SOURCE = R"(
__constant uchar SBOX[256] = {
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

uchar xtime(uchar x) {
    return ((x << 1) ^ (((x >> 7) & 1) * 0x1b));
}

void aes_round(uchar* state, __global const uchar* roundKey) {
    uchar tmp[16];
    
    for (int i = 0; i < 16; i++) {
        state[i] = SBOX[state[i]];
    }
    
    tmp[0] = state[0]; tmp[1] = state[5]; tmp[2] = state[10]; tmp[3] = state[15];
    tmp[4] = state[4]; tmp[5] = state[9]; tmp[6] = state[14]; tmp[7] = state[3];
    tmp[8] = state[8]; tmp[9] = state[13]; tmp[10] = state[2]; tmp[11] = state[7];
    tmp[12] = state[12]; tmp[13] = state[1]; tmp[14] = state[6]; tmp[15] = state[11];
    
    for (int i = 0; i < 16; i++) state[i] = tmp[i];
    
    for (int i = 0; i < 16; i += 4) {
        uchar a = state[i], b = state[i+1], c = state[i+2], d = state[i+3];
        uchar e = a ^ b ^ c ^ d;
        tmp[i] = a ^ e ^ xtime(a ^ b);
        tmp[i+1] = b ^ e ^ xtime(b ^ c);
        tmp[i+2] = c ^ e ^ xtime(c ^ d);
        tmp[i+3] = d ^ e ^ xtime(d ^ a);
    }
    
    for (int i = 0; i < 16; i++) state[i] = tmp[i] ^ roundKey[i];
}

void aes_final_round(uchar* state, __global const uchar* roundKey) {
    uchar tmp[16];
    
    for (int i = 0; i < 16; i++) {
        state[i] = SBOX[state[i]];
    }
    
    tmp[0] = state[0]; tmp[1] = state[5]; tmp[2] = state[10]; tmp[3] = state[15];
    tmp[4] = state[4]; tmp[5] = state[9]; tmp[6] = state[14]; tmp[7] = state[3];
    tmp[8] = state[8]; tmp[9] = state[13]; tmp[10] = state[2]; tmp[11] = state[7];
    tmp[12] = state[12]; tmp[13] = state[1]; tmp[14] = state[6]; tmp[15] = state[11];
    
    for (int i = 0; i < 16; i++) state[i] = tmp[i] ^ roundKey[i];
}

void aes256_encrypt_block(uchar* block, __global const uint* roundKeys) {
    __global const uchar* rk = (__global const uchar*)roundKeys;
    
    for (int i = 0; i < 16; i++) block[i] ^= rk[i];
    
    for (int round = 1; round < 14; round++) {
        aes_round(block, rk + round * 16);
    }
    
    aes_final_round(block, rk + 14 * 16);
}

__kernel void aes_ctr_encrypt(__global const uchar* input,
                               __global uchar* output,
                               __global const uint* roundKeys,
                               __global const uchar* iv,
                               const ulong numBlocks) {
    size_t blockIdx = get_global_id(0);
    if (blockIdx >= numBlocks) return;
    
    uchar counter[16];
    for (int i = 0; i < 16; i++) counter[i] = iv[i];
    
    ulong ctr = blockIdx;
    for (int i = 15; i >= 8 && ctr > 0; i--) {
        ulong sum = counter[i] + (ctr & 0xFF);
        counter[i] = (uchar)(sum & 0xFF);
        ctr = (ctr >> 8) + (sum >> 8);
    }
    
    aes256_encrypt_block(counter, roundKeys);
    
    size_t offset = blockIdx * 16;
    for (int i = 0; i < 16; i++) {
        output[offset + i] = input[offset + i] ^ counter[i];
    }
}
)";
#else
struct AesOpenCLEngine::Impl {};
#endif

AesOpenCLEngine::AesOpenCLEngine() : impl_(new Impl()), initialized_(false) {
#ifdef HAS_OPENCL
    for (int i = 0; i < 16; i++) impl_->defaultIV[i] = static_cast<uint8_t>(i);
#endif
}

AesOpenCLEngine::~AesOpenCLEngine() {
    cleanup();
    delete impl_;
}

bool AesOpenCLEngine::isAvailable() const {
#ifdef HAS_OPENCL
    cl_uint numPlatforms = 0;
    cl_int err = clGetPlatformIDs(0, nullptr, &numPlatforms);
    return (err == CL_SUCCESS && numPlatforms > 0);
#else
    return false;
#endif
}

void AesOpenCLEngine::initialize() {
#ifdef HAS_OPENCL
    if (initialized_) return;
    
    cl_int err;
    cl_platform_id platform;
    
    err = clGetPlatformIDs(1, &platform, nullptr);
    if (err != CL_SUCCESS) throw std::runtime_error("No OpenCL platforms found");
    
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &impl_->device, nullptr);
    if (err != CL_SUCCESS) {
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &impl_->device, nullptr);
    }
    if (err != CL_SUCCESS) throw std::runtime_error("No OpenCL devices found");
    
    impl_->context = clCreateContext(nullptr, 1, &impl_->device, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to create OpenCL context");
    
    impl_->queue = clCreateCommandQueue(impl_->context, impl_->device, 0, &err);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to create command queue");
    
    const char* source = AES_CTR_KERNEL_SOURCE;
    size_t sourceLen = strlen(source);
    impl_->program = clCreateProgramWithSource(impl_->context, 1, &source, &sourceLen, &err);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to create program");
    
    err = clBuildProgram(impl_->program, 1, &impl_->device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        char log[8192];
        clGetProgramBuildInfo(impl_->program, impl_->device, CL_PROGRAM_BUILD_LOG, sizeof(log), log, nullptr);
        throw std::runtime_error(std::string("Build failed: ") + log);
    }
    
    impl_->kernel = clCreateKernel(impl_->program, "aes_ctr_encrypt", &err);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to create kernel");
    
    initialized_ = true;
#endif
}

void AesOpenCLEngine::cleanup() {
#ifdef HAS_OPENCL
    if (impl_->kernel) { clReleaseKernel(impl_->kernel); impl_->kernel = nullptr; }
    if (impl_->program) { clReleaseProgram(impl_->program); impl_->program = nullptr; }
    if (impl_->queue) { clReleaseCommandQueue(impl_->queue); impl_->queue = nullptr; }
    if (impl_->context) { clReleaseContext(impl_->context); impl_->context = nullptr; }
    initialized_ = false;
#endif
}

void AesOpenCLEngine::encrypt(const uint8_t* input, uint8_t* output, 
                               size_t size, const uint8_t* key, size_t keyLen,
                               const uint8_t* iv) {
#ifdef HAS_OPENCL
    if (keyLen != 32) throw std::runtime_error("AES-256 requires 32-byte key");
    if (!initialized_) initialize();
    
    const uint8_t* actualIV = iv ? iv : impl_->defaultIV.data();
    
    uint32_t roundKeys[60];
    aes::keyExpansion256(key, roundKeys);
    
    size_t paddedSize = ((size + 15) / 16) * 16;
    size_t numBlocks = paddedSize / 16;
    
    std::vector<uint8_t> paddedInput(paddedSize, 0);
    std::memcpy(paddedInput.data(), input, size);
    
    cl_int err;
    cl_mem inputBuf = clCreateBuffer(impl_->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      paddedSize, paddedInput.data(), &err);
    cl_mem outputBuf = clCreateBuffer(impl_->context, CL_MEM_WRITE_ONLY, paddedSize, nullptr, &err);
    cl_mem keyBuf = clCreateBuffer(impl_->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    sizeof(roundKeys), roundKeys, &err);
    cl_mem ivBuf = clCreateBuffer(impl_->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   16, const_cast<uint8_t*>(actualIV), &err);
    
    cl_ulong numBlocksU = static_cast<cl_ulong>(numBlocks);
    
    clSetKernelArg(impl_->kernel, 0, sizeof(cl_mem), &inputBuf);
    clSetKernelArg(impl_->kernel, 1, sizeof(cl_mem), &outputBuf);
    clSetKernelArg(impl_->kernel, 2, sizeof(cl_mem), &keyBuf);
    clSetKernelArg(impl_->kernel, 3, sizeof(cl_mem), &ivBuf);
    clSetKernelArg(impl_->kernel, 4, sizeof(cl_ulong), &numBlocksU);
    
    size_t globalSize = numBlocks;
    size_t localSize = 256;
    globalSize = ((globalSize + localSize - 1) / localSize) * localSize;
    
    clEnqueueNDRangeKernel(impl_->queue, impl_->kernel, 1, nullptr, 
                            &globalSize, &localSize, 0, nullptr, nullptr);
    
    clEnqueueReadBuffer(impl_->queue, outputBuf, CL_TRUE, 0, size, output, 0, nullptr, nullptr);
    
    clReleaseMemObject(inputBuf);
    clReleaseMemObject(outputBuf);
    clReleaseMemObject(keyBuf);
    clReleaseMemObject(ivBuf);
#else
    throw std::runtime_error("OpenCL not available");
#endif
}

void AesOpenCLEngine::decrypt(const uint8_t* input, uint8_t* output, 
                               size_t size, const uint8_t* key, size_t keyLen,
                               const uint8_t* iv) {
    encrypt(input, output, size, key, keyLen, iv);
}

}
