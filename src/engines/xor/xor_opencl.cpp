#include "xor_opencl.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>

#ifdef HAS_OPENCL
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#include <cstring>
#endif

namespace hpc_benchmark {

#ifdef HAS_OPENCL
struct XorOpenCLEngine::Impl {
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_program program = nullptr;
    cl_kernel kernel = nullptr;
    cl_device_id device = nullptr;
    
    ~Impl() {
        if (kernel) clReleaseKernel(kernel);
        if (program) clReleaseProgram(program);
        if (queue) clReleaseCommandQueue(queue);
        if (context) clReleaseContext(context);
    }
};

static const char* XOR_KERNEL_SOURCE = R"(
__kernel void xor_encrypt(__global const uchar* input,
                          __global uchar* output,
                          __global const uchar* key,
                          const uint keyLen,
                          const ulong size) {
    size_t idx = get_global_id(0);
    if (idx < size) {
        output[idx] = input[idx] ^ key[idx % keyLen];
    }
}
)";
#else
struct XorOpenCLEngine::Impl {};
#endif

XorOpenCLEngine::XorOpenCLEngine() : impl_(new Impl()), initialized_(false) {}

XorOpenCLEngine::~XorOpenCLEngine() {
    cleanup();
    delete impl_;
}

bool XorOpenCLEngine::isAvailable() const {
#ifdef HAS_OPENCL
    cl_uint numPlatforms = 0;
    cl_int err = clGetPlatformIDs(0, nullptr, &numPlatforms);
    return (err == CL_SUCCESS && numPlatforms > 0);
#else
    return false;
#endif
}

void XorOpenCLEngine::initialize() {
#ifdef HAS_OPENCL
    if (initialized_) return;
    
    cl_int err;
    cl_platform_id platform;
    cl_uint numPlatforms;
    
    err = clGetPlatformIDs(1, &platform, &numPlatforms);
    if (err != CL_SUCCESS || numPlatforms == 0) {
        throw std::runtime_error("No OpenCL platforms found");
    }
    
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &impl_->device, nullptr);
    if (err != CL_SUCCESS) {
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &impl_->device, nullptr);
    }
    if (err != CL_SUCCESS) {
        throw std::runtime_error("No OpenCL devices found");
    }
    
    impl_->context = clCreateContext(nullptr, 1, &impl_->device, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to create OpenCL context");
    }
    
    impl_->queue = clCreateCommandQueue(impl_->context, impl_->device, 0, &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to create OpenCL command queue");
    }
    
    const char* source = XOR_KERNEL_SOURCE;
    size_t sourceLen = strlen(source);
    impl_->program = clCreateProgramWithSource(impl_->context, 1, &source, &sourceLen, &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to create OpenCL program");
    }
    
    err = clBuildProgram(impl_->program, 1, &impl_->device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        char log[4096];
        clGetProgramBuildInfo(impl_->program, impl_->device, CL_PROGRAM_BUILD_LOG, sizeof(log), log, nullptr);
        throw std::runtime_error(std::string("Failed to build OpenCL program: ") + log);
    }
    
    impl_->kernel = clCreateKernel(impl_->program, "xor_encrypt", &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to create OpenCL kernel");
    }
    
    initialized_ = true;
#endif
}

void XorOpenCLEngine::cleanup() {
#ifdef HAS_OPENCL
    if (impl_->kernel) { clReleaseKernel(impl_->kernel); impl_->kernel = nullptr; }
    if (impl_->program) { clReleaseProgram(impl_->program); impl_->program = nullptr; }
    if (impl_->queue) { clReleaseCommandQueue(impl_->queue); impl_->queue = nullptr; }
    if (impl_->context) { clReleaseContext(impl_->context); impl_->context = nullptr; }
    initialized_ = false;
#endif
}

void XorOpenCLEngine::encrypt(const uint8_t* input, uint8_t* output, 
                               size_t size, const uint8_t* key, size_t keyLen,
                               const uint8_t*) {
#ifdef HAS_OPENCL
    if (!initialized_) {
        initialize();
    }
    
    cl_int err;
    
    cl_mem inputBuf = clCreateBuffer(impl_->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      size, const_cast<uint8_t*>(input), &err);
    cl_mem outputBuf = clCreateBuffer(impl_->context, CL_MEM_WRITE_ONLY, size, nullptr, &err);
    cl_mem keyBuf = clCreateBuffer(impl_->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    keyLen, const_cast<uint8_t*>(key), &err);
    
    cl_uint keyLenU = static_cast<cl_uint>(keyLen);
    cl_ulong sizeU = static_cast<cl_ulong>(size);
    
    clSetKernelArg(impl_->kernel, 0, sizeof(cl_mem), &inputBuf);
    clSetKernelArg(impl_->kernel, 1, sizeof(cl_mem), &outputBuf);
    clSetKernelArg(impl_->kernel, 2, sizeof(cl_mem), &keyBuf);
    clSetKernelArg(impl_->kernel, 3, sizeof(cl_uint), &keyLenU);
    clSetKernelArg(impl_->kernel, 4, sizeof(cl_ulong), &sizeU);
    
    size_t globalSize = size;
    size_t localSize = 256;
    globalSize = ((globalSize + localSize - 1) / localSize) * localSize;
    
    err = clEnqueueNDRangeKernel(impl_->queue, impl_->kernel, 1, nullptr, 
                                  &globalSize, &localSize, 0, nullptr, nullptr);
    
    clEnqueueReadBuffer(impl_->queue, outputBuf, CL_TRUE, 0, size, output, 0, nullptr, nullptr);
    
    clReleaseMemObject(inputBuf);
    clReleaseMemObject(outputBuf);
    clReleaseMemObject(keyBuf);
#else
    for (size_t i = 0; i < size; ++i) {
        output[i] = input[i] ^ key[i % keyLen];
    }
#endif
}

void XorOpenCLEngine::decrypt(const uint8_t* input, uint8_t* output, 
                               size_t size, const uint8_t* key, size_t keyLen,
                               const uint8_t* iv) {
    encrypt(input, output, size, key, keyLen, iv);
}

}
