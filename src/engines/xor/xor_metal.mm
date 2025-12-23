#include "xor_metal.hpp"
#include <iostream>
#include <stdexcept>
#include <fstream>
#include <sstream>

#ifdef HAS_METAL
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#endif

namespace hpc_benchmark {

#ifdef HAS_METAL
struct XorMetalEngine::Impl {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> commandQueue = nil;
    id<MTLComputePipelineState> pipelineState = nil;
    id<MTLLibrary> library = nil;
    
    ~Impl() {
        pipelineState = nil;
        commandQueue = nil;
        library = nil;
        device = nil;
    }
};

static NSString* getXorShaderSource() {
    return @R"(
#include <metal_stdlib>
using namespace metal;

kernel void xor_encrypt(device const uchar* input [[buffer(0)]],
                        device uchar* output [[buffer(1)]],
                        device const uchar* key [[buffer(2)]],
                        constant uint& keyLen [[buffer(3)]],
                        constant ulong& size [[buffer(4)]],
                        uint idx [[thread_position_in_grid]]) {
    if (idx < size) {
        output[idx] = input[idx] ^ key[idx % keyLen];
    }
}

kernel void xor_encrypt_vec4(device const uint* input [[buffer(0)]],
                              device uint* output [[buffer(1)]],
                              device const uchar* key [[buffer(2)]],
                              constant uint& keyLen [[buffer(3)]],
                              constant ulong& numWords [[buffer(4)]],
                              uint idx [[thread_position_in_grid]]) {
    if (idx < numWords) {
        ulong byteIdx = idx * 4;
        uint in = input[idx];
        uint k = 0;
        for (int i = 0; i < 4; ++i) {
            k |= (uint(key[(byteIdx + i) % keyLen]) << (i * 8));
        }
        output[idx] = in ^ k;
    }
}
)";
}
#else
struct XorMetalEngine::Impl {};
#endif

XorMetalEngine::XorMetalEngine() : impl_(new Impl()), initialized_(false) {}

XorMetalEngine::~XorMetalEngine() {
    cleanup();
    delete impl_;
}

bool XorMetalEngine::isAvailable() const {
#ifdef HAS_METAL
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        return device != nil;
    }
#else
    return false;
#endif
}

void XorMetalEngine::initialize() {
#ifdef HAS_METAL
    if (initialized_) return;
    
    @autoreleasepool {
        impl_->device = MTLCreateSystemDefaultDevice();
        if (!impl_->device) {
            throw std::runtime_error("Failed to create Metal device");
        }
        
        impl_->commandQueue = [impl_->device newCommandQueue];
        if (!impl_->commandQueue) {
            throw std::runtime_error("Failed to create Metal command queue");
        }
        
        NSError* error = nil;
        MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
        
        impl_->library = [impl_->device newLibraryWithSource:getXorShaderSource()
                                                     options:options
                                                       error:&error];
        if (!impl_->library) {
            throw std::runtime_error("Failed to create Metal library: " + 
                std::string([[error localizedDescription] UTF8String]));
        }
        
        id<MTLFunction> function = [impl_->library newFunctionWithName:@"xor_encrypt"];
        if (!function) {
            throw std::runtime_error("Failed to find xor_encrypt function");
        }
        
        impl_->pipelineState = [impl_->device newComputePipelineStateWithFunction:function error:&error];
        if (!impl_->pipelineState) {
            throw std::runtime_error("Failed to create pipeline state: " + 
                std::string([[error localizedDescription] UTF8String]));
        }
        
        initialized_ = true;
    }
#endif
}

void XorMetalEngine::cleanup() {
#ifdef HAS_METAL
    @autoreleasepool {
        impl_->pipelineState = nil;
        impl_->library = nil;
        impl_->commandQueue = nil;
        impl_->device = nil;
    }
    initialized_ = false;
#endif
}

void XorMetalEngine::encrypt(const uint8_t* input, uint8_t* output, 
                              size_t size, const uint8_t* key, size_t keyLen,
                              const uint8_t*) {
#ifdef HAS_METAL
    if (!initialized_) {
        initialize();
    }
    
    @autoreleasepool {
        id<MTLBuffer> inputBuffer = [impl_->device newBufferWithBytes:input
                                                               length:size
                                                              options:MTLResourceStorageModeShared];
        id<MTLBuffer> outputBuffer = [impl_->device newBufferWithLength:size
                                                                options:MTLResourceStorageModeShared];
        id<MTLBuffer> keyBuffer = [impl_->device newBufferWithBytes:key
                                                             length:keyLen
                                                            options:MTLResourceStorageModeShared];
        
        if (!inputBuffer || !outputBuffer || !keyBuffer) {
            inputBuffer = nil;
            outputBuffer = nil;
            keyBuffer = nil;
            throw std::runtime_error("Failed to allocate Metal buffers");
        }
        
        uint32_t keyLenU = static_cast<uint32_t>(keyLen);
        uint64_t sizeU = static_cast<uint64_t>(size);
        
        id<MTLCommandBuffer> commandBuffer = [impl_->commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        
        [encoder setComputePipelineState:impl_->pipelineState];
        [encoder setBuffer:inputBuffer offset:0 atIndex:0];
        [encoder setBuffer:outputBuffer offset:0 atIndex:1];
        [encoder setBuffer:keyBuffer offset:0 atIndex:2];
        [encoder setBytes:&keyLenU length:sizeof(keyLenU) atIndex:3];
        [encoder setBytes:&sizeU length:sizeof(sizeU) atIndex:4];
        
        NSUInteger threadGroupSize = impl_->pipelineState.maxTotalThreadsPerThreadgroup;
        if (threadGroupSize > size) {
            threadGroupSize = size;
        }
        
        MTLSize gridSize = MTLSizeMake(size, 1, 1);
        MTLSize groupSize = MTLSizeMake(threadGroupSize, 1, 1);
        
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:groupSize];
        [encoder endEncoding];
        
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        memcpy(output, [outputBuffer contents], size);
        
        [inputBuffer setPurgeableState:MTLPurgeableStateEmpty];
        [outputBuffer setPurgeableState:MTLPurgeableStateEmpty];
        [keyBuffer setPurgeableState:MTLPurgeableStateEmpty];
        
        inputBuffer = nil;
        outputBuffer = nil;
        keyBuffer = nil;
    }
#else
    for (size_t i = 0; i < size; ++i) {
        output[i] = input[i] ^ key[i % keyLen];
    }
#endif
}

void XorMetalEngine::decrypt(const uint8_t* input, uint8_t* output, 
                              size_t size, const uint8_t* key, size_t keyLen,
                              const uint8_t* iv) {
    encrypt(input, output, size, key, keyLen, iv);
}

}
