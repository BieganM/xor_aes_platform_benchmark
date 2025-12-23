#include "aes_metal.hpp"
#include "kernels/aes_tables.hpp"
#include <iostream>
#include <stdexcept>
#include <cstring>

#ifdef HAS_METAL
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#endif

namespace hpc_benchmark {

#ifdef HAS_METAL
struct AesMetalEngine::Impl {
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

static NSString* getAesShaderSource() {
    return @R"(
#include <metal_stdlib>
using namespace metal;

constant uchar SBOX[256] = {
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

inline uchar xtime(uchar x) {
    return ((x << 1) ^ (((x >> 7) & 1) * 0x1b));
}

void aes_round(thread uchar* state, device const uchar* roundKey) {
    uchar tmp[16];
    
    for (int i = 0; i < 16; i++) state[i] = SBOX[state[i]];
    
    tmp[0] = state[0]; tmp[1] = state[5]; tmp[2] = state[10]; tmp[3] = state[15];
    tmp[4] = state[4]; tmp[5] = state[9]; tmp[6] = state[14]; tmp[7] = state[3];
    tmp[8] = state[8]; tmp[9] = state[13]; tmp[10] = state[2]; tmp[11] = state[7];
    tmp[12] = state[12]; tmp[13] = state[1]; tmp[14] = state[6]; tmp[15] = state[11];
    
    for (int i = 0; i < 16; i += 4) {
        uchar a = tmp[i], b = tmp[i+1], c = tmp[i+2], d = tmp[i+3];
        uchar e = a ^ b ^ c ^ d;
        state[i] = a ^ e ^ xtime(a ^ b) ^ roundKey[i];
        state[i+1] = b ^ e ^ xtime(b ^ c) ^ roundKey[i+1];
        state[i+2] = c ^ e ^ xtime(c ^ d) ^ roundKey[i+2];
        state[i+3] = d ^ e ^ xtime(d ^ a) ^ roundKey[i+3];
    }
}

void aes_final_round(thread uchar* state, device const uchar* roundKey) {
    uchar tmp[16];
    
    for (int i = 0; i < 16; i++) state[i] = SBOX[state[i]];
    
    tmp[0] = state[0]; tmp[1] = state[5]; tmp[2] = state[10]; tmp[3] = state[15];
    tmp[4] = state[4]; tmp[5] = state[9]; tmp[6] = state[14]; tmp[7] = state[3];
    tmp[8] = state[8]; tmp[9] = state[13]; tmp[10] = state[2]; tmp[11] = state[7];
    tmp[12] = state[12]; tmp[13] = state[1]; tmp[14] = state[6]; tmp[15] = state[11];
    
    for (int i = 0; i < 16; i++) state[i] = tmp[i] ^ roundKey[i];
}

kernel void aes_ctr_encrypt(device const uchar4* input [[buffer(0)]],
                            device uchar4* output [[buffer(1)]],
                            device const uchar* roundKeys [[buffer(2)]],
                            device const uchar* iv [[buffer(3)]],
                            constant ulong& numBlocks [[buffer(4)]],
                            uint blockIdx [[thread_position_in_grid]]) {
    if (blockIdx >= numBlocks) return;
    
    uchar counter[16];
    for (int i = 0; i < 16; i++) counter[i] = iv[i];
    
    ulong ctr = blockIdx;
    for (int i = 15; i >= 8 && ctr > 0; i--) {
        ulong sum = counter[i] + (ctr & 0xFF);
        counter[i] = (uchar)(sum & 0xFF);
        ctr = (ctr >> 8) + (sum >> 8);
    }
    
    for (int i = 0; i < 16; i++) counter[i] ^= roundKeys[i];
    
    for (int round = 1; round < 14; round++) {
        aes_round(counter, roundKeys + round * 16);
    }
    aes_final_round(counter, roundKeys + 14 * 16);
    
    ulong offset = blockIdx * 4;
    uchar4 in0 = input[offset];
    uchar4 in1 = input[offset + 1];
    uchar4 in2 = input[offset + 2];
    uchar4 in3 = input[offset + 3];
    
    output[offset] = uchar4(in0.x ^ counter[0], in0.y ^ counter[1], 
                            in0.z ^ counter[2], in0.w ^ counter[3]);
    output[offset + 1] = uchar4(in1.x ^ counter[4], in1.y ^ counter[5],
                                in1.z ^ counter[6], in1.w ^ counter[7]);
    output[offset + 2] = uchar4(in2.x ^ counter[8], in2.y ^ counter[9],
                                in2.z ^ counter[10], in2.w ^ counter[11]);
    output[offset + 3] = uchar4(in3.x ^ counter[12], in3.y ^ counter[13],
                                in3.z ^ counter[14], in3.w ^ counter[15]);
}
)";
}
#else
struct AesMetalEngine::Impl {};
#endif

AesMetalEngine::AesMetalEngine() : impl_(new Impl()), initialized_(false) {
    for (int i = 0; i < 16; i++) defaultIV_[i] = static_cast<uint8_t>(i);
}

AesMetalEngine::~AesMetalEngine() {
    cleanup();
    delete impl_;
}

bool AesMetalEngine::isAvailable() const {
#ifdef HAS_METAL
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        return device != nil;
    }
#else
    return false;
#endif
}

void AesMetalEngine::initialize() {
#ifdef HAS_METAL
    if (initialized_) return;
    
    @autoreleasepool {
        impl_->device = MTLCreateSystemDefaultDevice();
        if (!impl_->device) throw std::runtime_error("Failed to create Metal device");
        
        impl_->commandQueue = [impl_->device newCommandQueue];
        if (!impl_->commandQueue) throw std::runtime_error("Failed to create command queue");
        
        NSError* error = nil;
        MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
        
        impl_->library = [impl_->device newLibraryWithSource:getAesShaderSource()
                                                     options:options
                                                       error:&error];
        if (!impl_->library) {
            throw std::runtime_error("Failed to create Metal library: " + 
                std::string([[error localizedDescription] UTF8String]));
        }
        
        id<MTLFunction> function = [impl_->library newFunctionWithName:@"aes_ctr_encrypt"];
        if (!function) throw std::runtime_error("Failed to find aes_ctr_encrypt function");
        
        impl_->pipelineState = [impl_->device newComputePipelineStateWithFunction:function error:&error];
        if (!impl_->pipelineState) {
            throw std::runtime_error("Failed to create pipeline state");
        }
        
        initialized_ = true;
    }
#endif
}

void AesMetalEngine::cleanup() {
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

void AesMetalEngine::encrypt(const uint8_t* input, uint8_t* output, 
                              size_t size, const uint8_t* key, size_t keyLen,
                              const uint8_t* iv) {
#ifdef HAS_METAL
    if (keyLen != 32) throw std::runtime_error("AES-256 requires 32-byte key");
    if (!initialized_) initialize();
    
    const uint8_t* actualIV = iv ? iv : defaultIV_.data();
    
    uint32_t roundKeysU32[60];
    aes::keyExpansion256(key, roundKeysU32);
    
    uint8_t roundKeys[240];
    for (int i = 0; i < 60; i++) {
        roundKeys[i*4]   = (roundKeysU32[i] >> 0) & 0xFF;
        roundKeys[i*4+1] = (roundKeysU32[i] >> 8) & 0xFF;
        roundKeys[i*4+2] = (roundKeysU32[i] >> 16) & 0xFF;
        roundKeys[i*4+3] = (roundKeysU32[i] >> 24) & 0xFF;
    }
    
    @autoreleasepool {
        size_t paddedSize = ((size + 15) / 16) * 16;
        size_t numBlocks = paddedSize / 16;
        
        std::vector<uint8_t> paddedInput(paddedSize, 0);
        std::memcpy(paddedInput.data(), input, size);
        
        id<MTLBuffer> inputBuffer = [impl_->device newBufferWithBytes:paddedInput.data()
                                                               length:paddedSize
                                                              options:MTLResourceStorageModeShared];
        id<MTLBuffer> outputBuffer = [impl_->device newBufferWithLength:paddedSize
                                                                options:MTLResourceStorageModeShared];
        id<MTLBuffer> keyBuffer = [impl_->device newBufferWithBytes:roundKeys
                                                             length:sizeof(roundKeys)
                                                            options:MTLResourceStorageModeShared];
        id<MTLBuffer> ivBuffer = [impl_->device newBufferWithBytes:actualIV
                                                            length:16
                                                           options:MTLResourceStorageModeShared];
        
        uint64_t numBlocksU = static_cast<uint64_t>(numBlocks);
        
        id<MTLCommandBuffer> commandBuffer = [impl_->commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        
        [encoder setComputePipelineState:impl_->pipelineState];
        [encoder setBuffer:inputBuffer offset:0 atIndex:0];
        [encoder setBuffer:outputBuffer offset:0 atIndex:1];
        [encoder setBuffer:keyBuffer offset:0 atIndex:2];
        [encoder setBuffer:ivBuffer offset:0 atIndex:3];
        [encoder setBytes:&numBlocksU length:sizeof(numBlocksU) atIndex:4];
        
        NSUInteger threadGroupSize = impl_->pipelineState.maxTotalThreadsPerThreadgroup;
        if (threadGroupSize > numBlocks) threadGroupSize = numBlocks;
        if (threadGroupSize == 0) threadGroupSize = 1;
        
        MTLSize gridSize = MTLSizeMake(numBlocks, 1, 1);
        MTLSize groupSize = MTLSizeMake(threadGroupSize, 1, 1);
        
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:groupSize];
        [encoder endEncoding];
        
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        std::memcpy(output, [outputBuffer contents], size);
        
        [inputBuffer setPurgeableState:MTLPurgeableStateEmpty];
        [outputBuffer setPurgeableState:MTLPurgeableStateEmpty];
        [keyBuffer setPurgeableState:MTLPurgeableStateEmpty];
        [ivBuffer setPurgeableState:MTLPurgeableStateEmpty];
        
        inputBuffer = nil;
        outputBuffer = nil;
        keyBuffer = nil;
        ivBuffer = nil;
    }
#else
    throw std::runtime_error("Metal not available");
#endif
}

void AesMetalEngine::decrypt(const uint8_t* input, uint8_t* output, 
                              size_t size, const uint8_t* key, size_t keyLen,
                              const uint8_t* iv) {
    encrypt(input, output, size, key, keyLen, iv);
}

}
