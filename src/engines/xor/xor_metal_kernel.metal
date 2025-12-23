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
