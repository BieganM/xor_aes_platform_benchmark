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

__kernel void xor_encrypt_vec4(__global const uchar4* input,
                               __global uchar4* output,
                               __global const uchar* key,
                               const uint keyLen,
                               const ulong numVec4) {
    size_t idx = get_global_id(0);
    if (idx < numVec4) {
        size_t byteIdx = idx * 4;
        uchar4 in = input[idx];
        uchar4 k;
        k.x = key[(byteIdx + 0) % keyLen];
        k.y = key[(byteIdx + 1) % keyLen];
        k.z = key[(byteIdx + 2) % keyLen];
        k.w = key[(byteIdx + 3) % keyLen];
        output[idx] = in ^ k;
    }
}
