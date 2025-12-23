#include "xor_sequential.hpp"

namespace hpc_benchmark {

void XorSequentialEngine::encrypt(const uint8_t* input, uint8_t* output, 
                                   size_t size, const uint8_t* key, size_t keyLen,
                                   const uint8_t*) {
    for (size_t i = 0; i < size; ++i) {
        output[i] = input[i] ^ key[i % keyLen];
    }
}

void XorSequentialEngine::decrypt(const uint8_t* input, uint8_t* output, 
                                   size_t size, const uint8_t* key, size_t keyLen,
                                   const uint8_t* iv) {
    encrypt(input, output, size, key, keyLen, iv);
}

}
