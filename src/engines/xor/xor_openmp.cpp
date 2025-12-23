#include "xor_openmp.hpp"

#ifdef HAS_OPENMP
#include <omp.h>
#endif

namespace hpc_benchmark {

bool XorOpenMPEngine::isAvailable() const {
#ifdef HAS_OPENMP
    return true;
#else
    return false;
#endif
}

void XorOpenMPEngine::encrypt(const uint8_t* input, uint8_t* output, 
                               size_t size, const uint8_t* key, size_t keyLen,
                               const uint8_t*) {
#ifdef HAS_OPENMP
    if (numThreads_ > 0) {
        omp_set_num_threads(numThreads_);
    }
    
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < size; ++i) {
        output[i] = input[i] ^ key[i % keyLen];
    }
#else
    for (size_t i = 0; i < size; ++i) {
        output[i] = input[i] ^ key[i % keyLen];
    }
#endif
}

void XorOpenMPEngine::decrypt(const uint8_t* input, uint8_t* output, 
                               size_t size, const uint8_t* key, size_t keyLen,
                               const uint8_t* iv) {
    encrypt(input, output, size, key, keyLen, iv);
}

}
