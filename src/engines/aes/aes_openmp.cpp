#include "aes_openmp.hpp"
#include <openssl/evp.h>
#include <openssl/rand.h>
#include <stdexcept>
#include <cstring>
#include <vector>

#ifdef HAS_OPENMP
#include <omp.h>
#endif

namespace hpc_benchmark {

AesOpenMPEngine::AesOpenMPEngine() {
    RAND_bytes(defaultIV_.data(), 16);
}

AesOpenMPEngine::~AesOpenMPEngine() {}

bool AesOpenMPEngine::isAvailable() const {
#ifdef HAS_OPENMP
    return true;
#else
    return false;
#endif
}

void AesOpenMPEngine::encrypt(const uint8_t* input, uint8_t* output, 
                               size_t size, const uint8_t* key, size_t keyLen,
                               const uint8_t* iv) {
    if (keyLen != 32) {
        throw std::runtime_error("AES-256 requires 32-byte key");
    }
    
    const uint8_t* actualIV = iv ? iv : defaultIV_.data();
    
#ifdef HAS_OPENMP
    if (numThreads_ > 0) {
        omp_set_num_threads(numThreads_);
    }
    
    constexpr size_t BLOCK_SIZE = 16;
    constexpr size_t CHUNK_SIZE = 1024 * 1024;
    
    size_t numChunks = (size + CHUNK_SIZE - 1) / CHUNK_SIZE;
    
    #pragma omp parallel
    {
        EVP_CIPHER_CTX* ctx = EVP_CIPHER_CTX_new();
        
        #pragma omp for schedule(dynamic)
        for (size_t chunkIdx = 0; chunkIdx < numChunks; ++chunkIdx) {
            size_t offset = chunkIdx * CHUNK_SIZE;
            size_t chunkLen = std::min(CHUNK_SIZE, size - offset);
            
            std::array<uint8_t, 16> chunkIV;
            std::memcpy(chunkIV.data(), actualIV, 16);
            
            uint64_t blockOffset = offset / BLOCK_SIZE;
            for (int i = 15; i >= 8 && blockOffset > 0; --i) {
                chunkIV[i] = (chunkIV[i] + (blockOffset & 0xff)) & 0xff;
                blockOffset >>= 8;
            }
            
            EVP_CIPHER_CTX_reset(ctx);
            EVP_EncryptInit_ex(ctx, EVP_aes_256_ctr(), nullptr, key, chunkIV.data());
            
            int outLen = 0;
            EVP_EncryptUpdate(ctx, output + offset, &outLen, input + offset, static_cast<int>(chunkLen));
            EVP_EncryptFinal_ex(ctx, output + offset + outLen, &outLen);
        }
        
        EVP_CIPHER_CTX_free(ctx);
    }
#else
    EVP_CIPHER_CTX* ctx = EVP_CIPHER_CTX_new();
    EVP_EncryptInit_ex(ctx, EVP_aes_256_ctr(), nullptr, key, actualIV);
    
    int outLen = 0;
    EVP_EncryptUpdate(ctx, output, &outLen, input, static_cast<int>(size));
    EVP_EncryptFinal_ex(ctx, output + outLen, &outLen);
    
    EVP_CIPHER_CTX_free(ctx);
#endif
}

void AesOpenMPEngine::decrypt(const uint8_t* input, uint8_t* output, 
                               size_t size, const uint8_t* key, size_t keyLen,
                               const uint8_t* iv) {
    encrypt(input, output, size, key, keyLen, iv);
}

}
