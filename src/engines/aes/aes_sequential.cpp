#include "aes_sequential.hpp"
#include <openssl/evp.h>
#include <openssl/rand.h>
#include <stdexcept>
#include <cstring>
#include <array>

namespace hpc_benchmark {

struct AesSequentialEngine::Impl {
    EVP_CIPHER_CTX* ctx = nullptr;
    std::array<uint8_t, 16> defaultIV;
    
    Impl() {
        ctx = EVP_CIPHER_CTX_new();
        RAND_bytes(defaultIV.data(), 16);
    }
    
    ~Impl() {
        if (ctx) EVP_CIPHER_CTX_free(ctx);
    }
};

AesSequentialEngine::AesSequentialEngine() : impl_(new Impl()) {}

AesSequentialEngine::~AesSequentialEngine() {
    cleanup();
    delete impl_;
}

bool AesSequentialEngine::isAvailable() const {
    return true;
}

void AesSequentialEngine::initialize() {}

void AesSequentialEngine::cleanup() {}

void AesSequentialEngine::encrypt(const uint8_t* input, uint8_t* output, 
                                   size_t size, const uint8_t* key, size_t keyLen,
                                   const uint8_t* iv) {
    if (keyLen != 32) {
        throw std::runtime_error("AES-256 requires 32-byte key");
    }
    
    const uint8_t* actualIV = iv ? iv : impl_->defaultIV.data();
    
    EVP_CIPHER_CTX_reset(impl_->ctx);
    
    if (EVP_EncryptInit_ex(impl_->ctx, EVP_aes_256_ctr(), nullptr, key, actualIV) != 1) {
        throw std::runtime_error("EVP_EncryptInit_ex failed");
    }
    
    int outLen = 0;
    int totalLen = 0;
    
    if (EVP_EncryptUpdate(impl_->ctx, output, &outLen, input, static_cast<int>(size)) != 1) {
        throw std::runtime_error("EVP_EncryptUpdate failed");
    }
    totalLen = outLen;
    
    if (EVP_EncryptFinal_ex(impl_->ctx, output + totalLen, &outLen) != 1) {
        throw std::runtime_error("EVP_EncryptFinal_ex failed");
    }
}

void AesSequentialEngine::decrypt(const uint8_t* input, uint8_t* output, 
                                   size_t size, const uint8_t* key, size_t keyLen,
                                   const uint8_t* iv) {
    encrypt(input, output, size, key, keyLen, iv);
}

}
