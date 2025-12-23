#pragma once

#include "engines/i_cipher_engine.hpp"

namespace hpc_benchmark {

class AesOpenMPEngine : public ICipherEngine {
public:
    AesOpenMPEngine();
    ~AesOpenMPEngine() override;
    
    std::string getAlgorithmName() const override { return "AES-256-CTR"; }
    std::string getEngineName() const override { return "OpenMP"; }
    
    void encrypt(const uint8_t* input, uint8_t* output, 
                size_t size, const uint8_t* key, size_t keyLen,
                const uint8_t* iv = nullptr) override;
    
    void decrypt(const uint8_t* input, uint8_t* output, 
                size_t size, const uint8_t* key, size_t keyLen,
                const uint8_t* iv = nullptr) override;
    
    bool isAvailable() const override;
    
    void setNumThreads(int threads) { numThreads_ = threads; }
    
private:
    int numThreads_ = 0;
    std::array<uint8_t, 16> defaultIV_;
};

}
