#pragma once

#include "engines/i_cipher_engine.hpp"
#include <array>

namespace hpc_benchmark {

class AesMetalEngine : public ICipherEngine {
public:
    AesMetalEngine();
    ~AesMetalEngine() override;
    
    std::string getAlgorithmName() const override { return "AES-256-CTR"; }
    std::string getEngineName() const override { return "Metal"; }
    
    void encrypt(const uint8_t* input, uint8_t* output, 
                size_t size, const uint8_t* key, size_t keyLen,
                const uint8_t* iv = nullptr) override;
    
    void decrypt(const uint8_t* input, uint8_t* output, 
                size_t size, const uint8_t* key, size_t keyLen,
                const uint8_t* iv = nullptr) override;
    
    bool isAvailable() const override;
    void initialize() override;
    void cleanup() override;
    
private:
    struct Impl;
    Impl* impl_;
    bool initialized_;
    std::array<uint8_t, 16> defaultIV_;
};

}
