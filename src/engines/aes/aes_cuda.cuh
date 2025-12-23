#pragma once

#include "engines/i_cipher_engine.hpp"
#include <array>

namespace hpc_benchmark {

class AesCudaEngine : public ICipherEngine {
public:
    AesCudaEngine();
    ~AesCudaEngine() override;
    
    std::string getAlgorithmName() const override { return "AES-256-CTR"; }
    std::string getEngineName() const override { return "CUDA"; }
    
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
    uint8_t* d_input_;
    uint8_t* d_output_;
    uint32_t* d_roundKeys_;
    uint8_t* d_iv_;
    size_t allocatedSize_;
    bool initialized_;
    std::array<uint8_t, 16> defaultIV_;
};

}
