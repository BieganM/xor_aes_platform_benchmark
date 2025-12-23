#pragma once

#include "engines/i_cipher_engine.hpp"

namespace hpc_benchmark {

class XorCudaEngine : public ICipherEngine {
public:
    XorCudaEngine();
    ~XorCudaEngine() override;
    
    std::string getAlgorithmName() const override { return "XOR"; }
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
    uint8_t* d_key_;
    size_t allocatedSize_;
    size_t allocatedKeySize_;
    bool initialized_;
};

}
