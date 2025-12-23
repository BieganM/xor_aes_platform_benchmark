#pragma once

#include "engines/i_cipher_engine.hpp"

namespace hpc_benchmark {

class XorSequentialEngine : public ICipherEngine {
public:
    std::string getAlgorithmName() const override { return "XOR"; }
    std::string getEngineName() const override { return "Sequential"; }
    
    void encrypt(const uint8_t* input, uint8_t* output, 
                size_t size, const uint8_t* key, size_t keyLen,
                const uint8_t* iv = nullptr) override;
    
    void decrypt(const uint8_t* input, uint8_t* output, 
                size_t size, const uint8_t* key, size_t keyLen,
                const uint8_t* iv = nullptr) override;
    
    bool isAvailable() const override { return true; }
};

}
