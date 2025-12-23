#pragma once

#include <string>
#include <cstdint>
#include <cstddef>
#include <memory>
#include <vector>

namespace hpc_benchmark {

class ICipherEngine {
public:
    virtual ~ICipherEngine() = default;
    
    virtual std::string getAlgorithmName() const = 0;
    virtual std::string getEngineName() const = 0;
    
    virtual void encrypt(const uint8_t* input, uint8_t* output, 
                        size_t size, const uint8_t* key, size_t keyLen,
                        const uint8_t* iv = nullptr) = 0;
    
    virtual void decrypt(const uint8_t* input, uint8_t* output, 
                        size_t size, const uint8_t* key, size_t keyLen,
                        const uint8_t* iv = nullptr) = 0;
    
    virtual bool isAvailable() const = 0;
    
    virtual void initialize() {}
    virtual void cleanup() {}
    
    virtual size_t getOptimalBlockSize() const { return 1024 * 1024; }
};

using CipherEnginePtr = std::unique_ptr<ICipherEngine>;

struct BenchmarkResult {
    std::string platform;
    std::string algorithm;
    std::string engine;
    size_t fileSizeMB;
    int numThreads;
    double timeSec;
    double throughputMBs;
    double speedup;
    double efficiency;
    bool verified;
    double energyJoules;
    double powerWatts;
    std::string energySource;
    
    BenchmarkResult() : platform("Unknown"), fileSizeMB(0), numThreads(1), timeSec(0), 
                        throughputMBs(0), speedup(1.0), efficiency(1.0), verified(false),
                        energyJoules(0), powerWatts(0), energySource("N/A") {}
};

}
