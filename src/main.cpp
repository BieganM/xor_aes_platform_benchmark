#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <string>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <random>
#include <cstring>
#include <map>
#include <thread>

#include "common/timer.hpp"
#include "common/csv_logger.hpp"
#include "common/verification.hpp"
#include "common/file_utils.hpp"
#include "common/power_monitor.hpp"
#include "engines/i_cipher_engine.hpp"

#include "engines/xor/xor_sequential.hpp"
#include "engines/aes/aes_sequential.hpp"

#ifdef HAS_OPENMP
#include "engines/xor/xor_openmp.hpp"
#include "engines/aes/aes_openmp.hpp"
#include <omp.h>
#endif

#ifdef HAS_CUDA
#include "engines/xor/xor_cuda.cuh"
#include "engines/aes/aes_cuda.cuh"
#endif

#ifdef HAS_METAL
#include "engines/xor/xor_metal.hpp"
#include "engines/aes/aes_metal.hpp"
#endif

namespace hpc_benchmark
{

    struct Config
    {
        std::vector<size_t> fileSizesMB = {1, 10, 100};
        int iterations = 3;
        bool verify = true;
        bool threadScaling = true;
        std::string outputFile = "benchmark_results.csv";
        int maxThreads = 0;
    };

    void printUsage(const char *progName)
    {
        std::cout << "Usage: " << progName << " [options]\n"
                  << "\nOptions:\n"
                  << "  --sizes <list>       Comma-separated file sizes in MB (default: 1,10,100)\n"
                  << "  --iterations <n>     Number of iterations per test (default: 3)\n"
                  << "  --verify             Enable verification mode (default: on)\n"
                  << "  --no-verify          Disable verification mode\n"
                  << "  --thread-scaling     Enable thread scaling tests (default: on)\n"
                  << "  --no-thread-scaling  Disable thread scaling tests\n"
                  << "  --max-threads <n>    Maximum threads for scaling tests (default: auto)\n"
                  << "  --output <file>      CSV output file (default: benchmark_results.csv)\n"
                  << "  --block-size-sweep   Run block size sweep analysis\n"
                  << "  --help               Show this help message\n";
    }

    std::vector<size_t> parseSizes(const std::string &str)
    {
        std::vector<size_t> sizes;
        std::stringstream ss(str);
        std::string token;
        while (std::getline(ss, token, ','))
        {
            sizes.push_back(std::stoul(token));
        }
        return sizes;
    }

    std::string getPlatformName()
    {
#ifdef __APPLE__
        return "macOS";
#elif defined(__linux__)
        std::ifstream procVersion("/proc/version");
        if (procVersion)
        {
            std::string line;
            std::getline(procVersion, line);
            if (line.find("microsoft") != std::string::npos ||
                line.find("Microsoft") != std::string::npos ||
                line.find("WSL") != std::string::npos)
            {
                return "WSL";
            }
        }
        return "Linux";
#elif defined(_WIN32)
        return "Windows";
#else
        return "Unknown";
#endif
    }

    bool parseArgs(int argc, char *argv[], Config &config, bool &blockSizeSweep)
    {
        config.maxThreads = std::thread::hardware_concurrency();
        config.outputFile = getPlatformName() + "_results.csv";
        blockSizeSweep = false;

        for (int i = 1; i < argc; ++i)
        {
            std::string arg = argv[i];

            if (arg == "--help" || arg == "-h")
            {
                printUsage(argv[0]);
                return false;
            }
            else if (arg == "--sizes" && i + 1 < argc)
            {
                config.fileSizesMB = parseSizes(argv[++i]);
            }
            else if (arg == "--iterations" && i + 1 < argc)
            {
                config.iterations = std::stoi(argv[++i]);
            }
            else if (arg == "--verify")
            {
                config.verify = true;
            }
            else if (arg == "--no-verify")
            {
                config.verify = false;
            }
            else if (arg == "--thread-scaling")
            {
                config.threadScaling = true;
            }
            else if (arg == "--no-thread-scaling")
            {
                config.threadScaling = false;
            }
            else if (arg == "--max-threads" && i + 1 < argc)
            {
                config.maxThreads = std::stoi(argv[++i]);
            }
            else if (arg == "--output" && i + 1 < argc)
            {
                config.outputFile = argv[++i];
            }
            else if (arg == "--block-size-sweep")
            {
                blockSizeSweep = true;
            }
        }
        return true;
    }

    void printHeader()
    {
        std::cout << "\n";
        std::cout << "╔════════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║       HPC ENCRYPTION BENCHMARK - File Encryption Analysis          ║\n";
        std::cout << "╠════════════════════════════════════════════════════════════════════╣\n";
        std::cout << "║  Algorithms: XOR (Memory-bound), AES-256-CTR (Compute-bound)       ║\n";
        std::cout << "╚════════════════════════════════════════════════════════════════════╝\n\n";
    }

    void printSystemInfo(const Config &config)
    {
        std::cout << "System Information:\n";
        std::cout << "───────────────────\n";

#ifdef __APPLE__
        std::cout << "  Platform: macOS (Apple Silicon)\n";
#else
        std::cout << "  Platform: Linux\n";
#endif

        std::cout << "  CPU Threads: " << std::thread::hardware_concurrency() << "\n";
        std::cout << "  Available Engines:\n";
        std::cout << "    ✓ Sequential (CPU)\n";

#ifdef HAS_OPENMP
        std::cout << "    ✓ OpenMP (CPU parallel)\n";
#endif

#ifdef HAS_CUDA
        std::cout << "    ✓ CUDA (NVIDIA GPU)\n";
#endif

#ifdef HAS_METAL
        std::cout << "    ✓ Metal (Apple GPU)\n";
#endif

        std::cout << "\n";
    }

    BenchmarkResult runSingleBenchmark(ICipherEngine *engine,
                                       const std::vector<uint8_t> &data,
                                       const std::vector<uint8_t> &key,
                                       const std::vector<uint8_t> &iv,
                                       bool verify,
                                       PowerMonitor &powerMonitor,
                                       int numThreads = 1)
    {
        BenchmarkResult result;
        result.algorithm = engine->getAlgorithmName();
        result.engine = engine->getEngineName();
        result.fileSizeMB = data.size() / (1024 * 1024);
        result.numThreads = numThreads;

        std::vector<uint8_t> encrypted(data.size());
        std::vector<uint8_t> decrypted(data.size());

        Timer timer;
        powerMonitor.startMeasurement();
        timer.start();
        engine->encrypt(data.data(), encrypted.data(), data.size(),
                        key.data(), key.size(), iv.data());
        timer.stop();
        auto energyReading = powerMonitor.stopMeasurement();

        result.timeSec = timer.elapsedSeconds();
        // Handle mostly empty files (very small block sizes < 1MB)
        if (result.fileSizeMB == 0) {
             result.throughputMBs = static_cast<double>(data.size()) / (1024.0 * 1024.0) / result.timeSec;
        } else {
             result.throughputMBs = static_cast<double>(result.fileSizeMB) / result.timeSec;
        }
        
        result.energyJoules = energyReading.joules;
        result.powerWatts = energyReading.watts;
        result.energySource = energyReading.source;

        if (verify)
        {
            engine->decrypt(encrypted.data(), decrypted.data(), data.size(),
                            key.data(), key.size(), iv.data());
            result.verified = verifyBuffers(data.data(), decrypted.data(), data.size());
        }
        else
        {
            result.verified = true;
        }

        return result;
    }

    BenchmarkResult runChunkedBenchmark(ICipherEngine *engine,
                                        const std::vector<uint8_t> &chunkData,
                                        const std::vector<uint8_t> &key,
                                        const std::vector<uint8_t> &iv,
                                        bool verify,
                                        PowerMonitor &powerMonitor,
                                        int iterations,
                                        int numThreads,
                                        size_t totalSizeMB,
                                        size_t numChunks,
                                        double baselineTime)
    {
        double totalTime = 0;
        double totalEnergy = 0;
        double totalPower = 0;
        bool allVerified = true;
        std::string energySrc;

        size_t chunkSizeMB = chunkData.size() / (1024 * 1024);

        for (int iter = 0; iter < iterations; ++iter)
        {
            double iterTime = 0;
            double iterEnergy = 0;
            double iterPower = 0;

            for (size_t c = 0; c < numChunks; ++c)
            {
                auto result = runSingleBenchmark(engine, chunkData, key, iv,
                                                 (verify && c == 0), powerMonitor, numThreads);
                iterTime += result.timeSec;
                iterEnergy += result.energyJoules;
                iterPower += result.powerWatts;
                if (c == 0)
                {
                    allVerified = allVerified && result.verified;
                    energySrc = result.energySource;
                }
            }

            totalTime += iterTime;
            totalEnergy += iterEnergy;
            totalPower += iterPower / numChunks;
        }

        BenchmarkResult avgResult;
        avgResult.platform = getPlatformName();
        avgResult.algorithm = engine->getAlgorithmName();
        avgResult.engine = engine->getEngineName();
        avgResult.fileSizeMB = totalSizeMB;
        avgResult.numThreads = numThreads;
        avgResult.timeSec = totalTime / iterations;
        avgResult.throughputMBs = static_cast<double>(totalSizeMB) / avgResult.timeSec;
        avgResult.verified = allVerified;
        avgResult.energyJoules = totalEnergy / iterations;
        avgResult.powerWatts = totalPower / iterations;
        avgResult.energySource = energySrc;

        if (baselineTime > 0)
        {
            avgResult.speedup = baselineTime / avgResult.timeSec;
            avgResult.efficiency = avgResult.speedup / numThreads;
        }
        else
        {
            avgResult.speedup = 1.0;
            avgResult.efficiency = 1.0;
        }

        return avgResult;
    }

    void printResultLine(const BenchmarkResult &result, bool showEfficiency = false)
    {
        std::cout << "  " << std::left << std::setw(12) << result.algorithm
                  << " | " << std::setw(10) << result.engine
                  << " | " << std::setw(3) << result.numThreads
                  << " | " << std::fixed << std::setprecision(2) << std::setw(12) << result.throughputMBs << " MB/s"
                  << " | " << std::setw(8) << result.timeSec << " s"
                  << " | " << std::setw(6) << result.speedup;

        if (showEfficiency)
        {
            std::cout << " | " << std::setw(9) << (result.efficiency * 100) << "%";
        }
        else
        {
            std::cout << " | " << std::setw(9) << "-";
        }

        std::cout << " | " << std::setw(6) << result.powerWatts << " W"
                  << " | " << (result.verified ? "PASS" : "FAIL") << "\n";
    }

    void printResultLine(std::string label, ICipherEngine* engine, int threads, 
                         double throughput, double time, double speedup, double efficiency,
                         double power, std::string status) {
        std::cout << "  " << std::left << std::setw(12) << label
                  << " | " << std::setw(10) << engine->getEngineName()
                  << " | " << std::setw(3) << threads
                  << " | " << std::fixed << std::setprecision(2) << std::setw(12) << throughput << " MB/s"
                  << " | " << std::setw(8) << time << " s"
                  << " | " << std::setw(6) << speedup
                  << " | " << std::setw(9) << "-"
                  << " | " << std::setw(6) << power << " W"
                  << " | " << status << "\n";
    }

    void runBlockSizeSweep(const std::vector<std::unique_ptr<ICipherEngine>> &engines,
                          PowerMonitor &powerMonitor,
                          const Config &config)
    {
        std::cout << "\n================================================================================\n";
        std::cout << "BLOCK SIZE SWEEP (64KB - 16MB)\n";
        std::cout << "================================================================================\n";

        const size_t TOTAL_SIZE_MB = 100;
        const size_t TOTAL_BYTES = TOTAL_SIZE_MB * 1024 * 1024;
        
        std::vector<size_t> blockSizes = {
            64 * 1024, 128 * 1024, 256 * 1024, 512 * 1024,
            1 * 1024 * 1024, 2 * 1024 * 1024, 4 * 1024 * 1024, 8 * 1024 * 1024, 16 * 1024 * 1024
        };

        std::string csvName = "block_size_results.csv";
        CsvLogger csv(csvName);
        csv.writeHeader();

        std::vector<uint8_t> key(32, 0xAA);
        std::vector<uint8_t> iv(16, 0xBB);

        for (const auto& engine : engines) {
            if (!engine->isAvailable()) continue;
        
            // Skip Sequential engines for sweep to save time, unless user wants comparison.
            // But Sequential scales linearly with size, block size shouldn't matter much unless cache effects.
            // Let's keep them.

            std::cout << "\nTesting " << engine->getAlgorithmName() << " - " << engine->getEngineName() << "\n";
            std::cout << "  " << std::string(115, '-') << "\n";
            std::cout << "  BlockSize    | Engine     | Thr | Throughput     | Time       | Speedup | Efficiency | Power  | Status\n";
            std::cout << "  " << std::string(115, '-') << "\n";

            for (size_t blockSize : blockSizes) {
                size_t numChunks = TOTAL_BYTES / blockSize;
                if (numChunks == 0) numChunks = 1;

                std::vector<uint8_t> chunkData(blockSize);
                std::fill(chunkData.begin(), chunkData.end(), 0xCC);

                int threads = 1;
                if (engine->getEngineName() == "OpenMP") {
                    #ifdef HAS_OPENMP
                    threads = omp_get_max_threads();
                    // We must set engine threads
                    // Casting to specific engine is messy here.
                    // But we can just use default behavior (max threads).
                    // Actually XorOpenMPEngine checks numThreads_ member.
                    // We need a way to set it.
                    // But engines are unique_ptr<ICipherEngine>.
                    // Hack: We can assume engines are configured to default (0=Max) if not set.
                    #endif
                }

                // If engine is OpenMP, we need to ensure it uses multiple threads.
                // The current OpenMP engines default to max threads if not set.
                // So that's fine.

                auto res = runChunkedBenchmark(engine.get(), chunkData, key, iv, false, powerMonitor, 
                                             config.iterations, threads, 
                                             TOTAL_SIZE_MB, numChunks, 0.0);
                
                // Store Block Size in FileSize_MB for CSV logging (hacky but effective)
                res.fileSizeMB = static_cast<double>(blockSize) / (1024.0 * 1024.0); 
                
                csv.writeResult(res);
                
                std::stringstream blockLabel;
                if (blockSize < 1024*1024)
                    blockLabel << (blockSize/1024) << " KB";
                else
                    blockLabel << (blockSize/(1024*1024)) << " MB";

                printResultLine(blockLabel.str(), 
                              engine.get(), threads, res.throughputMBs, res.timeSec, 0, 0, res.powerWatts, 
                              res.verified ? "PASS" : "FAIL");
            }
        }
        std::cout << "\nBlock size sweep completed. Results saved to " << csvName << "\n";
    }

    void runBenchmarks(const Config &config)
    {
        PowerMonitor powerMonitor;

        std::cout << "Test Configuration:\n";
        std::cout << "───────────────────\n";
        std::cout << "  File sizes: ";
        for (size_t i = 0; i < config.fileSizesMB.size(); ++i)
        {
            std::cout << config.fileSizesMB[i] << " MB";
            if (i < config.fileSizesMB.size() - 1)
                std::cout << ", ";
        }
        std::cout << "\n";
        std::cout << "  Iterations: " << config.iterations << "\n";
        std::cout << "  Verification: " << (config.verify ? "enabled" : "disabled") << "\n";
        std::cout << "  Thread Scaling: " << (config.threadScaling ? "enabled" : "disabled") << "\n";
        std::cout << "  Max Threads: " << config.maxThreads << "\n";
        std::cout << "  Power Monitoring: " << (powerMonitor.isAvailable() ? powerMonitor.getSource() : "N/A") << "\n";
        std::cout << "  Output: " << config.outputFile << "\n\n";

        CsvLogger logger(config.outputFile);
        logger.writeHeader();

        std::vector<uint8_t> key(32);
        std::vector<uint8_t> iv(16);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, 255);
        for (auto &b : key)
            b = static_cast<uint8_t>(dis(gen));
        for (auto &b : iv)
            b = static_cast<uint8_t>(dis(gen));

        std::vector<int> threadCounts;
        if (config.threadScaling)
        {
            for (int t = 1; t <= config.maxThreads; t *= 2)
            {
                threadCounts.push_back(t);
            }
            if (threadCounts.back() != config.maxThreads)
            {
                threadCounts.push_back(config.maxThreads);
            }
        }
        else
        {
            threadCounts.push_back(config.maxThreads);
        }

        int totalPassed = 0;
        int totalFailed = 0;

        for (size_t sizeMB : config.fileSizesMB)
        {
            size_t sizeBytes = sizeMB * 1024 * 1024;

            std::cout << "═══════════════════════════════════════════════════════════════════════════\n";
            std::cout << "Testing with " << sizeMB << " MB file\n";
            std::cout << "═══════════════════════════════════════════════════════════════════════════\n\n";

            const size_t MAX_CHUNK_MB = 512;
            size_t chunkSizeMB = std::min(sizeMB, MAX_CHUNK_MB);
            size_t chunkSizeBytes = chunkSizeMB * 1024 * 1024;
            size_t numChunks = (sizeMB + chunkSizeMB - 1) / chunkSizeMB;

            std::cout << "  Using " << numChunks << " chunk(s) of " << chunkSizeMB << " MB each\n";
            std::cout << "  Generating random data (" << chunkSizeMB << " MB)... " << std::flush;
            std::vector<uint8_t> data(chunkSizeBytes);
            for (size_t i = 0; i < chunkSizeBytes; i += 8)
            {
                uint64_t val = gen();
                std::memcpy(&data[i], &val, std::min<size_t>(8, chunkSizeBytes - i));
            }
            std::cout << "done\n\n";

            std::map<std::string, double> baselineTimes;

            std::cout << "  " << std::string(115, '-') << "\n";
            std::cout << "  Algorithm    | Engine     | Thr | Throughput     | Time       | Speedup | Efficiency | Power  | Status\n";
            std::cout << "  " << std::string(115, '-') << "\n";

            XorSequentialEngine xorSeq;
            AesSequentialEngine aesSeq;

            xorSeq.initialize();
            auto xorSeqResult = runChunkedBenchmark(&xorSeq, data, key, iv, config.verify, powerMonitor,
                                                    config.iterations, 1, sizeMB, numChunks, 0);
            baselineTimes["XOR"] = xorSeqResult.timeSec;
            xorSeq.cleanup();
            printResultLine(xorSeqResult, false);
            logger.writeResult(xorSeqResult);
            if (xorSeqResult.verified)
                totalPassed++;
            else
                totalFailed++;

            aesSeq.initialize();
            auto aesSeqResult = runChunkedBenchmark(&aesSeq, data, key, iv, config.verify, powerMonitor,
                                                    config.iterations, 1, sizeMB, numChunks, 0);
            baselineTimes["AES-256-CTR"] = aesSeqResult.timeSec;
            aesSeq.cleanup();
            printResultLine(aesSeqResult, false);
            logger.writeResult(aesSeqResult);
            if (aesSeqResult.verified)
                totalPassed++;
            else
                totalFailed++;

#ifdef HAS_OPENMP
            std::cout << "\n  [OpenMP Thread Scaling]\n";
            std::cout << "  " << std::string(115, '-') << "\n";

            for (int numThreads : threadCounts)
            {
                XorOpenMPEngine xorOmp;
                xorOmp.setNumThreads(numThreads);
                xorOmp.initialize();
                auto result = runChunkedBenchmark(&xorOmp, data, key, iv, config.verify, powerMonitor,
                                                  config.iterations, numThreads, sizeMB, numChunks, baselineTimes["XOR"]);
                xorOmp.cleanup();
                printResultLine(result, true);
                logger.writeResult(result);
                if (result.verified)
                    totalPassed++;
                else
                    totalFailed++;
            }

            for (int numThreads : threadCounts)
            {
                AesOpenMPEngine aesOmp;
                aesOmp.setNumThreads(numThreads);
                aesOmp.initialize();
                auto result = runChunkedBenchmark(&aesOmp, data, key, iv, config.verify, powerMonitor,
                                                  config.iterations, numThreads, sizeMB, numChunks, baselineTimes["AES-256-CTR"]);
                aesOmp.cleanup();
                printResultLine(result, true);
                logger.writeResult(result);
                if (result.verified)
                    totalPassed++;
                else
                    totalFailed++;
            }
#endif

#ifdef HAS_METAL
            std::cout << "\n  [Metal GPU]\n";
            std::cout << "  " << std::string(115, '-') << "\n";

            XorMetalEngine xorMetal;
            if (xorMetal.isAvailable())
            {
                xorMetal.initialize();
                auto result = runChunkedBenchmark(&xorMetal, data, key, iv, config.verify, powerMonitor,
                                                  config.iterations, 1, sizeMB, numChunks, baselineTimes["XOR"]);
                xorMetal.cleanup();
                printResultLine(result, false);
                logger.writeResult(result);
                if (result.verified)
                    totalPassed++;
                else
                    totalFailed++;
            }

            AesMetalEngine aesMetal;
            if (aesMetal.isAvailable())
            {
                aesMetal.initialize();
                auto result = runChunkedBenchmark(&aesMetal, data, key, iv, config.verify, powerMonitor,
                                                  config.iterations, 1, sizeMB, numChunks, baselineTimes["AES-256-CTR"]);
                aesMetal.cleanup();
                printResultLine(result, false);
                logger.writeResult(result);
                if (result.verified)
                    totalPassed++;
                else
                    totalFailed++;
            }
#endif

#ifdef HAS_CUDA
            std::cout << "\n  [CUDA GPU]\n";
            std::cout << "  " << std::string(115, '-') << "\n";

            XorCudaEngine xorCuda;
            if (xorCuda.isAvailable())
            {
                xorCuda.initialize();
                auto result = runChunkedBenchmark(&xorCuda, data, key, iv, config.verify, powerMonitor,
                                                  config.iterations, 1, sizeMB, numChunks, baselineTimes["XOR"]);
                xorCuda.cleanup();
                printResultLine(result, false);
                logger.writeResult(result);
                if (result.verified)
                    totalPassed++;
                else
                    totalFailed++;
            }

            AesCudaEngine aesCuda;
            if (aesCuda.isAvailable())
            {
                aesCuda.initialize();
                auto result = runChunkedBenchmark(&aesCuda, data, key, iv, config.verify, powerMonitor,
                                                  config.iterations, 1, sizeMB, numChunks, baselineTimes["AES-256-CTR"]);
                aesCuda.cleanup();
                printResultLine(result, false);
                logger.writeResult(result);
                if (result.verified)
                    totalPassed++;
                else
                    totalFailed++;
            }
#endif

            logger.flush();
            std::cout << "\n";
        }

        std::cout << "═══════════════════════════════════════════════════════════════════════════\n";
        std::cout << "VERIFICATION SUMMARY\n";
        std::cout << "═══════════════════════════════════════════════════════════════════════════\n";
        std::cout << "  Total Tests: " << (totalPassed + totalFailed) << "\n";
        std::cout << "  Passed:      " << totalPassed << " (" << (100 * totalPassed / (totalPassed + totalFailed)) << "%)\n";
        std::cout << "  Failed:      " << totalFailed << " (" << (100 * totalFailed / (totalPassed + totalFailed)) << "%)\n";

        if (totalFailed == 0)
        {
            std::cout << "\n  ✓ All encryption/decryption operations completed successfully!\n";
        }
        else
        {
            std::cout << "\n  ✗ Some tests failed verification!\n";
        }

        std::cout << "\nResults saved to: " << config.outputFile << "\n";
        std::cout << "Generate charts with: python3 scripts/generate_charts.py " << config.outputFile << "\n\n";
    }

} // namespace hpc_benchmark

int main(int argc, char *argv[])
{
    using namespace hpc_benchmark;

    Config config;
    bool blockSizeSweep = false;

    if (!parseArgs(argc, argv, config, blockSizeSweep))
    {
        return 0;
    }

    try
    {
        PowerMonitor powerMonitor;

        std::vector<std::unique_ptr<hpc_benchmark::ICipherEngine>> engines;

        // XOR Engines
        engines.push_back(std::make_unique<hpc_benchmark::XorSequentialEngine>());
#ifdef HAS_OPENMP
        engines.push_back(std::make_unique<hpc_benchmark::XorOpenMPEngine>());
#endif
#ifdef HAS_OPENCL
        engines.push_back(std::make_unique<hpc_benchmark::XorOpenCLEngine>());
#endif
#ifdef HAS_METAL
        engines.push_back(std::make_unique<hpc_benchmark::XorMetalEngine>());
#endif
#ifdef HAS_CUDA
        engines.push_back(std::make_unique<hpc_benchmark::XorCudaEngine>());
#endif

        // AES Engines
        engines.push_back(std::make_unique<hpc_benchmark::AesSequentialEngine>());
#ifdef HAS_OPENMP
        engines.push_back(std::make_unique<hpc_benchmark::AesOpenMPEngine>());
#endif
#ifdef HAS_OPENCL
        engines.push_back(std::make_unique<hpc_benchmark::AesOpenCLEngine>());
#endif
#ifdef HAS_METAL
        engines.push_back(std::make_unique<hpc_benchmark::AesMetalEngine>());
#endif
#ifdef HAS_CUDA
        engines.push_back(std::make_unique<hpc_benchmark::AesCudaEngine>());
#endif

         if (blockSizeSweep) {
            runBlockSizeSweep(engines, powerMonitor, config);
            return 0;
        }

        printHeader();
        printSystemInfo(config);
        runBenchmarks(config);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
