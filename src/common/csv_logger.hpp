#pragma once

#include "engines/i_cipher_engine.hpp"
#include <string>
#include <fstream>
#include <vector>

namespace hpc_benchmark {

class CsvLogger {
public:
    explicit CsvLogger(const std::string& filename);
    ~CsvLogger();
    
    void writeHeader();
    void writeResult(const BenchmarkResult& result);
    void flush();
    
private:
    std::ofstream file_;
    bool headerWritten_;
};

}
