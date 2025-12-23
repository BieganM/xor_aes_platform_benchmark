#include "csv_logger.hpp"
#include <iomanip>

namespace hpc_benchmark {

CsvLogger::CsvLogger(const std::string& filename) 
    : file_(filename), headerWritten_(false) {
    if (!file_.is_open()) {
        throw std::runtime_error("Cannot open CSV file: " + filename);
    }
}

CsvLogger::~CsvLogger() {
    if (file_.is_open()) {
        file_.close();
    }
}

void CsvLogger::writeHeader() {
    if (!headerWritten_) {
        file_ << "Platform,Algorithm,Engine,FileSize_MB,NumThreads,Time_Sec,Throughput_MBs,Speedup,Efficiency,Verified,Energy_Joules,Power_Watts,Energy_Source\n";
        headerWritten_ = true;
    }
}

void CsvLogger::writeResult(const BenchmarkResult& result) {
    if (!headerWritten_) {
        writeHeader();
    }
    
    file_ << result.platform << ","
          << result.algorithm << ","
          << result.engine << ","
          << result.fileSizeMB << ","
          << result.numThreads << ","
          << std::fixed << std::setprecision(6) << result.timeSec << ","
          << std::fixed << std::setprecision(2) << result.throughputMBs << ","
          << std::fixed << std::setprecision(4) << result.speedup << ","
          << std::fixed << std::setprecision(4) << result.efficiency << ","
          << (result.verified ? "PASS" : "FAIL") << ","
          << std::fixed << std::setprecision(4) << result.energyJoules << ","
          << std::fixed << std::setprecision(2) << result.powerWatts << ","
          << result.energySource << "\n";
}

void CsvLogger::flush() {
    file_.flush();
}

}
