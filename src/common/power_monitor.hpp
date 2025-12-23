#pragma once

#include <string>
#include <cstdint>

namespace hpc_benchmark {

struct EnergyReading {
    double joules;
    double watts;
    double durationSec;
    bool valid;
    std::string source;
    
    EnergyReading() : joules(0), watts(0), durationSec(0), valid(false), source("unknown") {}
};

class PowerMonitor {
public:
    PowerMonitor();
    ~PowerMonitor();
    
    void startMeasurement();
    EnergyReading stopMeasurement();
    
    bool isAvailable() const;
    std::string getSource() const;
    
private:
    struct Impl;
    Impl* impl_;
};

}
