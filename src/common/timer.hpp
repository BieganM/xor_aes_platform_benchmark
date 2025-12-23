#pragma once

#include <chrono>

namespace hpc_benchmark {

class Timer {
public:
    void start();
    void stop();
    double elapsedSeconds() const;
    double elapsedMilliseconds() const;
    
private:
    std::chrono::high_resolution_clock::time_point startTime_;
    std::chrono::high_resolution_clock::time_point endTime_;
};

}
