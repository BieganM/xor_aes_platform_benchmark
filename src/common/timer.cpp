#include "timer.hpp"

namespace hpc_benchmark {

void Timer::start() {
    startTime_ = std::chrono::high_resolution_clock::now();
}

void Timer::stop() {
    endTime_ = std::chrono::high_resolution_clock::now();
}

double Timer::elapsedSeconds() const {
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime_ - startTime_);
    return duration.count() / 1e9;
}

double Timer::elapsedMilliseconds() const {
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime_ - startTime_);
    return duration.count() / 1e6;
}

}
