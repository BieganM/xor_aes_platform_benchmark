#include "power_monitor.hpp"
#include <chrono>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <array>
#include <memory>
#include <vector>
#include <algorithm>

#ifdef __APPLE__
#include <mach/mach.h>
#include <mach/processor_info.h>
#include <mach/mach_host.h>
#endif

namespace hpc_benchmark
{

    struct PowerMonitor::Impl
    {
        std::chrono::high_resolution_clock::time_point startTime;
        double startEnergy = 0;
        std::string source = "none";
        bool available = false;

#ifdef __linux__
        std::string raplPath;

        bool findRaplPath()
        {
            std::vector<std::string> paths = {
                "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj",
                "/sys/class/powercap/intel-rapl:0/energy_uj",
                "/sys/devices/virtual/powercap/intel-rapl/intel-rapl:0/energy_uj"};

            for (const auto &path : paths)
            {
                std::ifstream f(path);
                if (f.good())
                {
                    raplPath = path;
                    return true;
                }
            }
            return false;
        }

        double readRaplEnergy()
        {
            std::ifstream f(raplPath);
            if (!f)
                return 0;

            uint64_t microjoules;
            f >> microjoules;
            return microjoules / 1e6;
        }

        double readNvidiaPower()
        {
            std::array<char, 128> buffer;
            std::string result;

            FILE *pipe = popen("nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits 2>/dev/null", "r");
            if (!pipe)
                return 0;

            while (fgets(buffer.data(), buffer.size(), pipe) != nullptr)
            {
                result += buffer.data();
            }
            pclose(pipe);

            try
            {
                return std::stod(result);
            }
            catch (...)
            {
                return 0;
            }
        }
#endif

#ifdef __APPLE__
        double estimatedTDP = 30.0;

        double getCpuUsage()
        {
            host_cpu_load_info_data_t cpuinfo;
            mach_msg_type_number_t count = HOST_CPU_LOAD_INFO_COUNT;

            if (host_statistics(mach_host_self(), HOST_CPU_LOAD_INFO,
                                (host_info_t)&cpuinfo, &count) != KERN_SUCCESS)
            {
                return 0.5;
            }

            static uint64_t prevIdle = 0, prevTotal = 0;
            uint64_t idle = cpuinfo.cpu_ticks[CPU_STATE_IDLE];
            uint64_t total = cpuinfo.cpu_ticks[CPU_STATE_USER] +
                             cpuinfo.cpu_ticks[CPU_STATE_SYSTEM] +
                             cpuinfo.cpu_ticks[CPU_STATE_IDLE] +
                             cpuinfo.cpu_ticks[CPU_STATE_NICE];

            double usage = 1.0 - (double)(idle - prevIdle) / (double)(total - prevTotal + 1);
            prevIdle = idle;
            prevTotal = total;

            return std::max(0.0, std::min(1.0, usage));
        }
#endif

        void detectSource()
        {
#ifdef __linux__
            if (findRaplPath())
            {
                source = "Intel RAPL";
                available = true;
                return;
            }

            if (readNvidiaPower() > 0)
            {
                source = "NVIDIA SMI";
                available = true;
                return;
            }
#endif

#ifdef __APPLE__
            source = "Apple Silicon (estimated)";
            available = true;
#endif
        }
    };

    PowerMonitor::PowerMonitor() : impl_(new Impl())
    {
        impl_->detectSource();
    }

    PowerMonitor::~PowerMonitor()
    {
        delete impl_;
    }

    bool PowerMonitor::isAvailable() const
    {
        return impl_->available;
    }

    std::string PowerMonitor::getSource() const
    {
        return impl_->source;
    }

    void PowerMonitor::startMeasurement()
    {
        impl_->startTime = std::chrono::high_resolution_clock::now();

#ifdef __linux__
        if (impl_->source == "Intel RAPL")
        {
            impl_->startEnergy = impl_->readRaplEnergy();
        }
#endif

#ifdef __APPLE__
        impl_->getCpuUsage();
#endif
    }

    EnergyReading PowerMonitor::stopMeasurement()
    {
        auto endTime = std::chrono::high_resolution_clock::now();
        EnergyReading reading;

        reading.durationSec = std::chrono::duration<double>(endTime - impl_->startTime).count();
        reading.source = impl_->source;

#ifdef __linux__
        if (impl_->source == "Intel RAPL")
        {
            double endEnergy = impl_->readRaplEnergy();
            reading.joules = endEnergy - impl_->startEnergy;

            if (reading.joules < 0)
            {
                reading.joules += 16777.216;
            }

            reading.watts = reading.joules / reading.durationSec;
            reading.valid = true;
        }
        else if (impl_->source == "NVIDIA SMI")
        {
            reading.watts = impl_->readNvidiaPower();
            reading.joules = reading.watts * reading.durationSec;
            reading.valid = (reading.watts > 0);
        }
#endif

#ifdef __APPLE__
        double cpuUsage = impl_->getCpuUsage();

        double basePower = 5.0;
        double maxPower = impl_->estimatedTDP;
        reading.watts = basePower + cpuUsage * (maxPower - basePower);
        reading.joules = reading.watts * reading.durationSec;
        reading.valid = true;
#endif

        return reading;
    }

}
