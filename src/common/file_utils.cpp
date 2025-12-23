#include "file_utils.hpp"
#include <fstream>
#include <random>
#include <cstring>
#include <vector>
#include <stdexcept>
#include <sys/stat.h>

namespace hpc_benchmark
{

    void generateRandomFile(const std::string &filename, size_t sizeBytes)
    {
        std::ofstream file(filename, std::ios::binary);
        if (!file)
        {
            throw std::runtime_error("Cannot create file: " + filename);
        }

        std::random_device rd;
        std::mt19937_64 gen(rd());
        std::uniform_int_distribution<uint64_t> dis;

        constexpr size_t BUFFER_SIZE = 1024 * 1024;
        std::vector<uint8_t> buffer(BUFFER_SIZE);

        size_t remaining = sizeBytes;
        while (remaining > 0)
        {
            size_t toWrite = std::min(BUFFER_SIZE, remaining);

            size_t i = 0;
            while (i + 8 <= toWrite)
            {
                uint64_t val = dis(gen);
                std::memcpy(&buffer[i], &val, 8);
                i += 8;
            }
            while (i < toWrite)
            {
                buffer[i] = static_cast<uint8_t>(dis(gen));
                ++i;
            }

            file.write(reinterpret_cast<char *>(buffer.data()), toWrite);
            remaining -= toWrite;
        }
    }

    size_t getFileSize(const std::string &filename)
    {
        struct stat st;
        if (stat(filename.c_str(), &st) != 0)
        {
            throw std::runtime_error("Cannot get file size: " + filename);
        }
        return static_cast<size_t>(st.st_size);
    }

    bool fileExists(const std::string &filename)
    {
        struct stat st;
        return stat(filename.c_str(), &st) == 0;
    }

}
