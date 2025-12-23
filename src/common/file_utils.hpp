#pragma once

#include <string>
#include <cstddef>

namespace hpc_benchmark {

void generateRandomFile(const std::string& filename, size_t sizeBytes);
size_t getFileSize(const std::string& filename);
bool fileExists(const std::string& filename);

}
