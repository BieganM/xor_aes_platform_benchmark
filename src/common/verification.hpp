#pragma once

#include <string>
#include <cstdint>
#include <cstddef>
#include <vector>

namespace hpc_benchmark {

uint32_t calculateCRC32(const uint8_t* data, size_t size);
uint32_t calculateCRC32File(const std::string& filename);

std::string calculateSHA256(const uint8_t* data, size_t size);
std::string calculateSHA256File(const std::string& filename);

bool verifyBuffers(const uint8_t* original, const uint8_t* decrypted, size_t size);
bool verifyFiles(const std::string& original, const std::string& decrypted);

}
