//
// Created by Wyoming on 23/03/2021.
//

#include "file.h"

namespace util {

std::vector<std::byte> File::readFile(const std::filesystem::path &path) {
  auto file_size = std::filesystem::file_size(path);
  std::ifstream file(path, std::ios::binary);

  if (!file.is_open()) {
    throw std::runtime_error("failed to open file!");
  }
  std::vector<std::byte> buffer(file_size);
  file.read(reinterpret_cast<char *>(buffer.data()), file_size);
  file.close();

  // We have to check that reading from the stream actually worked.
  // If any of the stream operation above failed then `gcount()`
  // will return zero indicating that zero data was read from the
  // stream.
  buffer.resize(file.gcount());

  return buffer;
}

}  // namespace util
