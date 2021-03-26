//
// Created by Wyoming on 23/03/2021.
//

#ifndef CALYPSO_FILE_H
#define CALYPSO_FILE_H

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <vector>

namespace util {

class File {
 public:
  static std::vector<std::byte> readFile(const std::filesystem::path &path);
};

}

#endif  // CALYPSO_FILE_H
