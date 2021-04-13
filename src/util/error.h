//
// Created by Wyoming on 13/04/2021.
//

#ifndef CALYPSO_ERROR_H
#define CALYPSO_ERROR_H

#include <stdexcept>
#include <boxer/boxer.h>

namespace util {

class Error {
 public:
  static void displayFatalError(const std::exception &exception, const std::string_view &title = "Error");
};

}

#endif  // CALYPSO_ERROR_H
