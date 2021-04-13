//
// Created by Wyoming on 13/04/2021.
//

#include "error.h"

namespace util {

void Error::displayFatalError(const std::exception &exception, const std::string_view &title) {
  boxer::show(exception.what(), title.data(), boxer::Style::Error);
}

}