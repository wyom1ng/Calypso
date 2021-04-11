#include <iostream>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
#include "rendering/engine.h"

int main() {
  spdlog::set_level(spdlog::level::debug);
  spdlog::stdout_color_mt("validation_layer");

  rendering::Engine engine;

  try {
    engine.mainloop();
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}