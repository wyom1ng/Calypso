#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

#include "hello_triangle.h"
#include <spdlog/sinks/stdout_color_sinks.h>
#include <iostream>

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

int main() {
  spdlog::set_level(spdlog::level::debug);
  spdlog::stdout_color_mt("validation_layer");

  HelloTriangle app;

  try {
    app.mainloop();
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}