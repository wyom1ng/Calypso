#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
#include "rendering/engine.h"
#include "util/error.h"

#ifdef _WIN32
#ifdef NDEBUG
int WinMain(HINSTANCE, HINSTANCE, LPSTR, int) {
#else
int main() {
#endif
#else
int main() {
#endif
  spdlog::set_level(spdlog::level::debug);
  spdlog::stdout_color_mt("validation_layer");

  try {
    rendering::Engine engine;
    engine.mainloop();
  } catch (const std::exception &e) {
    util::Error::displayFatalError(e);
    return 1;
  }

  return 0;
}