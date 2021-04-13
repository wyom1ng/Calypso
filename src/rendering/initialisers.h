//
// Created by Wyoming on 11/04/2021.
//

#ifndef CALYPSO_INITIALISERS_H
#define CALYPSO_INITIALISERS_H

#include <filesystem>
#include <functional>
#include <GLFW/glfw3.h>
#include <stb_image.h>
#include <vulkan/vulkan.hpp>

namespace rendering {

class Initialisers {
 public:
  static void initDispatchLoader(const vk::DynamicLoader &dynamicLoader);
  
  static GLFWwindow *createWindow(void *user, std::function<void(GLFWwindow *window, int width, int height)> framebufferResizeCallback, uint16_t initialWidth, uint16_t initialHeight);

  static bool checkValidationLayerSupport(const std::vector<const char *> &validationLayers);

  static std::vector<const char *> getRequiredExtensions(bool enableValidationLayers);

  static vk::DebugUtilsMessengerCreateInfoEXT getDebugMessengerCreateInfo(PFN_vkDebugUtilsMessengerCallbackEXT debugCallback);
  
  static vk::Instance createInstance(bool enableValidationLayers, std::vector<const char *> validationLayers, PFN_vkDebugUtilsMessengerCallbackEXT debugCallback);

  static vk::DebugUtilsMessengerEXT setupDebugMessenger(const vk::Instance &instance, bool enableValidationLayers, PFN_vkDebugUtilsMessengerCallbackEXT debugCallback);
};

}  // namespace rendering

#endif  // CALYPSO_INITIALISERS_H
