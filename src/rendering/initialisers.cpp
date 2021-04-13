//
// Created by Wyoming on 11/04/2021.
//

#include "initialisers.h"

namespace rendering {

void Initialisers::initDispatchLoader(const vk::DynamicLoader &dynamicLoader) {
  auto vk_get_instance_proc_addr = dynamicLoader.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
  VULKAN_HPP_DEFAULT_DISPATCHER.init(vk_get_instance_proc_addr);
}

GLFWwindow *Initialisers::createWindow(void *user, std::function<void(GLFWwindow *, int, int)> framebufferResizeCallback, uint16_t initialWidth, uint16_t initialHeight) {
  glfwInit();
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

  auto *window = glfwCreateWindow(initialWidth, initialHeight, "Calypso", nullptr, nullptr);

#ifndef NDEBUG
  // move window to 2nd monitor so it doesn't overlap my IDE (:
  glfwSetWindowMonitor(window, nullptr, 3840 / 2 - initialWidth / 2, -2160 / 2 - initialHeight / 2, initialWidth, initialHeight, GLFW_DONT_CARE);
  glfwMaximizeWindow(window);
#endif

  glfwSetWindowUserPointer(window, user);
  glfwSetFramebufferSizeCallback(window, *framebufferResizeCallback.target<GLFWframebuffersizefun>());

  GLFWimage images[2];
  int large_width;
  int large_channels;
  int large_height;

  auto large_path = std::filesystem::path(ROOT_DIRECTORY) / "assets/calypso256x.png";
  stbi_uc *large_pixels = stbi_load(large_path.generic_string().c_str(), &large_width, &large_height, &large_channels, STBI_rgb_alpha);
  images[0] = {
      .width = large_width,
      .height = large_height,
      .pixels = large_pixels,
  };

  int small_width;
  int small_channels;
  int small_height;

  auto small_path = std::filesystem::path(ROOT_DIRECTORY) / "assets/calypso48x.png";
  stbi_uc *small_pixels = stbi_load(small_path.generic_string().c_str(), &small_width, &small_height, &small_channels, STBI_rgb_alpha);
  images[1] = {
      .width = small_width,
      .height = small_height,
      .pixels = small_pixels,
  };

  glfwSetWindowIcon(window, 2, images);

  return window;
}

bool Initialisers::checkValidationLayerSupport(const std::vector<const char *> &validationLayers) {
  auto available_layers = vk::enumerateInstanceLayerProperties();

  for (const char *layer_name : validationLayers) {
    bool layer_found = false;

    for (const auto &layer_properties : available_layers) {
      if (strcmp(layer_name, layer_properties.layerName) == 0) {
        layer_found = true;
        break;
      }
    }

    if (!layer_found) {
      return false;
    }
  }

  return true;
}

std::vector<const char *> Initialisers::getRequiredExtensions(bool enableValidationLayers) {
  uint32_t glfw_extension_count = 0;
  const char **glfw_extensions;
  glfw_extensions = glfwGetRequiredInstanceExtensions(&glfw_extension_count);

  std::vector<const char *> extensions(glfw_extensions, glfw_extensions + glfw_extension_count);

  if (enableValidationLayers) {
    extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
  }

  return extensions;
}

vk::DebugUtilsMessengerCreateInfoEXT Initialisers::getDebugMessengerCreateInfo(PFN_vkDebugUtilsMessengerCallbackEXT debugCallback) {
  vk::DebugUtilsMessengerCreateInfoEXT create_info = {};

  create_info.messageSeverity = vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo | vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
                                vk::DebugUtilsMessageSeverityFlagBitsEXT::eError;
  create_info.messageType = vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral | vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance |
                            vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation;
  create_info.pfnUserCallback = debugCallback;
  create_info.pUserData = nullptr;

  return create_info;
}

vk::Instance Initialisers::createInstance(bool enableValidationLayers, std::vector<const char *> validationLayers, PFN_vkDebugUtilsMessengerCallbackEXT debugCallback) {
  if (enableValidationLayers && !checkValidationLayerSupport(validationLayers)) {
    throw std::runtime_error("validation layers requested, but not available!");
  }

  vk::ApplicationInfo app_info = {};

  app_info.pApplicationName = "Calypso";
  app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
  app_info.pEngineName = "No Engine";
  app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
  app_info.apiVersion = VK_API_VERSION_1_2;

  vk::InstanceCreateInfo create_info = {};

  create_info.pApplicationInfo = &app_info;

  auto extensions = getRequiredExtensions(enableValidationLayers);
  create_info.setPEnabledExtensionNames(extensions);

  vk::StructureChain<vk::DebugUtilsMessengerCreateInfoEXT> structure_chain = {};
  if (enableValidationLayers) {
    auto &debug_create_info = structure_chain.get<vk::DebugUtilsMessengerCreateInfoEXT>();
    debug_create_info = getDebugMessengerCreateInfo(debugCallback);

    create_info.setPEnabledLayerNames(validationLayers);
    create_info.setPNext(&structure_chain);
  }

  vk::Instance instance;
  vk::Result result = vk::createInstance(&create_info, nullptr, &instance);
  if (result != vk::Result::eSuccess) {
    throw std::runtime_error("failed to create instance!");
  }

  // initialize function pointers for instance
  VULKAN_HPP_DEFAULT_DISPATCHER.init(instance);
  
  return instance;
}

vk::DebugUtilsMessengerEXT Initialisers::setupDebugMessenger(const vk::Instance &instance, bool enableValidationLayers, PFN_vkDebugUtilsMessengerCallbackEXT debugCallback) {
  vk::DebugUtilsMessengerEXT debug_messenger;
  if (!enableValidationLayers) return debug_messenger;
  
  auto debug_create_info = getDebugMessengerCreateInfo(debugCallback);
  if (instance.createDebugUtilsMessengerEXT(&debug_create_info, nullptr, &debug_messenger) != vk::Result::eSuccess) {
    throw std::runtime_error("failed to set up debug messenger!");
  }

  return debug_messenger;
}

}  // namespace rendering