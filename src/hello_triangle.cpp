//
// Created by Wyoming on 22/03/2021.
//

#include "hello_triangle.h"
#include "util/vulkan.h"

void HelloTriangle::initWindow() {
  glfwInit();
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

  window_ = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
}

void HelloTriangle::initVulkan() {
  createInstance();
  setupDebugMessenger();
  createSurface();
  pickPhysicalDevice();
  createLogicalDevice();
  createSwapChain();
}

bool HelloTriangle::checkValidationLayerSupport() {
  uint32_t layer_count;
  vkEnumerateInstanceLayerProperties(&layer_count, nullptr);

  std::vector<VkLayerProperties> available_layers(layer_count);
  vkEnumerateInstanceLayerProperties(&layer_count, available_layers.data());

  for (const char *layer_name : validationLayers_) {
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

VKAPI_ATTR VkBool32 VKAPI_CALL HelloTriangle::debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                                                            VkDebugUtilsMessageTypeFlagsEXT messageType,
                                                            const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData, void *) {
  auto console = spdlog::get("console");
  switch (messageSeverity) {
    case VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT:
      console->debug("[validation layer]: {}\n", pCallbackData->pMessage);
      break;
    case VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT:
      console->info("[validation layer]: {}\n", pCallbackData->pMessage);
      break;
    case VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT:
      console->warn("[validation layer]: {}\n", pCallbackData->pMessage);
      break;
    case VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT:
      console->error("[validation layer]: {}\n", pCallbackData->pMessage);
      break;
  }

  return VK_FALSE;
}

VkDebugUtilsMessengerCreateInfoEXT HelloTriangle::getDebugMessengerCreateInfo() {
  VkDebugUtilsMessengerCreateInfoEXT create_info = {};

  create_info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
  create_info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
  create_info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                            VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
  create_info.pfnUserCallback = debugCallback;
  create_info.pUserData = nullptr;

  return create_info;
}

void HelloTriangle::setupDebugMessenger() {
  if (!ENABLE_VALIDATION_LAYERS) return;

  auto create_info = getDebugMessengerCreateInfo();

  if (util::Vulkan::CreateDebugUtilsMessengerEXT(instance_, &create_info, nullptr, &debugMessenger_) != VK_SUCCESS) {
    throw std::runtime_error("failed to set up debug messenger!");
  }
}

HelloTriangle::QueueFamilyIndices HelloTriangle::findQueueFamilies(VkPhysicalDevice device) {
  QueueFamilyIndices indices;

  uint32_t queue_family_count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, nullptr);

  std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
  vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, queue_families.data());

  int i = 0;
  for (const auto &queue_family : queue_families) {
    if (queue_family.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
      indices.graphicsFamily = i;
    }

    VkBool32 present_support = 0U;
    vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface_, &present_support);

    if (present_support) {
      indices.presentFamily = i;
    }

    if (indices.isComplete()) {
      break;
    }

    i++;
  }

  return indices;
}

HelloTriangle::SwapChainSupportDetails HelloTriangle::querySwapChainSupport(VkPhysicalDevice device) {
  SwapChainSupportDetails details;

  vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface_, &details.capabilities);

  uint32_t format_count;
  vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface_, &format_count, nullptr);

  if (format_count != 0) {
    details.formats.resize(format_count);
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface_, &format_count, details.formats.data());
  }

  uint32_t present_mode_count;
  vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface_, &present_mode_count, nullptr);

  if (present_mode_count != 0) {
    details.presentModes.resize(present_mode_count);
    vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface_, &present_mode_count, details.presentModes.data());
  }

  return details;
}

void HelloTriangle::createSurface() {
  if (glfwCreateWindowSurface(instance_, window_, nullptr, &surface_) != VK_SUCCESS) {
    throw std::runtime_error("failed to create window surface!");
  }
}

void HelloTriangle::pickPhysicalDevice() {
  auto rate_device = [&](VkPhysicalDevice device) -> uint32_t {
    VkPhysicalDeviceProperties device_properties;
    vkGetPhysicalDeviceProperties(device, &device_properties);

    VkPhysicalDeviceFeatures device_features;
    vkGetPhysicalDeviceFeatures(device, &device_features);

    if (!device_features.geometryShader) return 0;

    QueueFamilyIndices indices = findQueueFamilies(device);
    if (!indices.isComplete()) return 0;

    bool extensions_supported = checkDeviceExtensionSupport(device);
    if (!extensions_supported) return 0;

    SwapChainSupportDetails swap_chain_support = querySwapChainSupport(device);
    bool swap_chain_adequate = !swap_chain_support.formats.empty() && !swap_chain_support.presentModes.empty();
    if (!swap_chain_adequate) return 0;

    int score = 0;

    if (device_properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
      score += 10000;
    }

    score += device_properties.limits.maxImageDimension2D;

    VkPhysicalDeviceMemoryProperties memory_props;
    vkGetPhysicalDeviceMemoryProperties(device, &memory_props);

    auto *heaps_pointer = memory_props.memoryHeaps;
    auto heaps = std::vector<VkMemoryHeap>(heaps_pointer, heaps_pointer + memory_props.memoryHeapCount);

    for (const auto &heap : heaps) {
      if (heap.flags & VkMemoryHeapFlagBits::VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
        score += heap.size;
      }
    }

    return score;
  };

  uint32_t device_count = 0;
  vkEnumeratePhysicalDevices(instance_, &device_count, nullptr);

  if (device_count == 0) {
    throw std::runtime_error("failed to find GPUs with Vulkan support!");
  }

  std::vector<VkPhysicalDevice> devices(device_count);
  vkEnumeratePhysicalDevices(instance_, &device_count, devices.data());

  physicalDevice_ = *std::max_element(devices.begin(), devices.end(),
                                      [&](VkPhysicalDevice d0, VkPhysicalDevice d1) -> bool { return rate_device(d0) < rate_device(d1); });

  if (rate_device(physicalDevice_) == 0) {
    throw std::runtime_error("failed to find a suitable GPU!");
  }
}

bool HelloTriangle::checkDeviceExtensionSupport(VkPhysicalDevice device) {
  uint32_t extension_count;
  vkEnumerateDeviceExtensionProperties(device, nullptr, &extension_count, nullptr);

  std::vector<VkExtensionProperties> available_extensions(extension_count);
  vkEnumerateDeviceExtensionProperties(device, nullptr, &extension_count, available_extensions.data());

  std::set<std::string_view> required_extensions(deviceExtensions_.begin(), deviceExtensions_.end());

  for (const auto &extension : available_extensions) {
    required_extensions.erase(extension.extensionName);
  }

  return required_extensions.empty();
}

void HelloTriangle::createLogicalDevice() {
  QueueFamilyIndices indices = findQueueFamilies(physicalDevice_);

  std::vector<VkDeviceQueueCreateInfo> queue_create_infos;
  std::set<uint32_t> unique_queue_families = {indices.graphicsFamily.value(), indices.presentFamily.value()};

  float queue_priority = 1.0f;
  for (uint32_t queue_family : unique_queue_families) {
    VkDeviceQueueCreateInfo queue_create_info = {};
    queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_create_info.queueFamilyIndex = queue_family;
    queue_create_info.queueCount = 1;
    queue_create_info.pQueuePriorities = &queue_priority;
    queue_create_infos.push_back(queue_create_info);
  }

  VkDeviceQueueCreateInfo queue_create_info;
  queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  queue_create_info.queueFamilyIndex = indices.graphicsFamily.value();
  queue_create_info.queueCount = 1;

  queue_create_info.pQueuePriorities = &queue_priority;

  VkPhysicalDeviceFeatures device_features = {};
  VkDeviceCreateInfo create_info = {};
  create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

  create_info.pQueueCreateInfos = queue_create_infos.data();
  create_info.queueCreateInfoCount = static_cast<uint32_t>(queue_create_infos.size());
  create_info.pEnabledFeatures = &device_features;

  create_info.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions_.size());
  create_info.ppEnabledExtensionNames = deviceExtensions_.data();

  if (ENABLE_VALIDATION_LAYERS) {
    create_info.enabledLayerCount = static_cast<uint32_t>(validationLayers_.size());
    create_info.ppEnabledLayerNames = validationLayers_.data();
  } else {
    create_info.enabledLayerCount = 0;
  }

  if (vkCreateDevice(physicalDevice_, &create_info, nullptr, &device_) != VK_SUCCESS) {
    throw std::runtime_error("failed to create logical device!");
  }

  vkGetDeviceQueue(device_, indices.graphicsFamily.value(), 0, &graphicsQueue_);
  vkGetDeviceQueue(device_, indices.presentFamily.value(), 0, &presentQueue_);
}

VkSurfaceFormatKHR HelloTriangle::chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR> &availableFormats) {
  for (const auto &available_format : availableFormats) {
    if (available_format.format == VK_FORMAT_B8G8R8A8_SRGB && available_format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
      return available_format;
    }
  }

  return availableFormats.at(0);
}

VkPresentModeKHR HelloTriangle::chooseSwapPresentMode(const std::vector<VkPresentModeKHR> &availablePresentModes) {
  if (std::find(availablePresentModes.begin(), availablePresentModes.end(), VK_PRESENT_MODE_IMMEDIATE_KHR) != availablePresentModes.end()) {
    return VK_PRESENT_MODE_IMMEDIATE_KHR;
  }

  return VK_PRESENT_MODE_FIFO_KHR;
}

VkExtent2D HelloTriangle::chooseSwapExtent(const VkSurfaceCapabilitiesKHR &capabilities) {
  if (capabilities.currentExtent.width != UINT32_MAX) {
    return capabilities.currentExtent;
  }
  
  int width;
  int height;
  glfwGetFramebufferSize(window_, &width, &height);

  VkExtent2D actual_extent = {static_cast<uint32_t>(width), static_cast<uint32_t>(height)};

  actual_extent.width = std::max(capabilities.minImageExtent.width, std::min(capabilities.maxImageExtent.width, actual_extent.width));
  actual_extent.height = std::max(capabilities.minImageExtent.height, std::min(capabilities.maxImageExtent.height, actual_extent.height));

  return actual_extent;
}

void HelloTriangle::createSwapChain() {
  SwapChainSupportDetails swap_chain_support = querySwapChainSupport(physicalDevice_);

  uint32_t image_count = swap_chain_support.capabilities.minImageCount + 1;
  if (swap_chain_support.capabilities.maxImageCount > 0 && image_count > swap_chain_support.capabilities.maxImageCount) {
    image_count = swap_chain_support.capabilities.maxImageCount;
  }

  VkSwapchainCreateInfoKHR create_info = {};
  create_info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
  create_info.surface = surface_;

  VkSurfaceFormatKHR surface_format = chooseSwapSurfaceFormat(swap_chain_support.formats);
  VkExtent2D extent = chooseSwapExtent(swap_chain_support.capabilities);
  create_info.minImageCount = image_count;
  create_info.imageFormat = surface_format.format;
  create_info.imageColorSpace = surface_format.colorSpace;
  create_info.imageExtent = extent;
  create_info.imageArrayLayers = 1;
  create_info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

  QueueFamilyIndices indices = findQueueFamilies(physicalDevice_);
  uint32_t queue_family_indices[] = {indices.graphicsFamily.value(), indices.presentFamily.value()};

  if (indices.graphicsFamily != indices.presentFamily) {
    create_info.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
    create_info.queueFamilyIndexCount = 2;
    create_info.pQueueFamilyIndices = queue_family_indices;
  } else {
    create_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
  }
  
  create_info.preTransform = swap_chain_support.capabilities.currentTransform;
  create_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;

  VkPresentModeKHR present_mode = chooseSwapPresentMode(swap_chain_support.presentModes);
  create_info.presentMode = present_mode;
  create_info.clipped = VK_TRUE;
  
  create_info.oldSwapchain = VK_NULL_HANDLE;

  if (vkCreateSwapchainKHR(device_, &create_info, nullptr, &swapChain_) != VK_SUCCESS) {
    throw std::runtime_error("failed to create swap chain!");
  }
}

std::vector<const char *> HelloTriangle::getRequiredExtensions() {
  uint32_t glfw_extension_count = 0;
  const char **glfw_extensions;
  glfw_extensions = glfwGetRequiredInstanceExtensions(&glfw_extension_count);

  std::vector<const char *> extensions(glfw_extensions, glfw_extensions + glfw_extension_count);

  if (ENABLE_VALIDATION_LAYERS) {
    extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
  }

  return extensions;
}

void HelloTriangle::createInstance() {
  if (ENABLE_VALIDATION_LAYERS && !checkValidationLayerSupport()) {
    throw std::runtime_error("validation layers requested, but not available!");
  }

  VkApplicationInfo app_info = {};
  app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  app_info.pApplicationName = "Vulkan";
  app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
  app_info.pEngineName = "No Engine";
  app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
  app_info.apiVersion = VK_API_VERSION_1_0;

  VkInstanceCreateInfo create_info = {};

  create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  create_info.pApplicationInfo = &app_info;

  auto extensions = getRequiredExtensions();
  create_info.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
  create_info.ppEnabledExtensionNames = extensions.data();

  auto debug_create_info = getDebugMessengerCreateInfo();
  if (ENABLE_VALIDATION_LAYERS) {
    create_info.enabledLayerCount = static_cast<uint32_t>(validationLayers_.size());
    create_info.ppEnabledLayerNames = validationLayers_.data();

    create_info.pNext = static_cast<VkDebugUtilsMessengerCreateInfoEXT *>(&debug_create_info);
  } else {
    create_info.enabledLayerCount = 0;
    create_info.pNext = nullptr;
  }

  if (vkCreateInstance(&create_info, nullptr, &instance_) != VK_SUCCESS) {
    throw std::runtime_error("failed to create instance!");
  }
}

void HelloTriangle::mainLoop() {
  while (!glfwWindowShouldClose(window_)) {
    glfwPollEvents();
  }
}

void HelloTriangle::cleanup() {
  if (ENABLE_VALIDATION_LAYERS) {
    util::Vulkan::DestroyDebugUtilsMessengerEXT(instance_, debugMessenger_, nullptr);
  }

  vkDestroySurfaceKHR(instance_, surface_, nullptr);
  vkDestroyDevice(device_, nullptr);
  vkDestroyInstance(instance_, nullptr);

  glfwDestroyWindow(window_);
  glfwTerminate();
}
