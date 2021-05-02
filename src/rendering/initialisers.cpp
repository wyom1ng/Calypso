//
// Created by Wyoming on 11/04/2021.
//

#include "initialisers.h"
#include <filesystem>
#include <functional>
#include <set>
#include <stb_image.h>

namespace rendering {

void Initialisers::initDispatchLoader(const vk::DynamicLoader &dynamicLoader) {
  auto vk_get_instance_proc_addr = dynamicLoader.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
  VULKAN_HPP_DEFAULT_DISPATCHER.init(vk_get_instance_proc_addr);
}

GLFWwindow *Initialisers::createWindow(void *user, std::function<void(GLFWwindow *, int, int)> framebufferResizeCallback,
                                       uint16_t initialWidth, uint16_t initialHeight) {
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

vk::Instance Initialisers::createInstance(bool enableValidationLayers, const std::vector<const char *> &validationLayers, PFN_vkDebugUtilsMessengerCallbackEXT debugCallback) {
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

vk::SurfaceKHR Initialisers::createSurface(const vk::Instance &instance, GLFWwindow *window) {
  VkSurfaceKHR c_surface;
  if (glfwCreateWindowSurface(instance, window, nullptr, &c_surface) != VK_SUCCESS) {
    throw std::runtime_error("failed to create window surface!");
  }

  return c_surface;
}

type::SwapchainSupportDetails Initialisers::querySwapchainSupport(const vk::PhysicalDevice &device, const vk::SurfaceKHR &surface) {
  type::SwapchainSupportDetails details = {};

  details.capabilities = device.getSurfaceCapabilitiesKHR(surface);
  details.formats = device.getSurfaceFormatsKHR(surface);
  details.presentModes = device.getSurfacePresentModesKHR(surface);

  return details;
}

type::QueueFamilyIndices Initialisers::findQueueFamilies(const vk::PhysicalDevice &device, const vk::SurfaceKHR &surface) {
  type::QueueFamilyIndices indices;

  auto queue_families = device.getQueueFamilyProperties();

  uint32_t i = 0;
  for (const auto &queue_family : queue_families) {
    if (queue_family.queueFlags & vk::QueueFlagBits::eGraphics) {
      indices.graphicsFamily = i;
    }

    if (queue_family.queueFlags & vk::QueueFlagBits::eTransfer && !(queue_family.queueFlags & vk::QueueFlagBits::eGraphics)) {
      indices.transferFamily = i;
    }

    auto present_support = device.getSurfaceSupportKHR(i, surface);

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

vk::PhysicalDevice Initialisers::createPhysicalDevice(const vk::Instance &instance, const vk::SurfaceKHR &surface,
                                                      const std::vector<const char *> &deviceExtensions) {
  auto devices = instance.enumeratePhysicalDevices();
  if (devices.empty()) {
    throw std::runtime_error("failed to find GPUs with Vulkan support!");
  }

  auto physical_device = *std::max_element(devices.begin(), devices.end(), [&](vk::PhysicalDevice d0, vk::PhysicalDevice d1) -> bool {
    return rateDevice(d0, surface, deviceExtensions) < rateDevice(d1, surface, deviceExtensions);
  });

  if (rateDevice(physical_device, surface, deviceExtensions) == 0) {
    throw std::runtime_error("failed to find a suitable GPU!");
  }
  
  return physical_device;
}

vk::SampleCountFlagBits Initialisers::getMaxUsableSampleCount(const vk::PhysicalDevice &physicalDevice) {
  auto physical_device_properties = physicalDevice.getProperties();

  vk::SampleCountFlags counts =
      physical_device_properties.limits.framebufferColorSampleCounts & physical_device_properties.limits.framebufferDepthSampleCounts;
  if (counts & vk::SampleCountFlagBits::e64) {
    return vk::SampleCountFlagBits::e64;
  }
  if (counts & vk::SampleCountFlagBits::e32) {
    return vk::SampleCountFlagBits::e32;
  }
  if (counts & vk::SampleCountFlagBits::e16) {
    return vk::SampleCountFlagBits::e16;
  }
  if (counts & vk::SampleCountFlagBits::e8) {
    return vk::SampleCountFlagBits::e8;
  }
  if (counts & vk::SampleCountFlagBits::e4) {
    return vk::SampleCountFlagBits::e4;
  }
  if (counts & vk::SampleCountFlagBits::e4) {
    return vk::SampleCountFlagBits::e4;
  }

  return vk::SampleCountFlagBits::e1;
}

vk::Device Initialisers::createLogicalDevice(const vk::PhysicalDevice &physicalDevice, const vk::SurfaceKHR &surface,
                                             bool enableValidationLayers, const std::vector<const char *> &validationLayers,
                                             const std::vector<const char *> &deviceExtensions) {
  auto indices = Initialisers::findQueueFamilies(physicalDevice, surface);

  std::vector<vk::DeviceQueueCreateInfo> queue_create_infos = {};
  std::set<uint32_t> unique_queue_families = {indices.graphicsFamily.value(), indices.presentFamily.value(),
                                              indices.transferFamily.value()};

  float queue_priority = 1.0f;
  for (uint32_t queue_family : unique_queue_families) {
    vk::DeviceQueueCreateInfo queue_create_info = {};
    queue_create_info.queueFamilyIndex = queue_family;
    queue_create_info.setQueuePriorities(queue_priority);
    queue_create_infos.emplace_back(queue_create_info);
  }

  vk::PhysicalDeviceFeatures device_features = {};
  device_features.samplerAnisotropy = VK_TRUE;
  device_features.fillModeNonSolid = VK_TRUE;
  device_features.sampleRateShading = VK_TRUE;

  vk::DeviceCreateInfo create_info = {};

  create_info.setQueueCreateInfos(queue_create_infos);
  create_info.setPEnabledFeatures(&device_features);

  if (enableValidationLayers) {
    create_info.setPEnabledLayerNames(validationLayers);
  }

  create_info.setPEnabledExtensionNames(deviceExtensions);

  vk::Device device;
  if (physicalDevice.createDevice(&create_info, nullptr, &device) != vk::Result::eSuccess) {
    throw std::runtime_error("failed to create logical device!");
  }
  VULKAN_HPP_DEFAULT_DISPATCHER.init(device);
  
  return device;
}

std::array<vk::Queue, 3> Initialisers::createQueues(const vk::PhysicalDevice &physicalDevice, const vk::SurfaceKHR &surface, const vk::Device &device) {
  auto indices = Initialisers::findQueueFamilies(physicalDevice, surface);

  auto graphics_queue = device.getQueue(indices.graphicsFamily.value(), 0);
  auto present_queue = device.getQueue(indices.presentFamily.value(), 0);
  auto transfer_queue = device.getQueue(indices.transferFamily.value(), 0);
  
  return {graphics_queue, present_queue, transfer_queue};
}

VmaAllocator Initialisers::createAllocator(const vk::Instance &instance, const vk::PhysicalDevice &physicalDevice,
                                           const vk::Device &device) {
  VmaAllocatorCreateInfo allocator_info = {};
  allocator_info.vulkanApiVersion = VK_API_VERSION_1_2;
  allocator_info.instance = instance;
  allocator_info.physicalDevice = physicalDevice;
  allocator_info.device = device;

  VmaAllocator allocator;
  vmaCreateAllocator(&allocator_info, &allocator);

  return allocator;
}

type::SwapchainData Initialisers::createSwapchain(const vk::PhysicalDevice &physicalDevice, const vk::SurfaceKHR &surface,
                                                          const vk::Device &device, GLFWwindow *window) {
  auto swap_chain_support = Initialisers::querySwapchainSupport(physicalDevice, surface);

  uint32_t image_count = swap_chain_support.capabilities.minImageCount + 1;
  if (swap_chain_support.capabilities.maxImageCount > 0 && image_count > swap_chain_support.capabilities.maxImageCount) {
    image_count = swap_chain_support.capabilities.maxImageCount;
  }

  vk::SwapchainCreateInfoKHR create_info = {};
  create_info.surface = surface;

  auto surface_format = chooseSwapchainSurfaceFormat(swap_chain_support.formats);
  vk::Extent2D extent = chooseSwapchainExtent(swap_chain_support.capabilities, window);
  create_info.minImageCount = image_count;
  create_info.imageFormat = surface_format.format;
  create_info.imageColorSpace = surface_format.colorSpace;
  create_info.imageExtent = extent;
  create_info.imageArrayLayers = 1;
  create_info.imageUsage = vk::ImageUsageFlagBits::eColorAttachment;

  auto indices = Initialisers::findQueueFamilies(physicalDevice, surface);
  std::set<uint32_t> all_queue_family_indices = {indices.graphicsFamily.value(), indices.presentFamily.value(),
                                                 indices.transferFamily.value()};
  std::vector<uint32_t> queue_family_indices(all_queue_family_indices.size());
  queue_family_indices.assign(all_queue_family_indices.begin(), all_queue_family_indices.end());

  create_info.imageSharingMode = vk::SharingMode::eConcurrent;
  create_info.setQueueFamilyIndices(queue_family_indices);

  create_info.preTransform = swap_chain_support.capabilities.currentTransform;
  create_info.compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque;

  auto present_mode = chooseSwapchainPresentMode(swap_chain_support.presentModes);
  create_info.presentMode = present_mode;
  create_info.clipped = VK_TRUE;

  create_info.oldSwapchain = nullptr;

  vk::SwapchainKHR swapchain;
  if (device.createSwapchainKHR(&create_info, nullptr, &swapchain) != vk::Result::eSuccess) {
    throw std::runtime_error("failed to create swap chain!");
  }

  const std::vector<vk::Image, std::allocator<vk::Image>> &images = device.getSwapchainImagesKHR(swapchain);
  vk::Format format = surface_format.format;
  return {
      .swapchain = swapchain,
      .images = images,
      .imageViews = createSwapchainImageViews(device, images, format),
      .format = format,
      .extent = extent,
  };
}

vk::RenderPass Initialisers::createRenderPass(const vk::Device &device, const vk::Format &colourFormat, const vk::SampleCountFlagBits &colourSampleCount,
                                              const vk::Format &depthFormat, const vk::SampleCountFlagBits &depthSampleCount) {
  vk::AttachmentDescription color_attachment = {};
  color_attachment.format = colourFormat;
  color_attachment.samples = colourSampleCount;
  color_attachment.loadOp = vk::AttachmentLoadOp::eClear;
  color_attachment.storeOp = vk::AttachmentStoreOp::eStore;
  color_attachment.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
  color_attachment.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
  color_attachment.initialLayout = vk::ImageLayout::eUndefined;
  color_attachment.finalLayout = vk::ImageLayout::eColorAttachmentOptimal;

  vk::AttachmentReference color_attachment_ref = {};
  color_attachment_ref.attachment = 0;
  color_attachment_ref.layout = vk::ImageLayout::eColorAttachmentOptimal;

  vk::AttachmentDescription depth_attachment = {};
  depth_attachment.format = depthFormat;
  depth_attachment.samples = depthSampleCount;
  depth_attachment.loadOp = vk::AttachmentLoadOp::eClear;
  depth_attachment.storeOp = vk::AttachmentStoreOp::eDontCare;
  depth_attachment.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
  depth_attachment.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
  depth_attachment.initialLayout = vk::ImageLayout::eUndefined;
  depth_attachment.finalLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal;

  vk::AttachmentReference depth_attachment_ref = {};
  depth_attachment_ref.attachment = 1;
  depth_attachment_ref.layout = vk::ImageLayout::eDepthStencilAttachmentOptimal;

  vk::AttachmentDescription color_attachment_resolve = {};
  color_attachment_resolve.format = colourFormat;
  color_attachment_resolve.samples = vk::SampleCountFlagBits::e1;
  color_attachment_resolve.loadOp = vk::AttachmentLoadOp::eDontCare;
  color_attachment_resolve.storeOp = vk::AttachmentStoreOp::eStore;
  color_attachment_resolve.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
  color_attachment_resolve.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
  color_attachment_resolve.initialLayout = vk::ImageLayout::eUndefined;
  color_attachment_resolve.finalLayout = vk::ImageLayout::ePresentSrcKHR;

  vk::AttachmentReference color_attachment_resolve_ref = {};
  color_attachment_resolve_ref.attachment = 2;
  color_attachment_resolve_ref.layout = vk::ImageLayout::eColorAttachmentOptimal;

  vk::SubpassDescription subpass = {};
  subpass.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
  subpass.setColorAttachments(color_attachment_ref);
  subpass.setPDepthStencilAttachment(&depth_attachment_ref);
  subpass.setResolveAttachments(color_attachment_resolve_ref);

  vk::SubpassDependency dependency = {};
  dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
  dependency.dstSubpass = 0;
  dependency.srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput | vk::PipelineStageFlagBits::eEarlyFragmentTests;
  dependency.srcAccessMask = vk::AccessFlagBits::eNoneKHR;
  dependency.dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput | vk::PipelineStageFlagBits::eEarlyFragmentTests;
  dependency.dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite | vk::AccessFlagBits::eDepthStencilAttachmentWrite;

  std::array<vk::AttachmentDescription, 3> attachments = {color_attachment, depth_attachment, color_attachment_resolve};
  vk::RenderPassCreateInfo render_pass_info = {};
  render_pass_info.setAttachments(attachments);
  render_pass_info.setSubpasses(subpass);
  render_pass_info.setDependencies(dependency);

  vk::RenderPass render_pass;
  if (device.createRenderPass(&render_pass_info, nullptr, &render_pass) != vk::Result::eSuccess) {
    throw std::runtime_error("failed to create render pass!");
  }
  
  return render_pass;
}

vk::DescriptorSetLayout Initialisers::createDescriptorSetLayout(const vk::Device &device) {
  vk::DescriptorSetLayoutBinding ubo_layout_binding = {};
  ubo_layout_binding.binding = 0;
  ubo_layout_binding.descriptorCount = 1;
  ubo_layout_binding.descriptorType = vk::DescriptorType::eUniformBuffer;
  ubo_layout_binding.stageFlags = vk::ShaderStageFlagBits::eVertex;

  vk::DescriptorSetLayoutBinding sampler_layout_binding = {};
  sampler_layout_binding.binding = 1;
  sampler_layout_binding.descriptorCount = 1;
  sampler_layout_binding.descriptorType = vk::DescriptorType::eCombinedImageSampler;
  sampler_layout_binding.stageFlags = vk::ShaderStageFlagBits::eFragment;

  std::array<vk::DescriptorSetLayoutBinding, 2> bindings = {ubo_layout_binding, sampler_layout_binding};
  vk::DescriptorSetLayoutCreateInfo layout_info = {};
  layout_info.setBindings(bindings);

  vk::DescriptorSetLayout descriptor_set_layout;
  if (device.createDescriptorSetLayout(&layout_info, nullptr, &descriptor_set_layout) != vk::Result::eSuccess) {
    throw std::runtime_error("failed to create descriptor set layout!");
  }
  
  return descriptor_set_layout;
}

bool Initialisers::checkDeviceExtensionSupport(const vk::PhysicalDevice &device, const std::vector<const char *> &deviceExtensions) {
  auto available_extensions = device.enumerateDeviceExtensionProperties();
  std::set<std::string_view> required_extensions(deviceExtensions.begin(), deviceExtensions.end());

  for (const auto &extension : available_extensions) {
    required_extensions.erase(extension.extensionName);
  }

  return required_extensions.empty();
}

uint32_t Initialisers::rateDevice(const vk::PhysicalDevice &physicalDevice, const vk::SurfaceKHR &surface, const std::vector<const char *> &deviceExtensions) {
  auto device_properties = physicalDevice.getProperties();
  auto device_features = physicalDevice.getFeatures();

  if (!device_features.geometryShader) return 0;

  auto indices = findQueueFamilies(physicalDevice, surface);
  if (!indices.isComplete()) return 0;

  bool extensions_supported = checkDeviceExtensionSupport(physicalDevice, deviceExtensions);
  if (!extensions_supported) return 0;

  auto swap_chain_support = querySwapchainSupport(physicalDevice, surface);
  bool swap_chain_adequate = !swap_chain_support.formats.empty() && !swap_chain_support.presentModes.empty();
  if (!swap_chain_adequate) return 0;

  auto supported_features = physicalDevice.getFeatures();
  if (!supported_features.samplerAnisotropy) return 0;

  uint32_t score = 0;

  if (device_properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu) {
    score += 10000;
  }

  score += device_properties.limits.maxImageDimension2D;

  auto memory_props = physicalDevice.getMemoryProperties();
  auto heaps = memory_props.memoryHeaps;

  for (const auto &heap : heaps) {
    if (heap.flags & vk::MemoryHeapFlagBits::eDeviceLocal) {
      score += heap.size;
    }
  }

  return score;
}

vk::SurfaceFormatKHR Initialisers::chooseSwapchainSurfaceFormat(const std::vector<vk::SurfaceFormatKHR> &availableFormats) {
  for (const auto &available_format : availableFormats) {
    if (available_format.format == vk::Format::eB8G8R8A8Srgb && available_format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
      return available_format;
    }
  }

  return availableFormats.at(0);
}

vk::PresentModeKHR Initialisers::chooseSwapchainPresentMode(const std::vector<vk::PresentModeKHR> &availablePresentModes) {
  if (std::find(availablePresentModes.begin(), availablePresentModes.end(), vk::PresentModeKHR::eImmediate) !=
      availablePresentModes.end()) {
    return vk::PresentModeKHR::eImmediate;
  }

  return vk::PresentModeKHR::eFifo;
}

vk::Extent2D Initialisers::chooseSwapchainExtent(const vk::SurfaceCapabilitiesKHR &capabilities, GLFWwindow *window) {
  if (capabilities.currentExtent.width != UINT32_MAX) {
    return capabilities.currentExtent;
  }

  int width;
  int height;
  glfwGetFramebufferSize(window, &width, &height);

  vk::Extent2D actual_extent = {static_cast<uint32_t>(width), static_cast<uint32_t>(height)};

  actual_extent.width = std::max(capabilities.minImageExtent.width, std::min(capabilities.maxImageExtent.width, actual_extent.width));
  actual_extent.height = std::max(capabilities.minImageExtent.height, std::min(capabilities.maxImageExtent.height, actual_extent.height));

  return actual_extent;
}

std::vector<vk::ImageView> Initialisers::createSwapchainImageViews(const vk::Device &device, const std::vector<vk::Image> &images, const vk::Format &format) {
  std::vector<vk::ImageView> image_views(images.size());

  std::size_t i = 0;
  for (const auto &swap_chain_image : images) {
    image_views[i] = createImageView(device, swap_chain_image, format, vk::ImageAspectFlagBits::eColor, 1);
    i++;
  }
  
  return image_views;
}

vk::ImageView Initialisers::createImageView(const vk::Device &device, const vk::Image &image, const vk::Format &format,
                                            vk::ImageAspectFlags aspectFlags, uint32_t mipLevels) {
  vk::ImageViewCreateInfo view_info = {};
  view_info.image = image;
  view_info.format = format;

  view_info.viewType = vk::ImageViewType::e2D;
  view_info.subresourceRange.aspectMask = aspectFlags;
  view_info.subresourceRange.baseMipLevel = 0;
  view_info.subresourceRange.levelCount = mipLevels;
  view_info.subresourceRange.baseArrayLayer = 0;
  view_info.subresourceRange.layerCount = 1;

  vk::ImageView image_view;
  if (device.createImageView(&view_info, nullptr, &image_view) != vk::Result::eSuccess) {
    throw std::runtime_error("failed to create texture image view!");
  }

  return image_view;
}

}  // namespace rendering