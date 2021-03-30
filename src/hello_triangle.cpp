//
// Created by Wyoming on 22/03/2021.
//

#include "hello_triangle.h"

HelloTriangle::HelloTriangle() {
  initDispatchLoader();
  initWindow();
  initVulkan();
}

void HelloTriangle::mainloop() {
  while (!glfwWindowShouldClose(window_)) {
    glfwPollEvents();
    drawFrame();
  }

  device_.waitIdle();
}

HelloTriangle::~HelloTriangle() {
  cleanupSwapChain();

  vmaDestroyBuffer(allocator_, vertexBuffer_, vertexBufferAllocation_);
  vmaDestroyAllocator(allocator_);

  for (std::size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
    device_.destroySemaphore(renderFinishedSemaphores_[i], nullptr);
    device_.destroySemaphore(imageAvailableSemaphores_[i], nullptr);
    device_.destroyFence(inFlightFences_[i], nullptr);
  }

  device_.destroyCommandPool(graphicsCommandPool_, nullptr);
  device_.destroyCommandPool(transferCommandPool_, nullptr);
  device_.destroy(nullptr);

  if (ENABLE_VALIDATION_LAYERS) {
    instance_.destroyDebugUtilsMessengerEXT(debugMessenger_, nullptr);
  }

  instance_.destroySurfaceKHR(surface_, nullptr);
  instance_.destroy();

  glfwDestroyWindow(window_);
  glfwTerminate();
}


void HelloTriangle::initDispatchLoader() {
  auto vk_get_instance_proc_addr = dynamicLoader_.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
  VULKAN_HPP_DEFAULT_DISPATCHER.init(vk_get_instance_proc_addr);
}

void HelloTriangle::framebufferResizeCallback(GLFWwindow *window, int, int) {
  auto *self = reinterpret_cast<HelloTriangle *>(glfwGetWindowUserPointer(window));
  self->framebufferResized_ = true;
}

void HelloTriangle::initWindow() {
  glfwInit();
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

  window_ = glfwCreateWindow(WIDTH, HEIGHT, "Calypso", nullptr, nullptr);
  glfwSetWindowUserPointer(window_, this);
  glfwSetFramebufferSizeCallback(window_, HelloTriangle::framebufferResizeCallback);
}

void HelloTriangle::initVulkan() {
  createInstance();
  setupDebugMessenger();
  createSurface();
  pickPhysicalDevice();
  createLogicalDevice();
  createAllocator();
  createSwapChain();
  createImageViews();
  createRenderPass();
  createGraphicsPipeline();
  createFramebuffers();
  createCommandPools();
  createVertexBuffer();
  createCommandBuffers();
  createSyncObjects();
}

bool HelloTriangle::checkValidationLayerSupport() {
  auto available_layers = vk::enumerateInstanceLayerProperties();

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

  vk::ApplicationInfo app_info;

  app_info.pApplicationName = "Calypso";
  app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
  app_info.pEngineName = "No Engine";
  app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
  app_info.apiVersion = VK_API_VERSION_1_0;

  vk::InstanceCreateInfo create_info;

  create_info.pApplicationInfo = &app_info;

  auto extensions = getRequiredExtensions();
  create_info.setPEnabledExtensionNames(extensions);

  vk::StructureChain<vk::DebugUtilsMessengerCreateInfoEXT> structure_chain;
  if (ENABLE_VALIDATION_LAYERS) {
    auto &debug_create_info = structure_chain.get<vk::DebugUtilsMessengerCreateInfoEXT>();
    debug_create_info = getDebugMessengerCreateInfo();

    create_info.setPEnabledLayerNames(validationLayers_);
    create_info.setPNext(&structure_chain);
  }

  vk::Result result = vk::createInstance(&create_info, nullptr, &instance_);
  if (result != vk::Result::eSuccess) {
    throw std::runtime_error("failed to create instance!");
  }

  // initialize function pointers for instance
  VULKAN_HPP_DEFAULT_DISPATCHER.init(instance_);
}

VKAPI_ATTR vk::Bool32 VKAPI_CALL HelloTriangle::debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                                                              VkDebugUtilsMessageTypeFlagsEXT,
                                                              const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData, void *) {
  auto logger = spdlog::get("validation_layer");
  switch (messageSeverity) {
    case VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT:
      logger->debug(pCallbackData->pMessage);
      break;
    case VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT:
      logger->info(pCallbackData->pMessage);
      break;
    case VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT:
      logger->warn(pCallbackData->pMessage);
      break;
    case VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT:
      logger->error(pCallbackData->pMessage);
      break;
    default:
      break;  // can't happen
  }

  return VK_FALSE;
}

vk::DebugUtilsMessengerCreateInfoEXT HelloTriangle::getDebugMessengerCreateInfo() {
  vk::DebugUtilsMessengerCreateInfoEXT create_info;

  create_info.messageSeverity = vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose | vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo |
                                vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning | vk::DebugUtilsMessageSeverityFlagBitsEXT::eError;
  create_info.messageType = vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral | vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance |
                            vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation;
  create_info.pfnUserCallback = debugCallback;
  create_info.pUserData = nullptr;

  return create_info;
}

void HelloTriangle::setupDebugMessenger() {
  if (!ENABLE_VALIDATION_LAYERS) return;

  auto create_info = getDebugMessengerCreateInfo();

  if (instance_.createDebugUtilsMessengerEXT(&create_info, nullptr, &debugMessenger_) != vk::Result::eSuccess) {
    throw std::runtime_error("failed to set up debug messenger!");
  }
}

HelloTriangle::QueueFamilyIndices HelloTriangle::findQueueFamilies(vk::PhysicalDevice device) {
  QueueFamilyIndices indices;

  auto queue_families = device.getQueueFamilyProperties();

  uint32_t i = 0;
  for (const auto &queue_family : queue_families) {
    if (queue_family.queueFlags & vk::QueueFlagBits::eGraphics) {
      indices.graphicsFamily = i;
    }

    if (queue_family.queueFlags & vk::QueueFlagBits::eTransfer && !(queue_family.queueFlags & vk::QueueFlagBits::eGraphics)) {
      indices.transferFamily = i;
    }

    auto present_support = device.getSurfaceSupportKHR(i, surface_);

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

HelloTriangle::SwapChainSupportDetails HelloTriangle::querySwapChainSupport(vk::PhysicalDevice device) {
  SwapChainSupportDetails details;

  details.capabilities = device.getSurfaceCapabilitiesKHR(surface_);
  details.formats = device.getSurfaceFormatsKHR(surface_);
  details.presentModes = device.getSurfacePresentModesKHR(surface_);

  return details;
}

void HelloTriangle::createSurface() {
  VkSurfaceKHR c_surface = surface_;
  if (glfwCreateWindowSurface(instance_, window_, nullptr, &c_surface) != VK_SUCCESS) {
    throw std::runtime_error("failed to create window surface!");
  }

  // Remember to cast back to vk::SurfaceKHR so vulkan keeps the handle
  surface_ = vk::SurfaceKHR(c_surface);
}

void HelloTriangle::pickPhysicalDevice() {
  auto rate_device = [&](vk::PhysicalDevice device) -> uint32_t {
    auto device_properties = device.getProperties();
    auto device_features = device.getFeatures();

    if (!device_features.geometryShader) return 0;

    QueueFamilyIndices indices = findQueueFamilies(device);
    if (!indices.isComplete()) return 0;

    bool extensions_supported = checkDeviceExtensionSupport(device);
    if (!extensions_supported) return 0;

    SwapChainSupportDetails swap_chain_support = querySwapChainSupport(device);
    bool swap_chain_adequate = !swap_chain_support.formats.empty() && !swap_chain_support.presentModes.empty();
    if (!swap_chain_adequate) return 0;

    int score = 0;

    if (device_properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu) {
      score += 10000;
    }

    score += device_properties.limits.maxImageDimension2D;

    auto memory_props = device.getMemoryProperties();
    auto heaps = memory_props.memoryHeaps;

    for (const auto &heap : heaps) {
      if (heap.flags & vk::MemoryHeapFlagBits::eDeviceLocal) {
        score += heap.size;
      }
    }

    return score;
  };

  auto devices = instance_.enumeratePhysicalDevices();
  if (devices.empty()) {
    throw std::runtime_error("failed to find GPUs with Vulkan support!");
  }

  physicalDevice_ = *std::max_element(devices.begin(), devices.end(), [&](vk::PhysicalDevice d0, vk::PhysicalDevice d1) -> bool {
    return rate_device(d0) < rate_device(d1);
  });

  if (rate_device(physicalDevice_) == 0) {
    throw std::runtime_error("failed to find a suitable GPU!");
  }
}

bool HelloTriangle::checkDeviceExtensionSupport(vk::PhysicalDevice device) {
  auto available_extensions = device.enumerateDeviceExtensionProperties();
  std::set<std::string_view> required_extensions(deviceExtensions_.begin(), deviceExtensions_.end());

  for (const auto &extension : available_extensions) {
    required_extensions.erase(extension.extensionName);
  }

  return required_extensions.empty();
}

void HelloTriangle::createLogicalDevice() {
  QueueFamilyIndices indices = findQueueFamilies(physicalDevice_);

  std::vector<vk::DeviceQueueCreateInfo> queue_create_infos;
  std::set<uint32_t> unique_queue_families = {indices.graphicsFamily.value(), indices.presentFamily.value(),
                                              indices.transferFamily.value()};

  float queue_priority = 1.0f;
  for (uint32_t queue_family : unique_queue_families) {
    vk::DeviceQueueCreateInfo queue_create_info;
    queue_create_info.queueFamilyIndex = queue_family;
    queue_create_info.queueCount = 1;
    queue_create_info.pQueuePriorities = &queue_priority;
    queue_create_infos.emplace_back(queue_create_info);
  }

  vk::PhysicalDeviceFeatures device_features;
  vk::DeviceCreateInfo create_info;

  create_info.setQueueCreateInfos(queue_create_infos);
  create_info.setPEnabledFeatures(&device_features);

  create_info.setPEnabledExtensionNames(deviceExtensions_);

  if (ENABLE_VALIDATION_LAYERS) {
    create_info.setPEnabledLayerNames(validationLayers_);
  }

  if (physicalDevice_.createDevice(&create_info, nullptr, &device_) != vk::Result::eSuccess) {
    throw std::runtime_error("failed to create logical device!");
  }
  VULKAN_HPP_DEFAULT_DISPATCHER.init(device_);

  graphicsQueue_ = device_.getQueue(indices.graphicsFamily.value(), 0);
  presentQueue_ = device_.getQueue(indices.presentFamily.value(), 0);
  transferQueue_ = device_.getQueue(indices.transferFamily.value(), 0);
}

void HelloTriangle::createAllocator() {
  VmaAllocatorCreateInfo allocator_info = {};
  allocator_info.vulkanApiVersion = VK_API_VERSION_1_2;
  allocator_info.physicalDevice = physicalDevice_;
  allocator_info.device = device_;
  allocator_info.instance = instance_;

  vmaCreateAllocator(&allocator_info, &allocator_);
}

vk::SurfaceFormatKHR HelloTriangle::chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR> &availableFormats) {
  for (const auto &available_format : availableFormats) {
    if (available_format.format == vk::Format::eB8G8R8A8Srgb && available_format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
      return available_format;
    }
  }

  return availableFormats.at(0);
}

vk::PresentModeKHR HelloTriangle::chooseSwapPresentMode(const std::vector<vk::PresentModeKHR> &availablePresentModes) {
  if (std::find(availablePresentModes.begin(), availablePresentModes.end(), vk::PresentModeKHR::eImmediate) !=
      availablePresentModes.end()) {
    return vk::PresentModeKHR::eImmediate;
  }

  return vk::PresentModeKHR::eFifo;
}

vk::Extent2D HelloTriangle::chooseSwapExtent(const vk::SurfaceCapabilitiesKHR &capabilities) {
  if (capabilities.currentExtent.width != UINT32_MAX) {
    return capabilities.currentExtent;
  }

  int width;
  int height;
  glfwGetFramebufferSize(window_, &width, &height);

  vk::Extent2D actual_extent = {static_cast<uint32_t>(width), static_cast<uint32_t>(height)};

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

  vk::SwapchainCreateInfoKHR create_info;
  create_info.surface = surface_;

  auto surface_format = chooseSwapSurfaceFormat(swap_chain_support.formats);
  vk::Extent2D extent = chooseSwapExtent(swap_chain_support.capabilities);
  create_info.minImageCount = image_count;
  create_info.imageFormat = surface_format.format;
  create_info.imageColorSpace = surface_format.colorSpace;
  create_info.imageExtent = extent;
  create_info.imageArrayLayers = 1;
  create_info.imageUsage = vk::ImageUsageFlagBits::eColorAttachment;

  QueueFamilyIndices indices = findQueueFamilies(physicalDevice_);
  std::set<uint32_t> all_queue_family_indices = {indices.graphicsFamily.value(), indices.presentFamily.value(),
                                                indices.transferFamily.value()};
  std::vector<uint32_t> queue_family_indices(all_queue_family_indices.size());
  queue_family_indices.assign(all_queue_family_indices.begin(), all_queue_family_indices.end());

  create_info.imageSharingMode = vk::SharingMode::eConcurrent;
  create_info.setQueueFamilyIndices(queue_family_indices);

  create_info.preTransform = swap_chain_support.capabilities.currentTransform;
  create_info.compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque;

  auto present_mode = chooseSwapPresentMode(swap_chain_support.presentModes);
  create_info.presentMode = present_mode;
  create_info.clipped = VK_TRUE;

  create_info.oldSwapchain = nullptr;

  if (device_.createSwapchainKHR(&create_info, nullptr, &swapChain_) != vk::Result::eSuccess) {
    throw std::runtime_error("failed to create swap chain!");
  }

  swapChainImages_ = device_.getSwapchainImagesKHR(swapChain_);

  swapChainImageFormat_ = surface_format.format;
  swapChainExtent_ = extent;
}

void HelloTriangle::cleanupSwapChain() {
  for (auto &framebuffer : swapChainFramebuffers_) device_.destroyFramebuffer(framebuffer, nullptr);

  device_.freeCommandBuffers(graphicsCommandPool_, graphicsCommandBuffers_);

  device_.destroyPipeline(graphicsPipeline_, nullptr);
  device_.destroyPipelineLayout(pipelineLayout_, nullptr);

  device_.destroyRenderPass(renderPass_, nullptr);
  for (auto &image_view : swapChainImageViews_) device_.destroyImageView(image_view, nullptr);
  device_.destroySwapchainKHR(swapChain_, nullptr);
}

void HelloTriangle::recreateSwapChain() {
  int width = 0;
  int height = 0;
  glfwGetFramebufferSize(window_, &width, &height);
  while (width == 0 || height == 0) {
    glfwGetFramebufferSize(window_, &width, &height);
    glfwWaitEvents();
  }

  device_.waitIdle();

  cleanupSwapChain();

  createSwapChain();
  createImageViews();
  createRenderPass();
  createGraphicsPipeline();
  createFramebuffers();
  createCommandBuffers();
}

void HelloTriangle::createImageViews() {
  swapChainImageViews_.resize(swapChainImages_.size());
  std::size_t i = 0;
  for (auto &swap_chain_image : swapChainImages_) {
    vk::ImageViewCreateInfo create_info;

    create_info.image = swap_chain_image;

    create_info.viewType = vk::ImageViewType::e2D;
    create_info.format = swapChainImageFormat_;

    create_info.components.r = vk::ComponentSwizzle::eIdentity;
    create_info.components.g = vk::ComponentSwizzle::eIdentity;
    create_info.components.b = vk::ComponentSwizzle::eIdentity;
    create_info.components.a = vk::ComponentSwizzle::eIdentity;

    create_info.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
    create_info.subresourceRange.baseMipLevel = 0;
    create_info.subresourceRange.levelCount = 1;
    create_info.subresourceRange.baseArrayLayer = 0;
    create_info.subresourceRange.layerCount = 1;

    if (device_.createImageView(&create_info, nullptr, &swapChainImageViews_[i]) != vk::Result::eSuccess) {
      throw std::runtime_error("failed to create image views!");
    }

    i++;
  }
}

vk::ShaderModule HelloTriangle::createShaderModule(const std::vector<std::byte> &code) {
  vk::ShaderModuleCreateInfo create_info;

  create_info.codeSize = code.size();
  create_info.pCode = reinterpret_cast<const uint32_t *>(code.data());

  vk::ShaderModule shader_module;
  if (device_.createShaderModule(&create_info, nullptr, &shader_module) != vk::Result::eSuccess) {
    throw std::runtime_error("failed to create shader module!");
  }

  return shader_module;
}

void HelloTriangle::createRenderPass() {
  vk::RenderPassCreateInfo render_pass_info;

  vk::AttachmentDescription color_attachment;
  color_attachment.format = swapChainImageFormat_;
  color_attachment.samples = vk::SampleCountFlagBits::e1;

  color_attachment.loadOp = vk::AttachmentLoadOp::eClear;
  color_attachment.storeOp = vk::AttachmentStoreOp::eStore;

  color_attachment.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
  color_attachment.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;

  color_attachment.initialLayout = vk::ImageLayout::eUndefined;
  color_attachment.finalLayout = vk::ImageLayout::ePresentSrcKHR;

  vk::AttachmentReference color_attachment_ref;
  color_attachment_ref.attachment = 0;
  color_attachment_ref.layout = vk::ImageLayout::eColorAttachmentOptimal;

  render_pass_info.attachmentCount = 1;
  render_pass_info.pAttachments = &color_attachment;

  vk::SubpassDescription subpass;
  subpass.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;

  subpass.colorAttachmentCount = 1;
  subpass.pColorAttachments = &color_attachment_ref;

  render_pass_info.subpassCount = 1;
  render_pass_info.pSubpasses = &subpass;

  vk::SubpassDependency dependency;
  dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
  dependency.dstSubpass = 0;

  dependency.srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
  dependency.srcAccessMask = vk::AccessFlagBits::eNoneKHR;

  dependency.dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
  dependency.dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite;

  render_pass_info.dependencyCount = 1;
  render_pass_info.pDependencies = &dependency;

  if (device_.createRenderPass(&render_pass_info, nullptr, &renderPass_) != vk::Result::eSuccess) {
    throw std::runtime_error("failed to create render pass!");
  }
}

void HelloTriangle::createGraphicsPipeline() {
  auto vert_shader_code = util::File::readFile(std::filesystem::path(ROOT_DIRECTORY) / "shaders/compiled/triangle.vert.spv");
  auto frag_shader_code = util::File::readFile(std::filesystem::path(ROOT_DIRECTORY) / "shaders/compiled/triangle.frag.spv");

  vk::ShaderModule vert_shader_module = createShaderModule(vert_shader_code);
  vk::ShaderModule frag_shader_module = createShaderModule(frag_shader_code);

  vk::PipelineShaderStageCreateInfo vert_shader_stage_info;
  vert_shader_stage_info.stage = vk::ShaderStageFlagBits::eVertex;

  vert_shader_stage_info.module = vert_shader_module;
  vert_shader_stage_info.pName = "main";
  vert_shader_stage_info.pSpecializationInfo = nullptr;

  vk::PipelineShaderStageCreateInfo frag_shader_stage_info;
  frag_shader_stage_info.stage = vk::ShaderStageFlagBits::eFragment;
  frag_shader_stage_info.module = frag_shader_module;
  frag_shader_stage_info.pName = "main";

  constexpr uint32_t N_SHADER_STAGES = 2;
  vk::PipelineShaderStageCreateInfo shader_stages[N_SHADER_STAGES] = {vert_shader_stage_info, frag_shader_stage_info};

  auto binding_description = Vertex::getBindingDescription();
  auto attribute_descriptions = Vertex::getAttributeDescriptions();

  vk::PipelineVertexInputStateCreateInfo vertex_input_info;
  vertex_input_info.setVertexBindingDescriptions(binding_description);
  vertex_input_info.setVertexAttributeDescriptions(attribute_descriptions);

  vk::PipelineInputAssemblyStateCreateInfo input_assembly;
  input_assembly.topology = vk::PrimitiveTopology::eTriangleList;
  input_assembly.primitiveRestartEnable = VK_FALSE;

  vk::Viewport viewport;
  viewport.x = 0.0f;
  viewport.y = 0.0f;
  viewport.width = static_cast<float>(swapChainExtent_.width);
  viewport.height = static_cast<float>(swapChainExtent_.height);
  viewport.minDepth = 0.0f;
  viewport.maxDepth = 1.0f;

  vk::Rect2D scissor;
  scissor.offset = vk::Offset2D(0, 0);
  scissor.extent = swapChainExtent_;

  vk::PipelineViewportStateCreateInfo viewport_state;
  viewport_state.viewportCount = 1;
  viewport_state.pViewports = &viewport;
  viewport_state.scissorCount = 1;
  viewport_state.pScissors = &scissor;

  vk::PipelineRasterizationStateCreateInfo rasterizer{};
  rasterizer.depthClampEnable = VK_FALSE;

  rasterizer.rasterizerDiscardEnable = VK_FALSE;

  rasterizer.polygonMode = vk::PolygonMode::eFill;  // could be wireframe too
  rasterizer.lineWidth = 1.0f;

  rasterizer.cullMode = vk::CullModeFlagBits::eBack;
  rasterizer.frontFace = vk::FrontFace::eClockwise;

  rasterizer.depthBiasEnable = VK_FALSE;
  rasterizer.depthBiasConstantFactor = 0.0f;
  rasterizer.depthBiasClamp = 0.0f;
  rasterizer.depthBiasSlopeFactor = 0.0f;

  vk::PipelineMultisampleStateCreateInfo multisampling;
  multisampling.sampleShadingEnable = VK_FALSE;
  multisampling.rasterizationSamples = vk::SampleCountFlagBits::e1;
  multisampling.minSampleShading = 1.0f;
  multisampling.pSampleMask = nullptr;
  multisampling.alphaToCoverageEnable = VK_FALSE;
  multisampling.alphaToOneEnable = VK_FALSE;

  vk::PipelineColorBlendAttachmentState color_blend_attachment;
  color_blend_attachment.colorWriteMask =
      vk::ColorComponentFlagBits::eA | vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB;
  color_blend_attachment.blendEnable = VK_FALSE;
  color_blend_attachment.srcColorBlendFactor = vk::BlendFactor::eOne;
  color_blend_attachment.dstColorBlendFactor = vk::BlendFactor::eZero;
  color_blend_attachment.colorBlendOp = vk::BlendOp::eAdd;
  color_blend_attachment.srcAlphaBlendFactor = vk::BlendFactor::eOne;
  color_blend_attachment.dstAlphaBlendFactor = vk::BlendFactor::eZero;
  color_blend_attachment.alphaBlendOp = vk::BlendOp::eAdd;

  vk::PipelineColorBlendStateCreateInfo color_blending;
  color_blending.logicOpEnable = VK_FALSE;
  color_blending.logicOp = vk::LogicOp::eCopy;
  color_blending.attachmentCount = 1;
  color_blending.pAttachments = &color_blend_attachment;
  color_blending.blendConstants[0] = 0.0f;
  color_blending.blendConstants[1] = 0.0f;
  color_blending.blendConstants[2] = 0.0f;
  color_blending.blendConstants[3] = 0.0f;

  vk::DynamicState dynamic_states[] = {vk::DynamicState::eViewport, vk::DynamicState::eLineWidth};

  vk::PipelineDynamicStateCreateInfo dynamic_state;
  dynamic_state.dynamicStateCount = 2;
  dynamic_state.pDynamicStates = dynamic_states;

  vk::PipelineLayoutCreateInfo pipeline_layout_info;
  pipeline_layout_info.setLayoutCount = 0;
  pipeline_layout_info.pSetLayouts = nullptr;
  pipeline_layout_info.pushConstantRangeCount = 0;
  pipeline_layout_info.pPushConstantRanges = nullptr;

  if (device_.createPipelineLayout(&pipeline_layout_info, nullptr, &pipelineLayout_) != vk::Result::eSuccess) {
    throw std::runtime_error("failed to create pipeline layout!");
  }

  vk::GraphicsPipelineCreateInfo pipeline_info;
  pipeline_info.stageCount = N_SHADER_STAGES;
  pipeline_info.pStages = shader_stages;

  pipeline_info.pVertexInputState = &vertex_input_info;
  pipeline_info.pInputAssemblyState = &input_assembly;
  pipeline_info.pViewportState = &viewport_state;
  pipeline_info.pRasterizationState = &rasterizer;
  pipeline_info.pMultisampleState = &multisampling;
  pipeline_info.pDepthStencilState = nullptr;
  pipeline_info.pColorBlendState = &color_blending;
  pipeline_info.pDynamicState = nullptr;  // &dynamic_state;

  pipeline_info.layout = pipelineLayout_;

  pipeline_info.renderPass = renderPass_;
  pipeline_info.subpass = 0;

  pipeline_info.basePipelineHandle = nullptr;  // requires VK_PIPELINE_CREATE_DERIVATIVE_BIT
  pipeline_info.basePipelineIndex = -1;

  if (device_.createGraphicsPipelines(nullptr, 1, &pipeline_info, nullptr, &graphicsPipeline_) != vk::Result::eSuccess) {
    throw std::runtime_error("failed to create graphics pipeline!");
  }

  device_.destroyShaderModule(vert_shader_module, nullptr);
  device_.destroyShaderModule(frag_shader_module, nullptr);
}

void HelloTriangle::createFramebuffers() {
  swapChainFramebuffers_.resize(swapChainImageViews_.size());

  std::size_t i = 0;
  for (auto &swap_chain_image_view : swapChainImageViews_) {
    constexpr uint32_t N_ATTACHMENTS = 1;
    vk::ImageView attachments[N_ATTACHMENTS] = {
        swap_chain_image_view,
    };

    vk::FramebufferCreateInfo framebuffer_info{};
    framebuffer_info.renderPass = renderPass_;
    framebuffer_info.attachmentCount = N_ATTACHMENTS;
    framebuffer_info.pAttachments = attachments;
    framebuffer_info.width = swapChainExtent_.width;
    framebuffer_info.height = swapChainExtent_.height;
    framebuffer_info.layers = 1;

    if (device_.createFramebuffer(&framebuffer_info, nullptr, &swapChainFramebuffers_[i]) != vk::Result::eSuccess) {
      throw std::runtime_error("failed to create framebuffer!");
    }

    i++;
  }
}

void HelloTriangle::createCommandPools() {
  QueueFamilyIndices queue_family_indices = findQueueFamilies(physicalDevice_);

  vk::CommandPoolCreateInfo graphics_pool_info;
  graphics_pool_info.queueFamilyIndex = queue_family_indices.graphicsFamily.value();
  graphics_pool_info.flags = {};

  if (device_.createCommandPool(&graphics_pool_info, nullptr, &graphicsCommandPool_) != vk::Result::eSuccess) {
    throw std::runtime_error("failed to create graphics command pool!");
  }

  vk::CommandPoolCreateInfo transfer_pool_info;
  transfer_pool_info.queueFamilyIndex = queue_family_indices.transferFamily.value();
  transfer_pool_info.flags = {};

  if (device_.createCommandPool(&transfer_pool_info, nullptr, &transferCommandPool_) != vk::Result::eSuccess) {
    throw std::runtime_error("failed to create transfer command pool!");
  }
}

vk::Buffer HelloTriangle::createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties,
                                       VmaAllocation &allocation) {
  VkBufferCreateInfo buffer_info = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
  buffer_info.size = size;
  buffer_info.usage = usage.m_mask;

  VmaAllocationCreateInfo alloc_info = {};
  alloc_info.usage = VMA_MEMORY_USAGE_GPU_ONLY;
  alloc_info.requiredFlags = properties.m_mask;

  VkBuffer buffer;
  vmaCreateBuffer(allocator_, &buffer_info, &alloc_info, &buffer, &allocation, nullptr);

  return buffer;
}

void HelloTriangle::copyBuffer(vk::Buffer srcBuffer, vk::Buffer dstBuffer, vk::DeviceSize size) {
  vk::CommandBufferAllocateInfo alloc_info;
  alloc_info.level = vk::CommandBufferLevel::ePrimary;
  alloc_info.commandPool = transferCommandPool_;
  alloc_info.commandBufferCount = 1;

  vk::CommandBuffer command_buffer;
  if (device_.allocateCommandBuffers(&alloc_info, &command_buffer) != vk::Result::eSuccess) {
    throw std::runtime_error("failed to allocate command buffer!");
  }

  vk::CommandBufferBeginInfo begin_info;
  begin_info.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;

  if (command_buffer.begin(&begin_info) != vk::Result::eSuccess) {
    throw std::runtime_error("failed to begin command buffer!");
  }

  vk::BufferCopy copy_region;
  copy_region.srcOffset = 0;
  copy_region.dstOffset = 0;
  copy_region.size = size;

  command_buffer.copyBuffer(srcBuffer, dstBuffer, 1, &copy_region);

  command_buffer.end();

  vk::SubmitInfo submit_info;
  submit_info.setCommandBuffers(command_buffer);

  transferQueue_.submit(submit_info, nullptr);
  transferQueue_.waitIdle();

  device_.freeCommandBuffers(transferCommandPool_, command_buffer);
}

void HelloTriangle::createVertexBuffer() {
  vk::DeviceSize buffer_size = sizeof(Vertex) * vertices_.size();

  VmaAllocation staging_buffer_allocation;
  auto staging_buffer =
      createBuffer(buffer_size, vk::BufferUsageFlagBits::eTransferSrc,
                   vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, staging_buffer_allocation);

  void *data;
  if (vmaMapMemory(allocator_, staging_buffer_allocation, &data) != VK_SUCCESS) {
    throw std::runtime_error("failed to map vertex buffer memory!");
  }
  std::memcpy(data, vertices_.data(), buffer_size);
  vmaUnmapMemory(allocator_, staging_buffer_allocation);

  vertexBuffer_ = createBuffer(buffer_size, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer,
                               vk::MemoryPropertyFlagBits::eDeviceLocal, vertexBufferAllocation_);

  copyBuffer(staging_buffer, vertexBuffer_, buffer_size);

  vmaDestroyBuffer(allocator_, staging_buffer, staging_buffer_allocation);
}

void HelloTriangle::createCommandBuffers() {
  graphicsCommandBuffers_.resize(swapChainFramebuffers_.size());

  vk::CommandBufferAllocateInfo alloc_info;
  alloc_info.commandPool = graphicsCommandPool_;
  alloc_info.level = vk::CommandBufferLevel::ePrimary;
  alloc_info.commandBufferCount = static_cast<uint32_t>(graphicsCommandBuffers_.size());

  if (device_.allocateCommandBuffers(&alloc_info, graphicsCommandBuffers_.data()) != vk::Result::eSuccess) {
    throw std::runtime_error("failed to allocate command buffers!");
  }

  std::size_t i = 0;
  for (auto &command_buffer : graphicsCommandBuffers_) {
    vk::CommandBufferBeginInfo begin_info;
    begin_info.flags = {};
    begin_info.pInheritanceInfo = nullptr;

    if (command_buffer.begin(&begin_info) != vk::Result::eSuccess) {
      throw std::runtime_error("failed to begin recording command buffer!");
    }

    vk::RenderPassBeginInfo render_pass_info;
    render_pass_info.renderPass = renderPass_;
    render_pass_info.framebuffer = swapChainFramebuffers_.at(i);

    render_pass_info.renderArea.offset = vk::Offset2D(0, 0);
    render_pass_info.renderArea.extent = swapChainExtent_;

    vk::ClearColorValue clear_color_value;
    clear_color_value.setFloat32({0.0f, 0.0f, 0.0f, 1.0f});
    vk::ClearValue clear_color(clear_color_value);
    render_pass_info.clearValueCount = 1;
    render_pass_info.pClearValues = &clear_color;

    command_buffer.beginRenderPass(&render_pass_info, vk::SubpassContents::eInline);
    {
      command_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, graphicsPipeline_);

      vk::DeviceSize offset = 0;
      graphicsCommandBuffers_[i].bindVertexBuffers(0, vertexBuffer_, offset);

      command_buffer.draw(vertices_.size(), 1, 0, 0);
    }
    command_buffer.endRenderPass();

    command_buffer.end();

    i++;
  }
}

void HelloTriangle::createSyncObjects() {
  imagesInFlight_.resize(swapChainImages_.size(), nullptr);

  vk::SemaphoreCreateInfo semaphore_info;
  vk::FenceCreateInfo fence_info;
  fence_info.flags = vk::FenceCreateFlagBits::eSignaled;

  for (std::size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
    if (device_.createSemaphore(&semaphore_info, nullptr, &imageAvailableSemaphores_[i]) != vk::Result::eSuccess ||
        device_.createSemaphore(&semaphore_info, nullptr, &renderFinishedSemaphores_[i]) != vk::Result::eSuccess ||
        device_.createFence(&fence_info, nullptr, &inFlightFences_[i]) != vk::Result::eSuccess) {
      throw std::runtime_error("failed to create semaphores for a frame!");
    }
  }
}

void HelloTriangle::drawFrame() {
  if (device_.waitForFences(1, &inFlightFences_[currentFrame_], VK_TRUE, UINT64_MAX) != vk::Result::eSuccess) {
    throw std::runtime_error("wait for fences timed out");
  }

  uint32_t image_index;
  vk::Result result = device_.acquireNextImageKHR(swapChain_, UINT64_MAX, imageAvailableSemaphores_[currentFrame_], nullptr, &image_index);
  if (result == vk::Result::eErrorOutOfDateKHR) {
    recreateSwapChain();
    return;
  }
  if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR) {
    throw std::runtime_error("failed to acquire swap chain image!");
  }

  // Check if a previous frame is using this image (i.e. there is its fence to wait on)
  if (imagesInFlight_[image_index]) {
    if (device_.waitForFences(1, &imagesInFlight_[image_index], VK_TRUE, UINT64_MAX) != vk::Result::eSuccess) {
      throw std::runtime_error("wait for fences timed out");
    };
  }
  // Mark the image as now being in use by this frame
  imagesInFlight_[image_index] = inFlightFences_[currentFrame_];

  vk::SubmitInfo submit_info{};

  constexpr uint32_t N_WAIT_SEMAPHORES = 1;
  vk::Semaphore wait_semaphores[N_WAIT_SEMAPHORES] = {imageAvailableSemaphores_[currentFrame_]};
  vk::PipelineStageFlags wait_stages[N_WAIT_SEMAPHORES] = {vk::PipelineStageFlagBits::eColorAttachmentOutput};

  submit_info.waitSemaphoreCount = N_WAIT_SEMAPHORES;
  submit_info.pWaitSemaphores = wait_semaphores;
  submit_info.pWaitDstStageMask = wait_stages;

  submit_info.commandBufferCount = 1;
  submit_info.pCommandBuffers = &graphicsCommandBuffers_[image_index];

  constexpr uint32_t N_SIGNAL_SEMAPHORES = 1;
  vk::Semaphore signal_semaphores[N_SIGNAL_SEMAPHORES] = {renderFinishedSemaphores_[currentFrame_]};
  submit_info.signalSemaphoreCount = N_SIGNAL_SEMAPHORES;
  submit_info.pSignalSemaphores = signal_semaphores;

  if (device_.resetFences(1, &inFlightFences_[currentFrame_]) != vk::Result::eSuccess) {
    throw std::runtime_error("reset fences failed!");
  };

  if (graphicsQueue_.submit(1, &submit_info, inFlightFences_[currentFrame_]) != vk::Result::eSuccess) {
    throw std::runtime_error("failed to submit draw command buffer!");
  }

  vk::PresentInfoKHR present_info;

  present_info.waitSemaphoreCount = N_SIGNAL_SEMAPHORES;
  present_info.pWaitSemaphores = signal_semaphores;

  constexpr uint32_t N_SWAP_CHAIN = 1;
  vk::SwapchainKHR swap_chains[N_SWAP_CHAIN] = {swapChain_};

  present_info.swapchainCount = N_SWAP_CHAIN;
  present_info.pSwapchains = swap_chains;
  present_info.pImageIndices = &image_index;

  present_info.pResults = nullptr;  // for error checking in case of multiple swaps

  result = presentQueue_.presentKHR(&present_info);
  if (result == vk::Result::eErrorOutOfDateKHR || result == vk::Result::eSuboptimalKHR || framebufferResized_) {
    framebufferResized_ = false;
    recreateSwapChain();
  } else if (result != vk::Result::eSuccess) {
    throw std::runtime_error("failed to present swap chain image!");
  }

  currentFrame_ = (currentFrame_ + 1) % MAX_FRAMES_IN_FLIGHT;
}
