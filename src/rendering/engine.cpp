﻿//
// Created by Wyoming on 11/04/2021.
//

#include "engine.h"

namespace rendering {

Engine::Engine() {
  Initialisers::initDispatchLoader(dynamicLoader_);
  window_ = Initialisers::createWindow(this, Engine::framebufferResizeCallback, WIDTH, HEIGHT);
  initVulkan();
}

void Engine::mainloop() {
  startTime_ = std::chrono::high_resolution_clock::now();

  while (!glfwWindowShouldClose(window_)) {
    glfwPollEvents();
    drawFrame();
  }

  device_.waitIdle();
}

Engine::~Engine() {
  cleanupSwapChain();

  device_.destroyDescriptorSetLayout(descriptorSetLayout_, nullptr);

  vmaDestroyBuffer(allocator_, vertexBuffer_, vertexBufferAllocation_);
  vmaDestroyBuffer(allocator_, indexBuffer_, indexBufferAllocation_);

  device_.destroyImageView(textureImageView_, nullptr);
  device_.destroySampler(textureSampler_, nullptr);
  vmaDestroyImage(allocator_, textureImage_, textureImageAllocation_);

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

void Engine::framebufferResizeCallback(GLFWwindow *window, int, int) {
  auto *self = reinterpret_cast<Engine *>(glfwGetWindowUserPointer(window));
  self->framebufferResized_ = true;
}

void Engine::initVulkan() {
  instance_ = Initialisers::createInstance(ENABLE_VALIDATION_LAYERS, validationLayers_, Engine::debugCallback);
  debugMessenger_ = Initialisers::setupDebugMessenger(instance_, ENABLE_VALIDATION_LAYERS, Engine::debugCallback);
  createSurface();
  pickPhysicalDevice();
  createLogicalDevice();
  createAllocator();
  createSwapChain();
  createImageViews();
  createRenderPass();
  createDescriptorSetLayout();
  createGraphicsPipeline();
  createCommandPools();
  createColourResources();
  createDepthResources();
  createFramebuffers();
  createTextureImage();
  createTextureImageView();
  createTextureSampler();
  loadModel();
  createVertexBuffer();
  createIndexBuffer();
  createUniformBuffers();
  createDescriptorPool();
  createDescriptorSets();
  createCommandBuffers();
  createSyncObjects();
}


VKAPI_ATTR vk::Bool32 VKAPI_CALL Engine::debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
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

Engine::QueueFamilyIndices Engine::findQueueFamilies(vk::PhysicalDevice device) {
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

Engine::SwapChainSupportDetails Engine::querySwapChainSupport(vk::PhysicalDevice device) {
  SwapChainSupportDetails details = {};

  details.capabilities = device.getSurfaceCapabilitiesKHR(surface_);
  details.formats = device.getSurfaceFormatsKHR(surface_);
  details.presentModes = device.getSurfacePresentModesKHR(surface_);

  return details;
}

void Engine::createSurface() {
  VkSurfaceKHR c_surface = surface_;
  if (glfwCreateWindowSurface(instance_, window_, nullptr, &c_surface) != VK_SUCCESS) {
    throw std::runtime_error("failed to create window surface!");
  }

  // Remember to cast back to vk::SurfaceKHR so vulkan keeps the handle
  surface_ = vk::SurfaceKHR(c_surface);
}

vk::SampleCountFlagBits Engine::getMaxUsableSampleCount() {
  auto physical_device_properties = physicalDevice_.getProperties();

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

void Engine::pickPhysicalDevice() {
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

    auto supported_features = device.getFeatures();
    if (!supported_features.samplerAnisotropy) return 0;

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

  sampleCount_ = getMaxUsableSampleCount();
}

bool Engine::checkDeviceExtensionSupport(vk::PhysicalDevice device) {
  auto available_extensions = device.enumerateDeviceExtensionProperties();
  std::set<std::string_view> required_extensions(deviceExtensions_.begin(), deviceExtensions_.end());

  for (const auto &extension : available_extensions) {
    required_extensions.erase(extension.extensionName);
  }

  return required_extensions.empty();
}

void Engine::createLogicalDevice() {
  QueueFamilyIndices indices = findQueueFamilies(physicalDevice_);

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

void Engine::createAllocator() {
  VmaAllocatorCreateInfo allocator_info = {};
  allocator_info.vulkanApiVersion = VK_API_VERSION_1_2;
  allocator_info.physicalDevice = physicalDevice_;
  allocator_info.device = device_;
  allocator_info.instance = instance_;

  vmaCreateAllocator(&allocator_info, &allocator_);
}

vk::SurfaceFormatKHR Engine::chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR> &availableFormats) {
  for (const auto &available_format : availableFormats) {
    if (available_format.format == vk::Format::eB8G8R8A8Srgb && available_format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
      return available_format;
    }
  }

  return availableFormats.at(0);
}

vk::PresentModeKHR Engine::chooseSwapPresentMode(const std::vector<vk::PresentModeKHR> &availablePresentModes) {
  if (std::find(availablePresentModes.begin(), availablePresentModes.end(), vk::PresentModeKHR::eImmediate) !=
      availablePresentModes.end()) {
    return vk::PresentModeKHR::eImmediate;
  }

  return vk::PresentModeKHR::eFifo;
}

vk::Extent2D Engine::chooseSwapExtent(const vk::SurfaceCapabilitiesKHR &capabilities) {
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

void Engine::createSwapChain() {
  SwapChainSupportDetails swap_chain_support = querySwapChainSupport(physicalDevice_);

  uint32_t image_count = swap_chain_support.capabilities.minImageCount + 1;
  if (swap_chain_support.capabilities.maxImageCount > 0 && image_count > swap_chain_support.capabilities.maxImageCount) {
    image_count = swap_chain_support.capabilities.maxImageCount;
  }

  vk::SwapchainCreateInfoKHR create_info = {};
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

void Engine::cleanupSwapChain() {
  for (auto &framebuffer : swapChainFramebuffers_) device_.destroyFramebuffer(framebuffer, nullptr);

  for (std::size_t i = 0; i < swapChainImages_.size(); i++) {
    vmaDestroyBuffer(allocator_, uniformBuffers_[i], uniformBuffersAllocation_[i]);
  }

  device_.destroyImageView(depthImageView_, nullptr);
  vmaDestroyImage(allocator_, depthImage_, depthImageAllocation_);

  device_.destroyImageView(colourImageView_, nullptr);
  vmaDestroyImage(allocator_, colourImage_, colourImageAllocation_);

  device_.destroyDescriptorPool(descriptorPool_, nullptr);

  device_.freeCommandBuffers(graphicsCommandPool_, graphicsCommandBuffers_);

  device_.destroyPipeline(graphicsPipeline_, nullptr);
  device_.destroyPipelineLayout(pipelineLayout_, nullptr);

  device_.destroyRenderPass(renderPass_, nullptr);
  for (auto &image_view : swapChainImageViews_) device_.destroyImageView(image_view, nullptr);
  device_.destroySwapchainKHR(swapChain_, nullptr);
}

void Engine::recreateSwapChain() {
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
  createColourResources();
  createDepthResources();
  createFramebuffers();
  createUniformBuffers();
  createDescriptorPool();
  createDescriptorSets();
  createCommandBuffers();
}

void Engine::createImageViews() {
  swapChainImageViews_.resize(swapChainImages_.size());
  std::size_t i = 0;
  for (auto &swap_chain_image : swapChainImages_) {
    swapChainImageViews_[i] = createImageView(swap_chain_image, swapChainImageFormat_, vk::ImageAspectFlagBits::eColor, 1);
    i++;
  }
}

vk::ShaderModule Engine::createShaderModule(const std::vector<std::byte> &code) {
  vk::ShaderModuleCreateInfo create_info = {};

  create_info.codeSize = code.size();
  create_info.pCode = reinterpret_cast<const uint32_t *>(code.data());

  vk::ShaderModule shader_module = {};
  if (device_.createShaderModule(&create_info, nullptr, &shader_module) != vk::Result::eSuccess) {
    throw std::runtime_error("failed to create shader module!");
  }

  return shader_module;
}

void Engine::createRenderPass() {
  vk::AttachmentDescription color_attachment = {};
  color_attachment.format = swapChainImageFormat_;
  color_attachment.samples = sampleCount_;
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
  depth_attachment.format = findDepthFormat().value();
  depth_attachment.samples = sampleCount_;
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
  color_attachment_resolve.format = swapChainImageFormat_;
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

  if (device_.createRenderPass(&render_pass_info, nullptr, &renderPass_) != vk::Result::eSuccess) {
    throw std::runtime_error("failed to create render pass!");
  }
}

void Engine::createDescriptorSetLayout() {
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

  if (device_.createDescriptorSetLayout(&layout_info, nullptr, &descriptorSetLayout_) != vk::Result::eSuccess) {
    throw std::runtime_error("failed to create descriptor set layout!");
  }
}

void Engine::createGraphicsPipeline() {
  auto vert_shader_code = util::File::readFile(std::filesystem::path(ROOT_DIRECTORY) / "shaders/compiled/rectangle.vert.spv");
  auto frag_shader_code = util::File::readFile(std::filesystem::path(ROOT_DIRECTORY) / "shaders/compiled/rectangle.frag.spv");

  vk::ShaderModule vert_shader_module = createShaderModule(vert_shader_code);
  vk::ShaderModule frag_shader_module = createShaderModule(frag_shader_code);

  vk::PipelineShaderStageCreateInfo vert_shader_stage_info = {};
  vert_shader_stage_info.stage = vk::ShaderStageFlagBits::eVertex;
  vert_shader_stage_info.module = vert_shader_module;
  vert_shader_stage_info.pName = "main";

  vk::PipelineShaderStageCreateInfo frag_shader_stage_info = {};
  frag_shader_stage_info.stage = vk::ShaderStageFlagBits::eFragment;
  frag_shader_stage_info.module = frag_shader_module;
  frag_shader_stage_info.pName = "main";

  std::array<vk::PipelineShaderStageCreateInfo, 2> shader_stages = {vert_shader_stage_info, frag_shader_stage_info};

  auto binding_description = type::Vertex::getBindingDescription();
  auto attribute_descriptions = type::Vertex::getAttributeDescriptions();

  vk::PipelineVertexInputStateCreateInfo vertex_input_info = {};
  vertex_input_info.setVertexBindingDescriptions(binding_description);
  vertex_input_info.setVertexAttributeDescriptions(attribute_descriptions);

  vk::PipelineInputAssemblyStateCreateInfo input_assembly = {};
  input_assembly.topology = vk::PrimitiveTopology::eTriangleList;
  input_assembly.primitiveRestartEnable = VK_FALSE;

  vk::Viewport viewport = {};
  viewport.x = 0.0f;
  viewport.y = 0.0f;
  viewport.width = static_cast<float>(swapChainExtent_.width);
  viewport.height = static_cast<float>(swapChainExtent_.height);
  viewport.minDepth = 0.0f;
  viewport.maxDepth = 1.0f;

  vk::Rect2D scissor = {};
  scissor.offset = vk::Offset2D(0, 0);
  scissor.extent = swapChainExtent_;

  vk::PipelineViewportStateCreateInfo viewport_state = {};
  viewport_state.setViewports(viewport);
  viewport_state.setScissors(scissor);

  vk::PipelineRasterizationStateCreateInfo rasterizer = {};
  rasterizer.depthClampEnable = VK_FALSE;
  rasterizer.rasterizerDiscardEnable = VK_FALSE;
  rasterizer.polygonMode = vk::PolygonMode::eFill;  // could be wireframe too
  rasterizer.lineWidth = 1.0f;
  rasterizer.cullMode = vk::CullModeFlagBits::eBack;
  rasterizer.frontFace = vk::FrontFace::eCounterClockwise;
  rasterizer.depthBiasEnable = VK_FALSE;
  rasterizer.depthBiasConstantFactor = 0.0f;
  rasterizer.depthBiasClamp = 0.0f;
  rasterizer.depthBiasSlopeFactor = 0.0f;

  vk::PipelineMultisampleStateCreateInfo multisampling = {};
  multisampling.sampleShadingEnable = VK_FALSE;
  multisampling.rasterizationSamples = sampleCount_;
  multisampling.minSampleShading = .2f;
  multisampling.pSampleMask = nullptr;
  multisampling.alphaToCoverageEnable = VK_FALSE;
  multisampling.alphaToOneEnable = VK_FALSE;

  vk::PipelineColorBlendAttachmentState color_blend_attachment = {};
  color_blend_attachment.colorWriteMask =
      vk::ColorComponentFlagBits::eA | vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB;
  color_blend_attachment.blendEnable = VK_FALSE;
  color_blend_attachment.srcColorBlendFactor = vk::BlendFactor::eOne;
  color_blend_attachment.dstColorBlendFactor = vk::BlendFactor::eZero;
  color_blend_attachment.colorBlendOp = vk::BlendOp::eAdd;
  color_blend_attachment.srcAlphaBlendFactor = vk::BlendFactor::eOne;
  color_blend_attachment.dstAlphaBlendFactor = vk::BlendFactor::eZero;
  color_blend_attachment.alphaBlendOp = vk::BlendOp::eAdd;

  vk::PipelineDepthStencilStateCreateInfo depth_stencil = {};
  depth_stencil.depthTestEnable = VK_TRUE;
  depth_stencil.depthWriteEnable = VK_TRUE;
  depth_stencil.depthCompareOp = vk::CompareOp::eLess;
  depth_stencil.depthBoundsTestEnable = VK_FALSE;
  depth_stencil.minDepthBounds = 0.0f;
  depth_stencil.maxDepthBounds = 1.0f;
  depth_stencil.stencilTestEnable = VK_FALSE;
  depth_stencil.front = vk::StencilOpState();
  depth_stencil.back = vk::StencilOpState();

  vk::PipelineColorBlendStateCreateInfo color_blending = {};
  color_blending.logicOpEnable = VK_FALSE;
  color_blending.logicOp = vk::LogicOp::eCopy;
  color_blending.setAttachments(color_blend_attachment);
  color_blending.blendConstants[0] = 0.0f;
  color_blending.blendConstants[1] = 0.0f;
  color_blending.blendConstants[2] = 0.0f;
  color_blending.blendConstants[3] = 0.0f;

  vk::DynamicState dynamic_states[] = {vk::DynamicState::eViewport, vk::DynamicState::eLineWidth};

  vk::PipelineDynamicStateCreateInfo dynamic_state = {};
  dynamic_state.dynamicStateCount = 2;
  dynamic_state.pDynamicStates = dynamic_states;

  vk::PipelineLayoutCreateInfo pipeline_layout_info = {};
  pipeline_layout_info.setSetLayouts(descriptorSetLayout_);
  pipeline_layout_info.pushConstantRangeCount = 0;
  pipeline_layout_info.pPushConstantRanges = nullptr;

  if (device_.createPipelineLayout(&pipeline_layout_info, nullptr, &pipelineLayout_) != vk::Result::eSuccess) {
    throw std::runtime_error("failed to create pipeline layout!");
  }

  vk::GraphicsPipelineCreateInfo pipeline_info = {};
  pipeline_info.setStages(shader_stages);

  pipeline_info.pVertexInputState = &vertex_input_info;
  pipeline_info.pInputAssemblyState = &input_assembly;
  pipeline_info.pViewportState = &viewport_state;
  pipeline_info.pRasterizationState = &rasterizer;
  pipeline_info.pMultisampleState = &multisampling;
  pipeline_info.pDepthStencilState = &depth_stencil;
  pipeline_info.pColorBlendState = &color_blending;
  pipeline_info.pDynamicState = nullptr;  // &dynamic_state;
  pipeline_info.layout = pipelineLayout_;
  pipeline_info.renderPass = renderPass_;
  pipeline_info.subpass = 0;
  pipeline_info.basePipelineHandle = nullptr;  // requires VK_PIPELINE_CREATE_DERIVATIVE_BIT

  if (device_.createGraphicsPipelines(nullptr, 1, &pipeline_info, nullptr, &graphicsPipeline_) != vk::Result::eSuccess) {
    throw std::runtime_error("failed to create graphics pipeline!");
  }

  device_.destroyShaderModule(vert_shader_module, nullptr);
  device_.destroyShaderModule(frag_shader_module, nullptr);
}

void Engine::createFramebuffers() {
  swapChainFramebuffers_.resize(swapChainImageViews_.size());

  std::size_t i = 0;
  for (auto &swap_chain_image_view : swapChainImageViews_) {
    std::array<vk::ImageView, 3> attachments = {
        colourImageView_,
        depthImageView_,
        swap_chain_image_view,
    };

    vk::FramebufferCreateInfo framebuffer_info = {};
    framebuffer_info.renderPass = renderPass_;
    framebuffer_info.setAttachments(attachments);
    framebuffer_info.width = swapChainExtent_.width;
    framebuffer_info.height = swapChainExtent_.height;
    framebuffer_info.layers = 1;

    if (device_.createFramebuffer(&framebuffer_info, nullptr, &swapChainFramebuffers_[i]) != vk::Result::eSuccess) {
      throw std::runtime_error("failed to create framebuffer!");
    }

    i++;
  }
}

void Engine::createCommandPools() {
  QueueFamilyIndices queue_family_indices = findQueueFamilies(physicalDevice_);

  vk::CommandPoolCreateInfo graphics_pool_info = {};
  graphics_pool_info.queueFamilyIndex = queue_family_indices.graphicsFamily.value();
  graphics_pool_info.flags = {};

  if (device_.createCommandPool(&graphics_pool_info, nullptr, &graphicsCommandPool_) != vk::Result::eSuccess) {
    throw std::runtime_error("failed to create graphics command pool!");
  }

  vk::CommandPoolCreateInfo transfer_pool_info = {};
  transfer_pool_info.queueFamilyIndex = queue_family_indices.transferFamily.value();
  transfer_pool_info.flags = {};

  if (device_.createCommandPool(&transfer_pool_info, nullptr, &transferCommandPool_) != vk::Result::eSuccess) {
    throw std::runtime_error("failed to create transfer command pool!");
  }
}

void Engine::createColourResources() {
  auto colour_format = swapChainImageFormat_;

  colourImage_ = createImage(swapChainExtent_.width, swapChainExtent_.height, 1, sampleCount_, colour_format, vk::ImageTiling::eOptimal,
                             vk::ImageUsageFlagBits::eTransientAttachment | vk::ImageUsageFlagBits::eColorAttachment,
                             vk::MemoryPropertyFlagBits::eDeviceLocal, VMA_MEMORY_USAGE_GPU_ONLY, colourImageAllocation_);
  colourImageView_ = createImageView(colourImage_, colour_format, vk::ImageAspectFlagBits::eColor, 1);
}

std::optional<vk::Format> Engine::findSupportedFormat(const std::vector<vk::Format> &candidates, vk::ImageTiling tiling,
                                                      vk::FormatFeatureFlags features) {
  for (const auto &format : candidates) {
    vk::FormatProperties props = physicalDevice_.getFormatProperties(format);

    if (tiling == vk::ImageTiling::eLinear && (props.linearTilingFeatures & features) == features) {
      return format;
    }
    if (tiling == vk::ImageTiling::eOptimal && (props.optimalTilingFeatures & features) == features) {
      return format;
    }
  }

  return {};
}

std::optional<vk::Format> Engine::findDepthFormat() {
  return findSupportedFormat({vk::Format::eD32Sfloat, vk::Format::eD32SfloatS8Uint, vk::Format::eD24UnormS8Uint}, vk::ImageTiling::eOptimal,
                             vk::FormatFeatureFlagBits::eDepthStencilAttachment);
}

bool Engine::hasStencilComponent(const vk::Format &format) {
  return format == vk::Format::eD32SfloatS8Uint || format == vk::Format::eD24UnormS8Uint;
}

void Engine::createDepthResources() {
  auto depth_format = findDepthFormat();

  if (!depth_format) {
    throw std::runtime_error("failed to find supported format!");
  }

  depthImage_ = createImage(swapChainExtent_.width, swapChainExtent_.height, 1, sampleCount_, depth_format.value(),
                            vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eDepthStencilAttachment,
                            vk::MemoryPropertyFlagBits::eDeviceLocal, VMA_MEMORY_USAGE_GPU_ONLY, depthImageAllocation_);
  depthImageView_ = createImageView(depthImage_, depth_format.value(), vk::ImageAspectFlagBits::eDepth, 1);

  transitionImageLayout(depthImage_, depth_format.value(), vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthStencilAttachmentOptimal, 1);
}

vk::CommandBuffer Engine::beginSingleTimeCommands(const vk::CommandPool &commandPool) const {
  vk::CommandBufferAllocateInfo alloc_info = {};
  alloc_info.level = vk::CommandBufferLevel::ePrimary;
  alloc_info.commandPool = commandPool;
  alloc_info.commandBufferCount = 1;

  vk::CommandBuffer command_buffer = {};
  if (device_.allocateCommandBuffers(&alloc_info, &command_buffer) != vk::Result::eSuccess) {
    throw std::runtime_error("failed to allocate command buffer!");
  }

  vk::CommandBufferBeginInfo begin_info = {};
  begin_info.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;

  if (command_buffer.begin(&begin_info) != vk::Result::eSuccess) {
    throw std::runtime_error("failed to begin command buffer!");
  }

  return command_buffer;
}

void Engine::endSingleTimeCommands(vk::CommandBuffer &commandBuffer, const vk::Queue &queue, const vk::CommandPool &commandPool) const {
  commandBuffer.end();

  vk::SubmitInfo submit_info = {};
  submit_info.setCommandBuffers(commandBuffer);

  queue.submit(submit_info, nullptr);
  queue.waitIdle();

  device_.freeCommandBuffers(commandPool, commandBuffer);
}

vk::Buffer Engine::createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties,
                                VmaAllocation &allocation) const {
  VkBufferCreateInfo buffer_info = {};
  buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  buffer_info.size = size;
  buffer_info.usage = usage.m_mask;
  buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VmaAllocationCreateInfo alloc_info = {};
  alloc_info.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;  // @FIXME
  alloc_info.requiredFlags = properties.m_mask;

  VkBuffer buffer;
  vmaCreateBuffer(allocator_, &buffer_info, &alloc_info, &buffer, &allocation, nullptr);

  return buffer;
}

void Engine::copyBuffer(vk::Buffer srcBuffer, vk::Buffer dstBuffer, vk::DeviceSize size) const {
  auto command_buffer = beginSingleTimeCommands(transferCommandPool_);
  {
    vk::BufferCopy copy_region = {};
    copy_region.srcOffset = 0;
    copy_region.dstOffset = 0;
    copy_region.size = size;

    command_buffer.copyBuffer(srcBuffer, dstBuffer, 1, &copy_region);
  }
  endSingleTimeCommands(command_buffer, transferQueue_, transferCommandPool_);
}

vk::Buffer Engine::createBufferWithStaging(vk::DeviceSize size, const void *data, VmaAllocation &bufferAllocation,
                                           vk::BufferUsageFlagBits usage) {
  VmaAllocation staging_buffer_allocation;
  auto staging_buffer =
      createBuffer(size, vk::BufferUsageFlagBits::eTransferSrc,
                   vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, staging_buffer_allocation);

  void *mapped_memory;
  if (vmaMapMemory(allocator_, staging_buffer_allocation, &mapped_memory) != VK_SUCCESS) {
    throw std::runtime_error("failed to map vertex buffer memory!");
  }
  std::memcpy(mapped_memory, data, size);
  vmaUnmapMemory(allocator_, staging_buffer_allocation);

  vk::Buffer buffer = {};
  buffer = createBuffer(size, vk::BufferUsageFlagBits::eTransferDst | usage, vk::MemoryPropertyFlagBits::eDeviceLocal, bufferAllocation);

  copyBuffer(staging_buffer, buffer, size);

  vmaDestroyBuffer(allocator_, staging_buffer, staging_buffer_allocation);

  return buffer;
}

vk::Image Engine::createImage(uint32_t width, uint32_t height, uint32_t mipLevels, vk::SampleCountFlagBits sampleCount, vk::Format format,
                              vk::ImageTiling tiling, vk::ImageUsageFlags usage, vk::MemoryPropertyFlags properties,
                              VmaMemoryUsage allocationUsage, VmaAllocation &imageAllocation) {
  vk::ImageCreateInfo image_info = {};
  image_info.imageType = vk::ImageType::e2D;
  image_info.extent.width = width;
  image_info.extent.height = height;
  image_info.extent.depth = 1;
  image_info.mipLevels = mipLevels;
  image_info.arrayLayers = 1;

  image_info.format = format;
  image_info.tiling = tiling;
  image_info.initialLayout = vk::ImageLayout::eUndefined;

  image_info.usage = usage;
  image_info.sharingMode = vk::SharingMode::eExclusive;

  image_info.samples = sampleCount;

  VmaAllocationCreateInfo alloc_info = {};
  alloc_info.usage = allocationUsage;
  alloc_info.requiredFlags = properties.m_mask;

  VkImage image;
  if (vmaCreateImage(allocator_, reinterpret_cast<VkImageCreateInfo *>(&image_info), &alloc_info, &image, &imageAllocation, nullptr) !=
      VK_SUCCESS) {
    throw std::runtime_error("failed to create image!");
  }

  return image;
}

void Engine::transitionImageLayout(vk::Image &image, vk::Format format, vk::ImageLayout oldLayout, vk::ImageLayout newLayout,
                                   uint32_t mipLevels) {
  auto command_buffer = beginSingleTimeCommands(graphicsCommandPool_);
  {
    vk::ImageMemoryBarrier barrier = {};
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;

    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

    barrier.image = image;
    barrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = mipLevels;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;

    if (newLayout == vk::ImageLayout::eDepthStencilAttachmentOptimal) {
      barrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eDepth;

      if (hasStencilComponent(format)) {
        barrier.subresourceRange.aspectMask |= vk::ImageAspectFlagBits::eStencil;
      }
    }

    vk::PipelineStageFlags source_stage = {};
    vk::PipelineStageFlags destination_stage = {};

    if (oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eTransferDstOptimal) {
      barrier.srcAccessMask = vk::AccessFlagBits::eNoneKHR;
      barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;

      source_stage = vk::PipelineStageFlagBits::eTopOfPipe;
      destination_stage = vk::PipelineStageFlagBits::eTransfer;
    } else if (oldLayout == vk::ImageLayout::eTransferDstOptimal && newLayout == vk::ImageLayout::eShaderReadOnlyOptimal) {
      barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
      barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

      source_stage = vk::PipelineStageFlagBits::eTransfer;
      destination_stage = vk::PipelineStageFlagBits::eFragmentShader;
    } else if (oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eDepthStencilAttachmentOptimal) {
      barrier.srcAccessMask = vk::AccessFlagBits::eNoneKHR;
      barrier.dstAccessMask = vk::AccessFlagBits::eDepthStencilAttachmentRead | vk::AccessFlagBits::eDepthStencilAttachmentWrite;

      source_stage = vk::PipelineStageFlagBits::eTopOfPipe;
      destination_stage = vk::PipelineStageFlagBits::eEarlyFragmentTests;
    } else {
      throw std::invalid_argument("unsupported layout transition!");
    }

    command_buffer.pipelineBarrier(source_stage, destination_stage, vk::DependencyFlags(), {}, {}, barrier);
  }
  endSingleTimeCommands(command_buffer, graphicsQueue_, graphicsCommandPool_);
}

void Engine::copyBufferToImage(vk::Buffer &buffer, vk::Image &image, uint32_t width, uint32_t height) {
  auto command_buffer = beginSingleTimeCommands(graphicsCommandPool_);
  {
    vk::BufferImageCopy region = {};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = vk::Offset3D(0, 0, 0);
    region.imageExtent = vk::Extent3D(width, height, 1);

    command_buffer.copyBufferToImage(buffer, image, vk::ImageLayout::eTransferDstOptimal, region);
  }
  endSingleTimeCommands(command_buffer, graphicsQueue_, graphicsCommandPool_);
}

void Engine::generateMipmaps(vk::Image image, vk::Format imageFormat, int32_t texWidth, int32_t texHeight, uint32_t mipLevels) {
  auto format_properties = physicalDevice_.getFormatProperties(imageFormat);
  if (!(format_properties.optimalTilingFeatures & vk::FormatFeatureFlagBits::eSampledImageFilterLinear)) {
    throw std::runtime_error("texture image format does not support linear blitting!");
  }

  auto command_buffer = beginSingleTimeCommands(graphicsCommandPool_);
  {
    vk::ImageMemoryBarrier barrier = {};
    barrier.image = image;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    barrier.subresourceRange.levelCount = 1;

    int32_t mip_width = texWidth;
    int32_t mip_height = texHeight;
    for (uint32_t i = 1; i < mipLevels; i++) {
      barrier.subresourceRange.baseMipLevel = i - 1;
      barrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
      barrier.newLayout = vk::ImageLayout::eTransferSrcOptimal;
      barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
      barrier.dstAccessMask = vk::AccessFlagBits::eTransferRead;

      command_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eTransfer, vk::DependencyFlags(), {},
                                     {}, barrier);

      vk::ImageBlit blit = {};
      blit.srcOffsets[0] = vk::Offset3D(0, 0, 0);
      blit.srcOffsets[1] = vk::Offset3D(mip_width, mip_height, 1);
      blit.srcSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
      blit.srcSubresource.mipLevel = i - 1;
      blit.srcSubresource.baseArrayLayer = 0;
      blit.srcSubresource.layerCount = 1;
      blit.dstOffsets[0] = vk::Offset3D(0, 0, 0);
      int next_mip_width = mip_width > 1 ? mip_width / 2 : 1;
      int next_mip_height = mip_height > 1 ? mip_height / 2 : 1;
      blit.dstOffsets[1] = vk::Offset3D(next_mip_width, next_mip_height, 1);
      blit.dstSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
      blit.dstSubresource.mipLevel = i;
      blit.dstSubresource.baseArrayLayer = 0;
      blit.dstSubresource.layerCount = 1;

      command_buffer.blitImage(image, vk::ImageLayout::eTransferSrcOptimal, image, vk::ImageLayout::eTransferDstOptimal, blit,
                               vk::Filter::eLinear);

      barrier.oldLayout = vk::ImageLayout::eTransferSrcOptimal;
      barrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
      barrier.srcAccessMask = vk::AccessFlagBits::eTransferRead;
      barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

      command_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eFragmentShader,
                                     vk::DependencyFlags(), {}, {}, barrier);

      if (mip_width > 1) mip_width /= 2;
      if (mip_height > 1) mip_height /= 2;
    }

    barrier.subresourceRange.baseMipLevel = mipLevels - 1;
    barrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
    barrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
    barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
    barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

    command_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eFragmentShader, vk::DependencyFlags(),
                                   {}, {}, barrier);
  }
  endSingleTimeCommands(command_buffer, graphicsQueue_, graphicsCommandPool_);
}

void Engine::createTextureImage() {
  int tex_width;
  int tex_channels;
  int tex_height;

  auto path = std::filesystem::path(ROOT_DIRECTORY) / TEXTURE_PATH;
  stbi_uc *pixels = stbi_load(path.generic_string().c_str(), &tex_width, &tex_height, &tex_channels, STBI_rgb_alpha);
  vk::DeviceSize image_size = tex_width * tex_height * 4;
  {
    const int &largest_dimension = std::max(tex_width, tex_height);
    double n_factor_of_two = std::log2(largest_dimension);
    mipLevels_ = static_cast<uint32_t>(std::floor(n_factor_of_two)) + 1;
  }

  if (!pixels) {
    throw std::runtime_error("failed to load texture image!");
  }

  VmaAllocation staging_buffer_allocation;
  vk::Buffer staging_buffer =
      createBuffer(image_size, vk::BufferUsageFlagBits::eTransferSrc,
                   vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, staging_buffer_allocation);

  void *data;
  vmaMapMemory(allocator_, staging_buffer_allocation, &data);
  std::memcpy(data, pixels, image_size);
  vmaUnmapMemory(allocator_, staging_buffer_allocation);

  stbi_image_free(pixels);

  textureImage_ =
      createImage(tex_width, tex_height, mipLevels_, vk::SampleCountFlagBits::e1, vk::Format::eR8G8B8A8Srgb, vk::ImageTiling::eOptimal,
                  vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
                  vk::MemoryPropertyFlagBits::eDeviceLocal, VMA_MEMORY_USAGE_GPU_ONLY, textureImageAllocation_);

  transitionImageLayout(textureImage_, vk::Format::eR8G8B8A8Srgb, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal,
                        mipLevels_);
  copyBufferToImage(staging_buffer, textureImage_, tex_width, tex_height);

  vmaDestroyBuffer(allocator_, staging_buffer, staging_buffer_allocation);

  generateMipmaps(textureImage_, vk::Format::eR8G8B8A8Srgb, tex_width, tex_height, mipLevels_);
}

vk::ImageView Engine::createImageView(vk::Image &image, vk::Format format, vk::ImageAspectFlags aspectFlags, uint32_t mipLevels) {
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
  if (device_.createImageView(&view_info, nullptr, &image_view) != vk::Result::eSuccess) {
    throw std::runtime_error("failed to create texture image view!");
  }

  return image_view;
}

void Engine::createTextureImageView() {
  textureImageView_ = createImageView(textureImage_, vk::Format::eR8G8B8A8Srgb, vk::ImageAspectFlagBits::eColor, mipLevels_);
}

void Engine::createTextureSampler() {
  vk::SamplerCreateInfo sampler_info = {};
  sampler_info.magFilter = vk::Filter::eLinear;
  sampler_info.minFilter = vk::Filter::eLinear;
  sampler_info.addressModeU = vk::SamplerAddressMode::eRepeat;
  sampler_info.addressModeV = vk::SamplerAddressMode::eRepeat;
  sampler_info.addressModeW = vk::SamplerAddressMode::eRepeat;
  sampler_info.anisotropyEnable = VK_TRUE;

  auto properties = physicalDevice_.getProperties();
  sampler_info.maxAnisotropy = properties.limits.maxSamplerAnisotropy;

  sampler_info.borderColor = vk::BorderColor::eIntOpaqueBlack;
  sampler_info.unnormalizedCoordinates = VK_FALSE;
  sampler_info.compareEnable = VK_FALSE;
  sampler_info.compareOp = vk::CompareOp::eAlways;
  sampler_info.mipmapMode = vk::SamplerMipmapMode::eLinear;
  sampler_info.mipLodBias = 0.0f;
  sampler_info.minLod = 0.0f;
  sampler_info.maxLod = static_cast<float>(mipLevels_);

  if (device_.createSampler(&sampler_info, nullptr, &textureSampler_) != vk::Result::eSuccess) {
    throw std::runtime_error("failed to create texture sampler!");
  }
}

void Engine::loadModel() {
  tinyobj::attrib_t attrib;
  std::vector<tinyobj::shape_t> shapes;
  std::vector<tinyobj::material_t> materials;
  std::string warn;
  std::string err;

  if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, MODEL_PATH.data())) {
    throw std::runtime_error(warn + err);
  }

  std::unordered_map<type::Vertex, uint32_t> unique_vertices = {};
  indices_.reserve(0);
  for (const auto &shape : shapes) {
    indices_.reserve(indices_.capacity() + shape.mesh.indices.size());
    for (const auto &index : shape.mesh.indices) {
      type::Vertex vertex = {};

      vertex.pos = {
          attrib.vertices[3 * index.vertex_index + 0],
          attrib.vertices[3 * index.vertex_index + 1],
          attrib.vertices[3 * index.vertex_index + 2],
      };

      vertex.texCoord = {
          attrib.texcoords[2 * index.texcoord_index + 0],
          1.0f - attrib.texcoords[2 * index.texcoord_index + 1],
      };
      vertex.colour = {1.0f, 1.0f, 1.0f};

      if (unique_vertices.count(vertex) == 0) {
        unique_vertices[vertex] = static_cast<uint32_t>(vertices_.size());
        vertices_.push_back(vertex);
      }

      indices_.emplace_back(unique_vertices[vertex]);
    }
  }
}

void Engine::createVertexBuffer() {
  vertexBuffer_ = createBufferWithStaging(vertices_.size() * sizeof(vertices_[0]), vertices_.data(), vertexBufferAllocation_,
                                          vk::BufferUsageFlagBits::eVertexBuffer);
}

void Engine::createIndexBuffer() {
  indexBuffer_ = createBufferWithStaging(indices_.size() * sizeof(indices_[0]), indices_.data(), indexBufferAllocation_,
                                         vk::BufferUsageFlagBits::eIndexBuffer);
}

void Engine::createUniformBuffers() {
  vk::DeviceSize buffer_size = sizeof(UniformBufferObject);

  uniformBuffers_.resize(swapChainImages_.size());
  uniformBuffersAllocation_.resize(swapChainImages_.size());

  for (std::size_t i = 0; i < swapChainImages_.size(); i++) {
    uniformBuffers_[i] =
        createBuffer(buffer_size, vk::BufferUsageFlagBits::eUniformBuffer,
                     vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, uniformBuffersAllocation_[i]);
  }
}

void Engine::createDescriptorPool() {
  std::array<vk::DescriptorPoolSize, 2> pool_sizes = {};
  pool_sizes[0].type = vk::DescriptorType::eUniformBuffer;
  pool_sizes[0].descriptorCount = static_cast<uint32_t>(swapChainImages_.size());
  pool_sizes[1].type = vk::DescriptorType::eCombinedImageSampler;
  pool_sizes[1].descriptorCount = static_cast<uint32_t>(swapChainImages_.size());

  vk::DescriptorPoolCreateInfo pool_info = {};
  pool_info.setPoolSizes(pool_sizes);
  pool_info.maxSets = static_cast<uint32_t>(swapChainImages_.size());

  if (device_.createDescriptorPool(&pool_info, nullptr, &descriptorPool_) != vk::Result::eSuccess) {
    throw std::runtime_error("failed to create descriptor pool!");
  }
}

void Engine::createDescriptorSets() {
  std::vector<vk::DescriptorSetLayout> layouts(swapChainImages_.size(), descriptorSetLayout_);
  vk::DescriptorSetAllocateInfo alloc_info = {};
  alloc_info.descriptorPool = descriptorPool_;
  alloc_info.descriptorSetCount = swapChainImages_.size();
  alloc_info.setSetLayouts(layouts);

  descriptorSets_.resize(swapChainImages_.size());
  if (device_.allocateDescriptorSets(&alloc_info, descriptorSets_.data()) != vk::Result::eSuccess) {
    throw std::runtime_error("failed to allocate descriptor sets!");
  }

  std::size_t i = 0;
  for (auto &descriptor_set : descriptorSets_) {
    std::array<vk::WriteDescriptorSet, 2> descriptor_writes = {};

    vk::DescriptorBufferInfo buffer_info = {};
    buffer_info.buffer = uniformBuffers_[i];
    buffer_info.offset = 0;
    buffer_info.range = sizeof(UniformBufferObject);

    descriptor_writes[0].dstSet = descriptor_set;
    descriptor_writes[0].dstBinding = 0;
    descriptor_writes[0].dstArrayElement = 0;
    descriptor_writes[0].descriptorType = vk::DescriptorType::eUniformBuffer;
    descriptor_writes[0].descriptorCount = 1;
    descriptor_writes[0].setBufferInfo(buffer_info);

    vk::DescriptorImageInfo image_info = {};
    image_info.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
    image_info.imageView = textureImageView_;
    image_info.sampler = textureSampler_;

    descriptor_writes[1].dstSet = descriptor_set;
    descriptor_writes[1].dstBinding = 1;
    descriptor_writes[1].dstArrayElement = 0;
    descriptor_writes[1].descriptorType = vk::DescriptorType::eCombinedImageSampler;
    descriptor_writes[1].descriptorCount = 1;
    descriptor_writes[1].setImageInfo(image_info);

    device_.updateDescriptorSets(descriptor_writes, {});

    i++;
  }
}

void Engine::createCommandBuffers() {
  graphicsCommandBuffers_.resize(swapChainFramebuffers_.size());

  vk::CommandBufferAllocateInfo alloc_info = {};
  alloc_info.commandPool = graphicsCommandPool_;
  alloc_info.level = vk::CommandBufferLevel::ePrimary;
  alloc_info.commandBufferCount = static_cast<uint32_t>(graphicsCommandBuffers_.size());

  if (device_.allocateCommandBuffers(&alloc_info, graphicsCommandBuffers_.data()) != vk::Result::eSuccess) {
    throw std::runtime_error("failed to allocate command buffers!");
  }

  std::size_t i = 0;
  for (auto &command_buffer : graphicsCommandBuffers_) {
    vk::CommandBufferBeginInfo begin_info = {};
    begin_info.flags = {};
    begin_info.pInheritanceInfo = nullptr;

    if (command_buffer.begin(&begin_info) != vk::Result::eSuccess) {
      throw std::runtime_error("failed to begin recording command buffer!");
    }

    vk::RenderPassBeginInfo render_pass_info = {};
    render_pass_info.renderPass = renderPass_;
    render_pass_info.framebuffer = swapChainFramebuffers_.at(i);
    render_pass_info.renderArea.offset = vk::Offset2D(0, 0);
    render_pass_info.renderArea.extent = swapChainExtent_;

    vk::ClearColorValue clear_color_value(std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f});
    vk::ClearDepthStencilValue clear_depth_stencil_value(1.0f, 0);

    std::array<vk::ClearValue, 2> clear_values{};
    clear_values[0].color = clear_color_value;
    clear_values[1].depthStencil = clear_depth_stencil_value;
    render_pass_info.setClearValues(clear_values);

    command_buffer.beginRenderPass(&render_pass_info, vk::SubpassContents::eInline);
    {
      command_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, graphicsPipeline_);

      vk::DeviceSize offset = 0;
      graphicsCommandBuffers_[i].bindVertexBuffers(0, vertexBuffer_, offset);
      graphicsCommandBuffers_[i].bindIndexBuffer(indexBuffer_, 0, vk::IndexType::eUint32);

      command_buffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout_, 0, descriptorSets_[i], {});

      command_buffer.drawIndexed(static_cast<uint32_t>(indices_.size()), 1, 0, 0, 0);
    }
    command_buffer.endRenderPass();

    command_buffer.end();

    i++;
  }
}

void Engine::createSyncObjects() {
  imagesInFlight_.resize(swapChainImages_.size(), {});

  vk::SemaphoreCreateInfo semaphore_info = {};
  vk::FenceCreateInfo fence_info = {};
  fence_info.flags = vk::FenceCreateFlagBits::eSignaled;

  for (std::size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
    if (device_.createSemaphore(&semaphore_info, nullptr, &imageAvailableSemaphores_[i]) != vk::Result::eSuccess ||
        device_.createSemaphore(&semaphore_info, nullptr, &renderFinishedSemaphores_[i]) != vk::Result::eSuccess ||
        device_.createFence(&fence_info, nullptr, &inFlightFences_[i]) != vk::Result::eSuccess) {
      throw std::runtime_error("failed to create semaphores for a frame!");
    }
  }
}

void Engine::updateUniformBuffer(uint32_t currentImage) {
  auto current_time = std::chrono::high_resolution_clock::now();
  float delta_time = std::chrono::duration<float, std::chrono::seconds::period>(current_time - startTime_).count();

  UniformBufferObject uniform_buffer_object = {};
  uniform_buffer_object.model = glm::rotate(glm::mat4(1.0f), glm::radians(0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
  uniform_buffer_object.view = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
  uniform_buffer_object.proj =
      glm::perspective(glm::radians(45.0f), swapChainExtent_.width / static_cast<float>(swapChainExtent_.height), 0.1f, 10.0f);

  uniform_buffer_object.proj[1][1] *= -1;  // y coordinate is inverted, flip the signed bit

  void *data;
  vmaMapMemory(allocator_, uniformBuffersAllocation_[currentImage], &data);
  std::memcpy(data, &uniform_buffer_object, sizeof(uniform_buffer_object));
  vmaUnmapMemory(allocator_, uniformBuffersAllocation_[currentImage]);
}

void Engine::drawFrame() {
  if (device_.waitForFences(inFlightFences_[currentFrame_], VK_TRUE, UINT64_MAX) != vk::Result::eSuccess) {
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

  updateUniformBuffer(image_index);

  // Check if a previous frame is using this image (i.e. there is its fence to wait on)
  if (imagesInFlight_[image_index]) {
    if (device_.waitForFences(imagesInFlight_[image_index], VK_TRUE, UINT64_MAX) != vk::Result::eSuccess) {
      throw std::runtime_error("wait for fences timed out");
    };
  }
  // Mark the image as now being in use by this frame
  imagesInFlight_[image_index] = inFlightFences_[currentFrame_];

  vk::SubmitInfo submit_info = {};

  std::array<vk::Semaphore, 1> wait_semaphores = {imageAvailableSemaphores_[currentFrame_]};
  std::array<vk::PipelineStageFlags, 1> wait_stages = {vk::PipelineStageFlagBits::eColorAttachmentOutput};
  submit_info.setWaitSemaphores(wait_semaphores);
  submit_info.setWaitDstStageMask(wait_stages);
  submit_info.setCommandBuffers(graphicsCommandBuffers_[image_index]);

  std::array<vk::Semaphore, 1> signal_semaphores = {renderFinishedSemaphores_[currentFrame_]};
  submit_info.setSignalSemaphores(signal_semaphores);

  if (device_.resetFences(1, &inFlightFences_[currentFrame_]) != vk::Result::eSuccess) {
    throw std::runtime_error("reset fences failed!");
  };

  if (graphicsQueue_.submit(1, &submit_info, inFlightFences_[currentFrame_]) != vk::Result::eSuccess) {
    throw std::runtime_error("failed to submit draw command buffer!");
  }

  vk::PresentInfoKHR present_info = {};

  present_info.setWaitSemaphores(signal_semaphores);

  std::array<vk::SwapchainKHR, 1> swap_chains = {swapChain_};
  present_info.setSwapchains(swap_chains);
  present_info.setImageIndices(image_index);

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

}  // namespace rendering