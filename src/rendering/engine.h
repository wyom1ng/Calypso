//
// Created by Wyoming on 11/04/2021.
//

#ifndef CALYPSO_ENGINE_H
#define CALYPSO_ENGINE_H

#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define NOMINMAX

#include <chrono>
#include <optional>
#include <vector>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vk_mem_alloc.h>
#include <vulkan/vulkan.hpp>
#include "type/misc.h"
#include "type/vertex.h"

namespace rendering {

class Engine {
 public:
  Engine();

  void mainloop();

  ~Engine();

 private:
  const std::vector<const char *> validationLayers_ = {"VK_LAYER_KHRONOS_validation"};
  const std::vector<const char *> deviceExtensions_ = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};

  vk::DebugUtilsMessengerEXT debugMessenger_;

#ifdef NDEBUG
  static constexpr bool ENABLE_VALIDATION_LAYERS = false;
#else
  static constexpr bool ENABLE_VALIDATION_LAYERS = true;
#endif

  static constexpr int MAX_FRAMES_IN_FLIGHT = 2;

  std::chrono::high_resolution_clock::time_point startTime_;

  VmaAllocator allocator_;
  vk::DynamicLoader dynamicLoader_;

  GLFWwindow *window_;
  vk::Instance instance_;
  vk::PhysicalDevice physicalDevice_;
  vk::Device device_;

  vk::Queue graphicsQueue_;
  vk::Queue presentQueue_;
  vk::Queue transferQueue_;
  vk::SurfaceKHR surface_;
  
  
  type::SwapchainData swapchainData_;

  std::vector<vk::ImageView> swapChainImageViews_;
  std::vector<vk::Framebuffer> swapChainFramebuffers_;

  vk::RenderPass renderPass_;

  vk::DescriptorSetLayout descriptorSetLayout_;
  vk::DescriptorPool descriptorPool_;
  std::vector<vk::DescriptorSet> descriptorSets_;

  vk::PipelineLayout pipelineLayout_;
  vk::Pipeline graphicsPipeline_;

  vk::CommandPool graphicsCommandPool_;
  std::vector<vk::CommandBuffer> graphicsCommandBuffers_;
  vk::CommandPool transferCommandPool_;

  std::vector<vk::Semaphore> imageAvailableSemaphores_ = std::vector<vk::Semaphore>(MAX_FRAMES_IN_FLIGHT);
  std::vector<vk::Semaphore> renderFinishedSemaphores_ = std::vector<vk::Semaphore>(MAX_FRAMES_IN_FLIGHT);
  std::vector<vk::Fence> inFlightFences_ = std::vector<vk::Fence>(MAX_FRAMES_IN_FLIGHT);
  std::vector<vk::Fence> imagesInFlight_;
  std::size_t currentFrame_ = 0;
  bool framebufferResized_ = false;

  static constexpr uint16_t WIDTH = 1024;
  static constexpr uint16_t HEIGHT = 768;

  static constexpr std::string_view MODEL_PATH = "assets/viking_room.obj";
  static constexpr std::string_view TEXTURE_PATH = "assets/viking_room.png";

  std::vector<type::Vertex> vertices_;

  std::vector<uint32_t> indices_;

  struct UniformBufferObject {
    alignas(16) glm::mat4 model;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 proj;
  };

  vk::Buffer vertexBuffer_;
  VmaAllocation vertexBufferAllocation_;
  vk::Buffer indexBuffer_;
  VmaAllocation indexBufferAllocation_;
  std::vector<vk::Buffer> uniformBuffers_;
  std::vector<VmaAllocation> uniformBuffersAllocation_;
  uint32_t mipLevels_;

  vk::Sampler textureSampler_;

  vk::Image textureImage_;
  VmaAllocation textureImageAllocation_;
  vk::ImageView textureImageView_;

  vk::Image depthImage_;
  VmaAllocation depthImageAllocation_;
  vk::ImageView depthImageView_;

  vk::Image colourImage_;
  VmaAllocation colourImageAllocation_;
  vk::ImageView colourImageView_;

  vk::SampleCountFlagBits sampleCount_ = vk::SampleCountFlagBits::e1;

  static void framebufferResizeCallback(GLFWwindow *window, int width, int height);

  void initVulkan();

  static VKAPI_ATTR vk::Bool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                                                        VkDebugUtilsMessageTypeFlagsEXT messageType,
                                                        const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData, void *pUserData);

  void cleanupSwapChain();

  void recreateSwapchain();

  void createImageViews();

  vk::ShaderModule createShaderModule(const std::vector<std::byte> &code);

  void createRenderPass();

  void createDescriptorSetLayout();

  void createGraphicsPipeline();

  void createFramebuffers();

  void createCommandPools();

  void createColourResources();

  std::optional<vk::Format> findSupportedFormat(const std::vector<vk::Format>& candidates, vk::ImageTiling tiling, vk::FormatFeatureFlags features);

  std::optional<vk::Format> findDepthFormat();

  static bool hasStencilComponent(const vk::Format &format);

  void createDepthResources();

  [[nodiscard]] vk::CommandBuffer beginSingleTimeCommands(const vk::CommandPool &commandPool) const;

  void endSingleTimeCommands(vk::CommandBuffer &commandBuffer, const vk::Queue &queue, const vk::CommandPool &commandPool) const;

  vk::Buffer createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties, VmaAllocation &allocation) const;

  void copyBuffer(vk::Buffer srcBuffer, vk::Buffer dstBuffer, vk::DeviceSize size) const;

  vk::Buffer createBufferWithStaging(vk::DeviceSize size, const void *data, VmaAllocation &bufferAllocation, vk::BufferUsageFlagBits usage);

  vk::Image createImage(uint32_t width, uint32_t height, uint32_t mipLevels, vk::SampleCountFlagBits sampleCount, vk::Format format,
                        vk::ImageTiling tiling, vk::ImageUsageFlags usage, vk::MemoryPropertyFlags properties,
                        VmaMemoryUsage allocationUsage, VmaAllocation &imageAllocation);

  void transitionImageLayout(vk::Image &image, vk::Format format, vk::ImageLayout oldLayout, vk::ImageLayout newLayout, uint32_t mipLevels);

  void copyBufferToImage(vk::Buffer &buffer, vk::Image &image, uint32_t width, uint32_t height);

  void generateMipmaps(vk::Image image, vk::Format imageFormat, int32_t texWidth, int32_t texHeight, uint32_t mipLevels);

  void createTextureImage();

  vk::ImageView createImageView(vk::Image &image, vk::Format format, vk::ImageAspectFlags aspectFlags, uint32_t mipLevels);

  void createTextureImageView();

  void createTextureSampler();

  void loadModel();

  void createVertexBuffer();

  void createIndexBuffer();

  void createUniformBuffers();

  void createDescriptorPool();

  void createDescriptorSets();

  void createCommandBuffers();

  void createSyncObjects();

  void updateUniformBuffer(uint32_t currentImage);

  void drawFrame();
};

}  // namespace rendering

#endif  // CALYPSO_ENGINE_H
