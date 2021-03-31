//
// Created by Wyoming on 22/03/2021.
//

#ifndef CALYPSO_HELLO_TRIANGLE_H
#define CALYPSO_HELLO_TRIANGLE_H

#define GLFW_INCLUDE_VULKAN

#include <iostream>
#include <set>
#include <stdexcept>
#include <vector>
#include <chrono>

#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <stb_image.h>
#include <spdlog/spdlog.h>
#include <vulkan/vulkan.hpp>
#include <vk_mem_alloc.h>

#include "util/file.h"

class HelloTriangle {
 public:
  HelloTriangle();

  void mainloop();

  ~HelloTriangle();

 private:
  const std::vector<const char *> validationLayers_ = {"VK_LAYER_KHRONOS_validation"};
  const std::vector<const char *> deviceExtensions_ = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};

  vk::DebugUtilsMessengerEXT debugMessenger_;

#ifdef NDEBUG
  static constexpr ENABLE_VALIDATION_LAYERS = false;
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

  vk::SwapchainKHR swapChain_;
  std::vector<vk::Image> swapChainImages_;
  vk::Format swapChainImageFormat_;
  vk::Extent2D swapChainExtent_;

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

  static constexpr uint32_t WIDTH = 1024;
  static constexpr uint32_t HEIGHT = 768;

  struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;
    std::optional<uint32_t> transferFamily;

    [[nodiscard]] bool isComplete() const { return graphicsFamily.has_value() && presentFamily.has_value() && transferFamily.has_value(); }
  };

  struct SwapChainSupportDetails {
    vk::SurfaceCapabilitiesKHR capabilities;
    std::vector<vk::SurfaceFormatKHR> formats;
    std::vector<vk::PresentModeKHR> presentModes;
  };

  struct Vertex {
    glm::vec2 pos;
    glm::vec3 colour;

    static vk::VertexInputBindingDescription getBindingDescription() {
      vk::VertexInputBindingDescription binding_description;
      binding_description.binding = 0;
      binding_description.stride = sizeof(Vertex);
      binding_description.inputRate = vk::VertexInputRate::eVertex;

      return binding_description;
    }

    static std::array<vk::VertexInputAttributeDescription, 2> getAttributeDescriptions() {
      std::array<vk::VertexInputAttributeDescription, 2> attribute_descriptions;

      attribute_descriptions[0].binding = 0;
      attribute_descriptions[0].location = 0;
      attribute_descriptions[0].format = vk::Format::eR32G32Sfloat;
      attribute_descriptions[0].offset = offsetof(Vertex, pos);

      attribute_descriptions[1].binding = 0;
      attribute_descriptions[1].location = 1;
      attribute_descriptions[1].format = vk::Format::eR32G32B32Sfloat;
      attribute_descriptions[1].offset = offsetof(Vertex, colour);

      return attribute_descriptions;
    }
  };

  const std::vector<Vertex> vertices_ = {{{-0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}},
                                         {{0.5f, -0.5f}, {1.0f, 0.0f, 1.0f}},
                                         {{0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}},
                                         {{-0.5f, 0.5f}, {1.0f, 0.0f, 1.0f}}};

  const std::vector<uint16_t> indices_ = {
      0, 1, 2, 2, 3, 0,
  };

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
  vk::Image textureImage_;
  VmaAllocation textureImageAllocation_;

  void initDispatchLoader();

  static void framebufferResizeCallback(GLFWwindow *window, int width, int height);

  void initWindow();

  void initVulkan();

  bool checkValidationLayerSupport();

  static std::vector<const char *> getRequiredExtensions();

  void createInstance();

  static vk::DebugUtilsMessengerCreateInfoEXT getDebugMessengerCreateInfo();

  static VKAPI_ATTR vk::Bool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                                                        VkDebugUtilsMessageTypeFlagsEXT messageType,
                                                        const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData, void *pUserData);

  void setupDebugMessenger();

  void createSurface();

  QueueFamilyIndices findQueueFamilies(vk::PhysicalDevice device);

  SwapChainSupportDetails querySwapChainSupport(vk::PhysicalDevice device);

  void pickPhysicalDevice();

  bool checkDeviceExtensionSupport(vk::PhysicalDevice device);

  void createLogicalDevice();
  
  void createAllocator();

  static vk::SurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR> &availableFormats);

  static vk::PresentModeKHR chooseSwapPresentMode(const std::vector<vk::PresentModeKHR> &availablePresentModes);

  vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR &capabilities);

  void createSwapChain();

  void cleanupSwapChain();

  void recreateSwapChain();

  void createImageViews();

  vk::ShaderModule createShaderModule(const std::vector<std::byte> &code);

  void createRenderPass();

  void createDescriptorSetLayout();

  void createGraphicsPipeline();

  void createFramebuffers();

  void createCommandPools();
  
  [[nodiscard]] vk::CommandBuffer beginSingleTimeCommands(const vk::CommandPool &commandPool) const;
  
  void endSingleTimeCommands(vk::CommandBuffer &commandBuffer, const vk::Queue &queue, const vk::CommandPool &commandPool) const;

  vk::Buffer createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties, VmaAllocation &allocation) const;
  
  void copyBuffer(vk::Buffer srcBuffer, vk::Buffer dstBuffer, vk::DeviceSize size) const;

  vk::Buffer createBufferWithStaging(vk::DeviceSize size, const void *data, VmaAllocation &bufferAllocation, vk::BufferUsageFlagBits usage);

  vk::Image createImage(uint32_t width, uint32_t height, vk::Format format, vk::ImageTiling tiling, vk::ImageUsageFlags usage, vk::MemoryPropertyFlags properties, VmaAllocation &imageAllocation);
  
  void transitionImageLayout(vk::Image &image, vk::Format format, vk::ImageLayout oldLayout, vk::ImageLayout newLayout);

  void copyBufferToImage(vk::Buffer &buffer, vk::Image &image, uint32_t width, uint32_t height);
  
  void createTextureImage();
  
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

#endif  // CALYPSO_HELLO_TRIANGLE_H
