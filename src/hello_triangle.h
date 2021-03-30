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
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <spdlog/spdlog.h>
#include <vulkan/vulkan.hpp>
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

  static constexpr int MAX_FRAMES_IN_FLIGHT = 5;

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

  const std::vector<Vertex> vertices_ = {
      {{0.0f, -0.5f}, {1.0f, 0.0f, 1.0f}},
      {{0.5f, 0.5f}, {1.0f, 0.0f, 0.0f}},
      {{-0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}},
  };

  vk::Buffer vertexBuffer_;
  vk::DeviceMemory vertexBufferMemory_;

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

  void createGraphicsPipeline();

  void createFramebuffers();

  void createCommandPools();

  uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties);
  
  vk::Buffer createBuffer(vk::DeviceSize size,  vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties, vk::DeviceMemory &bufferMemory);

  void copyBuffer(vk::Buffer srcBuffer, vk::Buffer dstBuffer, vk::DeviceSize size);
  
  void createVertexBuffer();

  void createCommandBuffers();

  void createSyncObjects();

  void drawFrame();
};

#endif  // CALYPSO_HELLO_TRIANGLE_H
