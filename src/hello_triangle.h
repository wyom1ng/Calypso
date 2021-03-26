﻿//
// Created by Wyoming on 22/03/2021.
//

#ifndef CALYPSO_HELLO_TRIANGLE_H
#define CALYPSO_HELLO_TRIANGLE_H

#include <iostream>
#include <set>
#include <stdexcept>
#include <vector>
#define GLFW_INCLUDE_VULKAN
#include "util/file.h"
#include <GLFW/glfw3.h>
#include <spdlog/spdlog.h>
#include <vulkan/vulkan.hpp>

class HelloTriangle {
 public:
  void run() {
    initWindow();
    initVulkan();
    mainLoop();
    cleanup();
  }

 private:
  const std::vector<const char *> validationLayers_ = {"VK_LAYER_KHRONOS_validation"};
  const std::vector<const char *> deviceExtensions_ = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};

  VkDebugUtilsMessengerEXT debugMessenger_;

#ifdef NDEBUG
  static constexpr ENABLE_VALIDATION_LAYERS = false;
#else
  static constexpr bool ENABLE_VALIDATION_LAYERS = true;
#endif

  static constexpr int MAX_FRAMES_IN_FLIGHT = 5;

  GLFWwindow *window_;
  VkInstance instance_;
  VkPhysicalDevice physicalDevice_ = VK_NULL_HANDLE;
  VkDevice device_;

  VkQueue graphicsQueue_;
  VkQueue presentQueue_;
  VkSurfaceKHR surface_;

  VkSwapchainKHR swapChain_;
  std::vector<VkImage> swapChainImages_;
  VkFormat swapChainImageFormat_;
  VkExtent2D swapChainExtent_;
  
  std::vector<VkImageView> swapChainImageViews_;
  std::vector<VkFramebuffer> swapChainFramebuffers_;
  
  VkRenderPass renderPass_;
  VkPipelineLayout pipelineLayout_;
  VkPipeline graphicsPipeline_;

  VkCommandPool commandPool_;
  std::vector<VkCommandBuffer> commandBuffers_;

  std::vector<VkSemaphore> imageAvailableSemaphores_ = std::vector<VkSemaphore>(MAX_FRAMES_IN_FLIGHT);
  std::vector<VkSemaphore> renderFinishedSemaphores_ = std::vector<VkSemaphore>(MAX_FRAMES_IN_FLIGHT);
  std::vector<VkFence> inFlightFences_ = std::vector<VkFence>(MAX_FRAMES_IN_FLIGHT);
  std::vector<VkFence> imagesInFlight_;
  std::size_t currentFrame_ = 0;
  bool framebufferResized_ = false;
  
  static constexpr uint32_t WIDTH = 1024;
  static constexpr uint32_t HEIGHT = 768;

  struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;

    [[nodiscard]] bool isComplete() const { return graphicsFamily.has_value() && presentFamily.has_value(); }
  };

  struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
  };

  static void framebufferResizeCallback(GLFWwindow* window, int width, int height);

  void initWindow();

  void initVulkan();

  bool checkValidationLayerSupport();

  static std::vector<const char *> getRequiredExtensions();

  void createInstance();

  VkDebugUtilsMessengerCreateInfoEXT getDebugMessengerCreateInfo();

  static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                                                      VkDebugUtilsMessageTypeFlagsEXT messageType,
                                                      const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData, void *pUserData);

  void setupDebugMessenger();

  void createSurface();

  QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device);

  SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device);

  void pickPhysicalDevice();

  bool checkDeviceExtensionSupport(VkPhysicalDevice device);

  void createLogicalDevice();

  static VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR> &availableFormats);

  static VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR> &availablePresentModes);

  VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR &capabilities);

  void createSwapChain();
  
  void cleanupSwapChain();
  
  void recreateSwapChain();

  void createImageViews();
  
  VkShaderModule createShaderModule(const std::vector<std::byte> &code);
  
  void createRenderPass();

  void createGraphicsPipeline();

  void createFramebuffers();
  
  void createCommandPool();

  void createCommandBuffers();
  
  void createSyncObjects();

  void drawFrame();

  void mainLoop();

  void cleanup();
};

#endif  // CALYPSO_HELLO_TRIANGLE_H
