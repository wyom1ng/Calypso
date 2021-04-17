//
// Created by Wyoming on 11/04/2021.
//

#ifndef CALYPSO_INITIALISERS_H
#define CALYPSO_INITIALISERS_H

#define GLFW_INCLUDE_VULKAN
#define NOMINMAX

#include <vk_mem_alloc.h>
#include <GLFW/glfw3.h>
#include <vulkan/vulkan.hpp>
#include "type/misc.h"

namespace rendering {

class Initialisers {
 public:
  
  static void initDispatchLoader(const vk::DynamicLoader &dynamicLoader);
  
  static GLFWwindow *createWindow(void *user, std::function<void(GLFWwindow *window, int width, int height)> framebufferResizeCallback, uint16_t initialWidth, uint16_t initialHeight);

  static bool checkValidationLayerSupport(const std::vector<const char *> &validationLayers);

  static std::vector<const char *> getRequiredExtensions(bool enableValidationLayers);

  static vk::DebugUtilsMessengerCreateInfoEXT getDebugMessengerCreateInfo(PFN_vkDebugUtilsMessengerCallbackEXT debugCallback);

  static vk::Instance createInstance(bool enableValidationLayers, const std::vector<const char *> &validationLayers,
                                     PFN_vkDebugUtilsMessengerCallbackEXT debugCallback);

  static vk::DebugUtilsMessengerEXT setupDebugMessenger(const vk::Instance &instance, bool enableValidationLayers,
                                                        PFN_vkDebugUtilsMessengerCallbackEXT debugCallback);

  static vk::SurfaceKHR createSurface(const vk::Instance &instance, GLFWwindow *window);

  static type::SwapchainSupportDetails querySwapchainSupport(const vk::PhysicalDevice &device, const vk::SurfaceKHR &surface);

  static type::QueueFamilyIndices findQueueFamilies(const vk::PhysicalDevice &device, const vk::SurfaceKHR &surface);

  static vk::PhysicalDevice createPhysicalDevice(const vk::Instance &instance, const vk::SurfaceKHR &surface,
                                                 const std::vector<const char *> &deviceExtensions);

  static vk::SampleCountFlagBits getMaxUsableSampleCount(const vk::PhysicalDevice &physicalDevice);

  static vk::Device createLogicalDevice(const vk::PhysicalDevice &physicalDevice, const vk::SurfaceKHR &surface,
                                        bool enableValidationLayers, const std::vector<const char *> &validationLayers,
                                        const std::vector<const char *> &deviceExtensions);

  static std::array<vk::Queue, 3> createQueues(const vk::PhysicalDevice &physicalDevice, const vk::SurfaceKHR &surface,
                                               const vk::Device &device);

  static VmaAllocator createAllocator(const vk::Instance &instance, const vk::PhysicalDevice &physicalDevice, const vk::Device &device);
  
  static type::SwapchainData createSwapchain(const vk::PhysicalDevice &physicalDevice, const vk::SurfaceKHR &surface, const vk::Device &device,  GLFWwindow *window);
  
 private:
  static bool checkDeviceExtensionSupport(const vk::PhysicalDevice &device, const std::vector<const char *> &deviceExtensions);

  static uint32_t rateDevice(const vk::PhysicalDevice &physicalDevice, const vk::SurfaceKHR &surface,
                             const std::vector<const char *> &deviceExtensions);

  static vk::SurfaceFormatKHR chooseSwapchainSurfaceFormat(const std::vector<vk::SurfaceFormatKHR> &availableFormats);

  static vk::PresentModeKHR chooseSwapchainPresentMode(const std::vector<vk::PresentModeKHR> &availablePresentModes);

  static vk::Extent2D chooseSwapchainExtent(const vk::SurfaceCapabilitiesKHR &capabilities, GLFWwindow *window);

  static std::vector<vk::ImageView> createSwapchainImageViews(const vk::Device &device, const std::vector<vk::Image> &images, const vk::Format &format);
  
  static vk::ImageView createImageView(const vk::Device &device, const vk::Image &image, const vk::Format &format, vk::ImageAspectFlags aspectFlags, uint32_t mipLevels);
};

}  // namespace rendering

#endif  // CALYPSO_INITIALISERS_H
