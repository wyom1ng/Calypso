//
// Created by Wyoming on 14/04/2021.
//

#ifndef CALYPSO_MISC_H
#define CALYPSO_MISC_H

#include <optional>
#include <vector>
#include <vulkan/vulkan.hpp>

namespace rendering::type {

struct QueueFamilyIndices {
  std::optional<uint32_t> graphicsFamily;
  std::optional<uint32_t> presentFamily;
  std::optional<uint32_t> transferFamily;

  [[nodiscard]] bool isComplete() const { return graphicsFamily.has_value() && presentFamily.has_value() && transferFamily.has_value(); }
};

struct SwapchainSupportDetails {
  vk::SurfaceCapabilitiesKHR capabilities;
  std::vector<vk::SurfaceFormatKHR> formats;
  std::vector<vk::PresentModeKHR> presentModes;
};

struct SwapchainData {
  vk::SwapchainKHR swapchain;
  std::vector<vk::Image> images;
  std::vector<vk::ImageView> imageViews;
  vk::Format format;
  vk::Extent2D extent;
};

}  // namespace rendering::type

#endif  // CALYPSO_MISC_H
