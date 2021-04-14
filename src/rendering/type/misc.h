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

struct SwapChainSupportDetails {
  vk::SurfaceCapabilitiesKHR capabilities;
  std::vector<vk::SurfaceFormatKHR> formats;
  std::vector<vk::PresentModeKHR> presentModes;
};

}  // namespace rendering::type

#endif  // CALYPSO_MISC_H
