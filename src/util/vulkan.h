//
// Created by Wyoming on 22/03/2021.
//

#ifndef CALYPSO_VULKAN_H
#define CALYPSO_VULKAN_H

#include <vulkan/vulkan.h>

namespace util {
class Vulkan {
 public:
  static VkResult CreateDebugUtilsMessengerEXT(
      VkInstance instance,
      const VkDebugUtilsMessengerCreateInfoEXT *pCreateInfo,
      const VkAllocationCallbacks *pAllocator,
      VkDebugUtilsMessengerEXT *pDebugMessenger);

  static void DestroyDebugUtilsMessengerEXT(
      VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger,
      const VkAllocationCallbacks *pAllocator) ;
};

}  // namespace util

#endif  // CALYPSO_VULKAN_H
