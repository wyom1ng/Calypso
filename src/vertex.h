//
// Created by Wyoming on 01/04/2021.
//

#ifndef CALYPSO_VERTEX_H
#define CALYPSO_VERTEX_H

#include <glm/glm.hpp>
#include <vulkan/vulkan.hpp>

struct Vertex {
  glm::vec2 pos;
  glm::vec3 colour;
  glm::vec2 texCoord;

  static vk::VertexInputBindingDescription getBindingDescription();

  static std::array<vk::VertexInputAttributeDescription, 3> getAttributeDescriptions();
};

#endif  // CALYPSO_VERTEX_H
