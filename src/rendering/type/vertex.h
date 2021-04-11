//
// Created by Wyoming on 01/04/2021.
//

#ifndef CALYPSO_VERTEX_H
#define CALYPSO_VERTEX_H

#include <glm/glm.hpp>
#include <vulkan/vulkan.hpp>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>

namespace rendering::type {

struct Vertex {
  glm::vec3 pos;
  glm::vec3 colour;
  glm::vec2 texCoord;

  static vk::VertexInputBindingDescription getBindingDescription();

  static std::array<vk::VertexInputAttributeDescription, 3> getAttributeDescriptions();

  bool operator==(const Vertex &other) const { return pos == other.pos && colour == other.colour && texCoord == other.texCoord; }
};

}  // namespace rendering::types

namespace std {
template <>
struct hash<rendering::type::Vertex> {
  size_t operator()(rendering::type::Vertex const &vertex) const {
    return ((hash<glm::vec3>()(vertex.pos) ^ (hash<glm::vec3>()(vertex.colour) << 1)) >> 1) ^ (hash<glm::vec2>()(vertex.texCoord) << 1);
  }
};
}  // namespace std

#endif  // CALYPSO_VERTEX_H
