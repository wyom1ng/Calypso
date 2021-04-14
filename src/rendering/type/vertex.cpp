//
// Created by Wyoming on 01/04/2021.
//

#include "vertex.h"

namespace rendering::type {

vk::VertexInputBindingDescription Vertex::getBindingDescription() {
  vk::VertexInputBindingDescription binding_description;
  binding_description.binding = 0;
  binding_description.stride = sizeof(Vertex);
  binding_description.inputRate = vk::VertexInputRate::eVertex;

  return binding_description;
}

std::array<vk::VertexInputAttributeDescription, 3> Vertex::getAttributeDescriptions() {
  std::array<vk::VertexInputAttributeDescription, 3> attribute_descriptions;

  attribute_descriptions[0].binding = 0;
  attribute_descriptions[0].location = 0;
  attribute_descriptions[0].format = vk::Format::eR32G32B32Sfloat;
  attribute_descriptions[0].offset = offsetof(Vertex, pos);

  attribute_descriptions[1].binding = 0;
  attribute_descriptions[1].location = 1;
  attribute_descriptions[1].format = vk::Format::eR32G32B32Sfloat;
  attribute_descriptions[1].offset = offsetof(Vertex, colour);

  attribute_descriptions[2].binding = 0;
  attribute_descriptions[2].location = 2;
  attribute_descriptions[2].format = vk::Format::eR32G32Sfloat;
  attribute_descriptions[2].offset = offsetof(Vertex, texCoord);

  return attribute_descriptions;
}

bool Vertex::operator==(const Vertex &other) const { return pos == other.pos && colour == other.colour && texCoord == other.texCoord; }

}  // namespace rendering::type