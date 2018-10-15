/**
 * @file      rasterize.h
 * @brief     CUDA-accelerated rasterization pipeline.
 * @authors   Skeleton code: Yining Karl Li, Kai Ninomiya, Shuai Shao (Shrek)
 * @date      2012-2016
 * @copyright University of Pennsylvania & STUDENT
 */

#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>

namespace tinygltf {
class Scene;
}

namespace {
typedef unsigned short VertexIndex;
typedef glm::vec3 VertexAttributePosition;
typedef glm::vec3 VertexAttributeNormal;
typedef glm::vec2 VertexAttributeTexcoord;
typedef unsigned char TextureData;

typedef unsigned char BufferByte;

enum PrimitiveType { Point = 1, Line = 2, Triangle = 3 };

struct VertexOut {
  glm::vec4 pos;

  // TODO: add new attributes to your VertexOut
  // The attributes listed below might be useful,
  // but always feel free to modify on your own

  glm::vec3 eyePos; // eye space position used for shading
  glm::vec3 eyeNor; // eye space normal used for shading, cuz normal will go
                    // wrong after perspective transformation
  // glm::vec3 col;
  glm::vec2 texcoord0;
  TextureData *dev_diffuseTex = NULL;
  int texWidth, texHeight;
  // ...
};

struct Primitive {
  PrimitiveType primitiveType = Triangle; // C++ 11 init
  VertexOut v[3];
  bool backface_culled = false;
};

struct Fragment {
  glm::vec3 color;

  // TODO: add new attributes to your Fragment
  // The attributes listed below might be useful,
  // but always feel free to modify on your own

  glm::vec3 eyePos; // eye space position used for shading
  glm::vec3 eyeNor;
  VertexAttributeTexcoord texcoord0;
  TextureData *dev_diffuseTex;

  // ...
};

struct PrimitiveDevBufPointers {
  int primitiveMode; // from tinygltfloader macro
  PrimitiveType primitiveType;
  int numPrimitives;
  int numIndices;
  int numVertices;

  // Vertex In, const after loaded
  VertexIndex *dev_indices;
  VertexAttributePosition *dev_position;
  VertexAttributeNormal *dev_normal;
  VertexAttributeTexcoord *dev_texcoord0;

  // Materials, add more attributes when needed
  TextureData *dev_diffuseTex;
  int diffuseTexWidth;
  int diffuseTexHeight;
  // TextureData* dev_specularTex;
  // TextureData* dev_normalTex;
  // ...

  // Vertex Out, vertex used for rasterization, this is changing every frame
  VertexOut *dev_verticesOut;

  // TODO: add more attributes when needed
};
} // namespace

struct ObjectData {
  int totalNumPrimitives = 0;
  Primitive *dev_primitives = NULL;
  bool is_copy = false;
  glm::vec3 transformation = {0.0f, 0.0f, -10.0f};
  bool is_deleted = false;
};

extern std::vector<ObjectData> objects;

void rasterizeInit(int width, int height);
void rasterizeSetBuffers(const tinygltf::Scene &scene);

void rasterize(uchar4 *pbo, const glm::mat4 &MVP, const glm::mat4 &MV,
               const glm::mat3 MV_normal, glm::vec3 &camera_pos);
void rasterizeFree();

void set_scene(int index);
void copy_object(int index);

void zero_frame_buffer();

void write_to_pbo(uchar4* pbo);
