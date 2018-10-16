/**
 * @file      rasterize.cu
 * @brief     CUDA-accelerated rasterization pipeline.
 * @authors   Skeleton code: Yining Karl Li, Kai Ninomiya, Shuai Shao (Shrek)
 * @date      2012-2016
 * @copyright University of Pennsylvania & STUDENT
 */

#include <cmath>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/random.h>
#include <util/checkCUDAError.h>
#include <util/tiny_gltf_loader.h>
#include "rasterizeTools.h"
#include "rasterize.h"
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>

 // #define USE_LINE_SEGMENT_CHECK
#define USE_BARY_CHECK

#define DO_COLOR_LERP

namespace
{
  typedef unsigned short VertexIndex;
  typedef glm::vec3 VertexAttributePosition;
  typedef glm::vec3 VertexAttributeNormal;
  typedef glm::vec2 VertexAttributeTexcoord;
  typedef unsigned char TextureData;

  typedef unsigned char BufferByte;

  enum PrimitiveType
  {
    Point = 1,
    Line = 2,
    Triangle = 3
  };

  struct VertexOut
  {
    glm::vec4 pos;

    // TODO: add new attributes to your VertexOut
    // The attributes listed below might be useful, 
    // but always feel free to modify on your own

    glm::vec3 eyePos; // eye space position used for shading
    glm::vec3 eyeNor; // eye space normal used for shading, cuz normal will go wrong after perspective transformation
    glm::vec3 col;
    glm::vec2 texcoord0;
    TextureData* dev_diffuseTex = NULL;
    int diffuseTexWidth;
    int diffuseTexHeight;
  };

  struct Primitive
  {
    PrimitiveType primitiveType = Triangle; // C++ 11 init
    VertexOut v[3];
  };

  struct Fragment
  {
    glm::vec3 color;

    // TODO: add new attributes to your Fragment
    // The attributes listed below might be useful, 
    // but always feel free to modify on your own

    glm::vec3 pos;
    glm::vec3 normal;
    glm::vec2 uv;


    // VertexAttributeTexcoord texcoord0;
    TextureData* dev_diffuseTex;
    int diffuseTexWidth;
    int diffuseTexHeight;
  };

  struct PrimitiveDevBufPointers
  {
    int primitiveMode; //from tinygltfloader macro
    PrimitiveType primitiveType;
    int numPrimitives;
    int numIndices;
    int numVertices;

    // Vertex In, const after loaded
    VertexIndex* dev_indices;
    VertexAttributePosition* dev_position;
    VertexAttributeNormal* dev_normal;
    VertexAttributeTexcoord* dev_texcoord0;

    // Materials, add more attributes when needed
    TextureData* dev_diffuseTex;
    int diffuseTexWidth;
    int diffuseTexHeight;
    // TextureData* dev_specularTex;
    // TextureData* dev_normalTex;
    // ...

    // Vertex Out, vertex used for rasterization, this is changing every frame
    VertexOut* dev_verticesOut;

    // TODO: add more attributes when needed
  };
}

static std::map<std::string, std::vector<PrimitiveDevBufPointers>> mesh2PrimitivesMap;

static PrimitiveType GLOBAL_DRAW_MODE = PrimitiveType::Triangle;

static int width = 0;
static int height = 0;

static int baseWidth = 0;
static int baseHeight = 0;

static const int ALIASING_VALUE = 2;
static const glm::mat3 ALIASING_SCALE = glm::mat3(glm::scale(glm::mat4(), glm::vec3(ALIASING_VALUE)));

static int totalNumPrimitives = 0;
static Primitive* dev_primitives = NULL;
static Fragment* dev_fragmentBuffer = NULL;
static glm::vec3* dev_framebuffer = NULL;

static int* dev_depth = NULL; // you might need this buffer when doing depth test

/**
 * Kernel that writes the image to the OpenGL PBO directly.
 */
__global__

void sendImageToPBO(uchar4* pbo, int baseW, int baseH, int alias, glm::vec3* image)
{
  const int row = (blockIdx.x * blockDim.x) + threadIdx.x;
  const int col = (blockIdx.y * blockDim.y) + threadIdx.y;
  const int index = row + (col * baseW);

  if (row < baseW && col < baseH)
  {
    const int startX = (row * alias);
    const int startY = (col * alias);

    const int totalPartSize = alias * alias;

    const int screenWidth = baseW * alias;
    const int screenHeight = baseH * alias;

    glm::vec3 color = glm::vec3();

    for (int p = 0; p < alias; p++) {
      int x = startX + p;

      for (int q = 0; q < alias; q++) {
        int y = startY + q;
        int idx = x + (screenWidth * y);
        color += image[idx];
      }
    }

    color = color / (float)totalPartSize;

    color.x = glm::clamp(color.x, 0.0f, 1.0f) * 255.0;
    color.y = glm::clamp(color.y, 0.0f, 1.0f) * 255.0;
    color.z = glm::clamp(color.z, 0.0f, 1.0f) * 255.0;

    // Each thread writes one pixel location in the texture (textel)
    pbo[index].w = 0;
    pbo[index].x = color.x;
    pbo[index].y = color.y;
    pbo[index].z = color.z;
  }
}

/** 
* Writes fragment colors to the framebuffer
*/
__global__

void render(int w, int h, Fragment* fragmentBuffer, glm::vec3* framebuffer)
{
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * w);

  if (x < w && y < h)
  {
    const Fragment& frag = fragmentBuffer[index];

    float ambientTerm = 0.2f;

    glm::vec3 fragColor = frag.color;
    if (frag.dev_diffuseTex)
    {
      const int pixelX = glm::floor(frag.uv.x * frag.diffuseTexWidth);
      const int pixelY = glm::floor(frag.uv.y * frag.diffuseTexHeight);
      const int linearCoordinate = pixelX + (frag.diffuseTexWidth * pixelY);

      const int strideFormat = 3;
      const uint8_t red = *((uint8_t*)&frag.dev_diffuseTex[strideFormat * linearCoordinate]);
      const uint8_t green = *((uint8_t*)&frag.dev_diffuseTex[strideFormat * linearCoordinate + 1]);
      const uint8_t blue = *((uint8_t*)&frag.dev_diffuseTex[strideFormat * linearCoordinate + 2]);

      fragColor = glm::vec3(red / 255.0f, green / 255.0f, blue / 255.0f);
    }

    glm::vec3 lightVector = glm::normalize(glm::vec3(glm::vec3(5, 5, 0) - frag.pos));

    float diffuseTerm = glm::dot(lightVector, glm::normalize(frag.normal));
    diffuseTerm = glm::clamp(diffuseTerm, 0.0f, 1.0f);

    framebuffer[index] = (ambientTerm + diffuseTerm) * fragColor;

    // TODO: add your fragment shader code here
  }
}

/**
 * Called once at the beginning of the program to allocate memory.
 */
void rasterizeInit(int w, int h)
{
  width = w * ALIASING_VALUE;
  height = h * ALIASING_VALUE;

  baseWidth = w;
  baseHeight = h;

  cudaFree(dev_fragmentBuffer);
  cudaMalloc(&dev_fragmentBuffer, width * height * sizeof(Fragment));
  cudaMemset(dev_fragmentBuffer, 0, width * height * sizeof(Fragment));
  cudaFree(dev_framebuffer);
  cudaMalloc(&dev_framebuffer, width * height * sizeof(glm::vec3));
  cudaMemset(dev_framebuffer, 0, width * height * sizeof(glm::vec3));

  cudaFree(dev_depth);
  cudaMalloc(&dev_depth, width * height * sizeof(int));

  checkCUDAError("rasterizeInit");
}

__global__

void initDepth(int w, int h, int* depth)
{
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;

  if (x < w && y < h)
  {
    int index = x + (y * w);
    depth[index] = INT_MAX;
  }
}


/**
* kern function with support for stride to sometimes replace cudaMemcpy
* One thread is responsible for copying one component
*/
__global__

void _deviceBufferCopy(int N, BufferByte* dev_dst, const BufferByte* dev_src, int n, int byteStride, int byteOffset,
                       int componentTypeByteSize)
{
  // Attribute (vec3 position)
  // component (3 * float)
  // byte (4 * byte)

  // id of component
  int i = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (i < N)
  {
    int count = i / n;
    int offset = i - count * n; // which component of the attribute

    for (int j = 0; j < componentTypeByteSize; j++)
    {
      dev_dst[count * componentTypeByteSize * n
          + offset * componentTypeByteSize
          + j]

        =

        dev_src[byteOffset
          + count * (byteStride == 0 ? componentTypeByteSize * n : byteStride)
          + offset * componentTypeByteSize
          + j];
    }
  }
}

__global__

void _nodeMatrixTransform(
  int numVertices,
  VertexAttributePosition* position,
  VertexAttributeNormal* normal,
  glm::mat4 MV, glm::mat3 MV_normal)
{
  // vertex id
  int vid = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (vid < numVertices)
  {
    position[vid] = glm::vec3(MV * glm::vec4(position[vid], 1.0f));
    normal[vid] = glm::normalize(MV_normal * normal[vid]);
  }
}

glm::mat4 getMatrixFromNodeMatrixVector(const tinygltf::Node& n)
{
  glm::mat4 curMatrix(1.0);

  const std::vector<double>& m = n.matrix;
  if (m.size() > 0)
  {
    // matrix, copy it

    for (int i = 0; i < 4; i++)
    {
      for (int j = 0; j < 4; j++)
      {
        curMatrix[i][j] = (float)m.at(4 * i + j);
      }
    }
  }
  else
  {
    // no matrix, use rotation, scale, translation

    if (n.translation.size() > 0)
    {
      curMatrix[3][0] = n.translation[0];
      curMatrix[3][1] = n.translation[1];
      curMatrix[3][2] = n.translation[2];
    }

    if (n.rotation.size() > 0)
    {
      glm::mat4 R;
      glm::quat q;
      q[0] = n.rotation[0];
      q[1] = n.rotation[1];
      q[2] = n.rotation[2];

      R = glm::mat4_cast(q);
      curMatrix = curMatrix * R;
    }

    if (n.scale.size() > 0)
    {
      curMatrix = curMatrix * glm::scale(glm::vec3(n.scale[0], n.scale[1], n.scale[2]));
    }
  }

  return curMatrix;
}

void traverseNode(
  std::map<std::string, glm::mat4>& n2m,
  const tinygltf::Scene& scene,
  const std::string& nodeString,
  const glm::mat4& parentMatrix
)
{
  const tinygltf::Node& n = scene.nodes.at(nodeString);
  glm::mat4 M = parentMatrix * getMatrixFromNodeMatrixVector(n);
  n2m.insert(std::pair<std::string, glm::mat4>(nodeString, M));

  auto it = n.children.begin();
  auto itEnd = n.children.end();

  for (; it != itEnd; ++it)
  {
    traverseNode(n2m, scene, *it, M);
  }
}

void rasterizeSetBuffers(const tinygltf::Scene& scene)
{
  totalNumPrimitives = 0;

  std::map<std::string, BufferByte*> bufferViewDevPointers;

  // 1. copy all `bufferViews` to device memory
  {
    std::map<std::string, tinygltf::BufferView>::const_iterator it(
      scene.bufferViews.begin());
    std::map<std::string, tinygltf::BufferView>::const_iterator itEnd(
      scene.bufferViews.end());

    for (; it != itEnd; it++)
    {
      const std::string key = it->first;
      const tinygltf::BufferView& bufferView = it->second;
      if (bufferView.target == 0)
      {
        continue; // Unsupported bufferView.
      }

      const tinygltf::Buffer& buffer = scene.buffers.at(bufferView.buffer);

      BufferByte* dev_bufferView;
      cudaMalloc(&dev_bufferView, bufferView.byteLength);
      cudaMemcpy(dev_bufferView, &buffer.data.front() + bufferView.byteOffset, bufferView.byteLength,
                 cudaMemcpyHostToDevice);

      checkCUDAError("Set BufferView Device Mem");

      bufferViewDevPointers.insert(std::make_pair(key, dev_bufferView));
    }
  }


  // 2. for each mesh: 
  //		for each primitive: 
  //			build device buffer of indices, materail, and each attributes
  //			and store these pointers in a map
  {
    std::map<std::string, glm::mat4> nodeString2Matrix;
    auto rootNodeNamesList = scene.scenes.at(scene.defaultScene);

    {
      auto it = rootNodeNamesList.begin();
      auto itEnd = rootNodeNamesList.end();
      for (; it != itEnd; ++it)
      {
        traverseNode(nodeString2Matrix, scene, *it, glm::mat4(1.0f));
      }
    }


    // parse through node to access mesh

    auto itNode = nodeString2Matrix.begin();
    auto itEndNode = nodeString2Matrix.end();
    for (; itNode != itEndNode; ++itNode)
    {
      const tinygltf::Node& N = scene.nodes.at(itNode->first);
      const glm::mat4& matrix = itNode->second;
      const glm::mat3& matrixNormal = glm::transpose(glm::inverse(glm::mat3(matrix)));

      auto itMeshName = N.meshes.begin();
      auto itEndMeshName = N.meshes.end();

      for (; itMeshName != itEndMeshName; ++itMeshName)
      {
        const tinygltf::Mesh& mesh = scene.meshes.at(*itMeshName);

        auto res = mesh2PrimitivesMap.insert(
          std::pair<std::string, std::vector<PrimitiveDevBufPointers>>(
            mesh.name, std::vector<PrimitiveDevBufPointers>()));
        std::vector<PrimitiveDevBufPointers>& primitiveVector = (res.first)->second;

        // for each primitive
        for (size_t i = 0; i < mesh.primitives.size(); i++)
        {
          const tinygltf::Primitive& primitive = mesh.primitives[i];

          if (primitive.indices.empty())
            return;

          // TODO: add new attributes for your PrimitiveDevBufPointers when you add new attributes
          VertexIndex* dev_indices = NULL;
          VertexAttributePosition* dev_position = NULL;
          VertexAttributeNormal* dev_normal = NULL;
          VertexAttributeTexcoord* dev_texcoord0 = NULL;

          // ----------Indices-------------

          const tinygltf::Accessor& indexAccessor = scene.accessors.at(primitive.indices);
          const tinygltf::BufferView& bufferView = scene.bufferViews.at(indexAccessor.bufferView);
          BufferByte* dev_bufferView = bufferViewDevPointers.at(indexAccessor.bufferView);

          // assume type is SCALAR for indices
          int n = 1;
          int numIndices = indexAccessor.count;
          int componentTypeByteSize = sizeof(VertexIndex);
          int byteLength = numIndices * n * componentTypeByteSize;

          dim3 numThreadsPerBlock(128);
          dim3 numBlocks((numIndices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
          cudaMalloc(&dev_indices, byteLength);
          _deviceBufferCopy << <numBlocks, numThreadsPerBlock >> >(
            numIndices,
            (BufferByte*)dev_indices,
            dev_bufferView,
            n,
            indexAccessor.byteStride,
            indexAccessor.byteOffset,
            componentTypeByteSize);


          checkCUDAError("Set Index Buffer");


          // ---------Primitive Info-------

          // Warning: LINE_STRIP is not supported in tinygltfloader
          int numPrimitives;
          PrimitiveType primitiveType;
          switch (primitive.mode)
          {
          case TINYGLTF_MODE_TRIANGLES:
            primitiveType = PrimitiveType::Triangle;
            numPrimitives = numIndices / 3;
            break;
          case TINYGLTF_MODE_TRIANGLE_STRIP:
            primitiveType = PrimitiveType::Triangle;
            numPrimitives = numIndices - 2;
            break;
          case TINYGLTF_MODE_TRIANGLE_FAN:
            primitiveType = PrimitiveType::Triangle;
            numPrimitives = numIndices - 2;
            break;
          case TINYGLTF_MODE_LINE:
            primitiveType = PrimitiveType::Line;
            numPrimitives = numIndices / 2;
            break;
          case TINYGLTF_MODE_LINE_LOOP:
            primitiveType = PrimitiveType::Line;
            numPrimitives = numIndices + 1;
            break;
          case TINYGLTF_MODE_POINTS:
            primitiveType = PrimitiveType::Point;
            numPrimitives = numIndices;
            break;
          default:
            // output error
            break;
          };

          // GLOBAL_DRAW_MODE = primitiveType;

          // ----------Attributes-------------

          auto it(primitive.attributes.begin());
          auto itEnd(primitive.attributes.end());

          int numVertices = 0;
          // for each attribute
          for (; it != itEnd; it++)
          {
            const tinygltf::Accessor& accessor = scene.accessors.at(it->second);
            const tinygltf::BufferView& bufferView = scene.bufferViews.at(accessor.bufferView);

            int n = 1;
            if (accessor.type == TINYGLTF_TYPE_SCALAR)
            {
              n = 1;
            }
            else if (accessor.type == TINYGLTF_TYPE_VEC2)
            {
              n = 2;
            }
            else if (accessor.type == TINYGLTF_TYPE_VEC3)
            {
              n = 3;
            }
            else if (accessor.type == TINYGLTF_TYPE_VEC4)
            {
              n = 4;
            }

            BufferByte* dev_bufferView = bufferViewDevPointers.at(accessor.bufferView);
            BufferByte** dev_attribute = NULL;

            numVertices = accessor.count;
            int componentTypeByteSize;

            // Note: since the type of our attribute array (dev_position) is static (float32)
            // We assume the glTF model attribute type are 5126(FLOAT) here

            if (it->first.compare("POSITION") == 0)
            {
              componentTypeByteSize = sizeof(VertexAttributePosition) / n;
              dev_attribute = (BufferByte**)&dev_position;
            }
            else if (it->first.compare("NORMAL") == 0)
            {
              componentTypeByteSize = sizeof(VertexAttributeNormal) / n;
              dev_attribute = (BufferByte**)&dev_normal;
            }
            else if (it->first.compare("TEXCOORD_0") == 0)
            {
              componentTypeByteSize = sizeof(VertexAttributeTexcoord) / n;
              dev_attribute = (BufferByte**)&dev_texcoord0;
            }

            std::cout << accessor.bufferView << "  -  " << it->second << "  -  " << it->first << '\n';

            dim3 numThreadsPerBlock(128);
            dim3 numBlocks((n * numVertices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
            int byteLength = numVertices * n * componentTypeByteSize;
            cudaMalloc(dev_attribute, byteLength);

            _deviceBufferCopy << <numBlocks, numThreadsPerBlock >> >(
              n * numVertices,
              *dev_attribute,
              dev_bufferView,
              n,
              accessor.byteStride,
              accessor.byteOffset,
              componentTypeByteSize);

            std::string msg = "Set Attribute Buffer: " + it->first;
            checkCUDAError(msg.c_str());
          }

          // malloc for VertexOut
          VertexOut* dev_vertexOut;
          cudaMalloc(&dev_vertexOut, numVertices * sizeof(VertexOut));
          checkCUDAError("Malloc VertexOut Buffer");

          // ----------Materials-------------

          // You can only worry about this part once you started to 
          // implement textures for your rasterizer
          TextureData* dev_diffuseTex = NULL;
          int diffuseTexWidth = 0;
          int diffuseTexHeight = 0;
          if (!primitive.material.empty())
          {
            const tinygltf::Material& mat = scene.materials.at(primitive.material);
            printf("material.name = %s\n", mat.name.c_str());

            if (mat.values.find("diffuse") != mat.values.end())
            {
              std::string diffuseTexName = mat.values.at("diffuse").string_value;
              if (scene.textures.find(diffuseTexName) != scene.textures.end())
              {
                const tinygltf::Texture& tex = scene.textures.at(diffuseTexName);
                if (scene.images.find(tex.source) != scene.images.end())
                {
                  const tinygltf::Image& image = scene.images.at(tex.source);

                  size_t s = image.image.size() * sizeof(TextureData);
                  cudaMalloc(&dev_diffuseTex, s);
                  cudaMemcpy(dev_diffuseTex, &image.image.at(0), s, cudaMemcpyHostToDevice);

                  diffuseTexWidth = image.width;
                  diffuseTexHeight = image.height;

                  checkCUDAError("Set Texture Image data");
                }
              }
            }

            // TODO: write your code for other materails
            // You may have to take a look at tinygltfloader
            // You can also use the above code loading diffuse material as a start point 
          }


          // ---------Node hierarchy transform--------
          cudaDeviceSynchronize();

          dim3 numBlocksNodeTransform((numVertices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
          _nodeMatrixTransform << <numBlocksNodeTransform, numThreadsPerBlock >> >(
            numVertices,
            dev_position,
            dev_normal,
            matrix,
            matrixNormal);

          checkCUDAError("Node hierarchy transformation");

          // at the end of the for loop of primitive
          // push dev pointers to map
          primitiveVector.push_back(PrimitiveDevBufPointers{
            primitive.mode,
            primitiveType,
            numPrimitives,
            numIndices,
            numVertices,

            dev_indices,
            dev_position,
            dev_normal,
            dev_texcoord0,

            dev_diffuseTex,
            diffuseTexWidth,
            diffuseTexHeight,

            dev_vertexOut //VertexOut
          });

          totalNumPrimitives += numPrimitives;
        } // for each primitive
      } // for each mesh
    } // for each node
  }


  // 3. Malloc for dev_primitives
  {
    cudaMalloc(&dev_primitives, totalNumPrimitives * sizeof(Primitive));
  }


  // Finally, cudaFree raw dev_bufferViews
  {
    std::map<std::string, BufferByte*>::const_iterator it(bufferViewDevPointers.begin());
    std::map<std::string, BufferByte*>::const_iterator itEnd(bufferViewDevPointers.end());

    //bufferViewDevPointers

    for (; it != itEnd; it++)
    {
      cudaFree(it->second);
    }

    checkCUDAError("Free BufferView Device Mem");
  }
}


__global__

void _vertexTransformAndAssembly(
  int numVertices,
  PrimitiveDevBufPointers primitive,
  glm::mat4 MVP, glm::mat4 MV, glm::mat3 MV_normal,
  int width, int height)
{
  // vertex id
  int vid = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (vid < numVertices)
  {
    // TODO: Apply vertex transformation here
    // Multiply the MVP matrix for each vertex position, this will transform everything into clipping space
    // Then divide the pos by its w element to transform into NDC space
    // Finally transform x and y to viewport space

    const glm::vec3 devicePosition = primitive.dev_position[vid];
    glm::vec4 screenPosition = MVP * glm::vec4(devicePosition, 1.0f); // CLIP SPACE
    screenPosition /= screenPosition.w; // NDC SPACE
    screenPosition.x = 0.5f * width * (1.0f + screenPosition.x); // VIEWPORT SPACE
    screenPosition.y = 0.5f * height * (1.0f - screenPosition.y);

    primitive.dev_verticesOut[vid].pos = screenPosition;
    primitive.dev_verticesOut[vid].col = glm::vec3(1, 0, 0); // TODO: red
    primitive.dev_verticesOut[vid].eyePos = glm::vec3(MV * glm::vec4(devicePosition, 1.0f));
    primitive.dev_verticesOut[vid].eyeNor = MV_normal * primitive.dev_normal[vid];
    primitive.dev_verticesOut[vid].dev_diffuseTex = primitive.dev_diffuseTex;
    if (primitive.dev_texcoord0)
    {
      primitive.dev_verticesOut[vid].texcoord0 = primitive.dev_texcoord0[vid];
    }

    primitive.dev_verticesOut[vid].diffuseTexWidth = primitive.diffuseTexWidth;
    primitive.dev_verticesOut[vid].diffuseTexHeight = primitive.diffuseTexHeight;

    // TODO: Apply vertex assembly here
    // Assemble all attribute arraies into the primitive array
  }
}


static int curPrimitiveBeginId = 0;

__global__ void _primitiveAssembly(int numIndices, int curPrimitiveBeginId, Primitive* dev_primitives,
                                   PrimitiveDevBufPointers primitive, PrimitiveType drawMode)
{
  // index id
  int iid = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (iid < numIndices)
  {
    // TODO: uncomment the following code for a start
    // This is primitive assembly for triangles

    int pid; // id for cur primitives vector
    // if (drawMode == TINYGLTF_MODE_TRIANGLES)
    // {
      pid = iid / (int)primitive.primitiveType;
      dev_primitives[pid + curPrimitiveBeginId].primitiveType = drawMode;
      dev_primitives[pid + curPrimitiveBeginId].v[iid % (int)primitive.primitiveType] = primitive.dev_verticesOut[primitive.dev_indices[iid]];
    // }


    // TODO: other primitive types (point, line)
  }
}

__device__ void ClampRange(float& actualStart, float& actualEnd, float targetStart, float targetEnd)
{
  if (actualStart < targetStart)
  {
    actualStart = targetStart;
  }

  if (actualEnd > targetEnd)
  {
    actualEnd = targetEnd;
  }
}

__device__ void ClampRangeInt(int& actualStart, int& actualEnd, int targetStart, int targetEnd)
{
  if (actualStart < targetStart)
  {
    actualStart = targetStart;
  }

  if (actualEnd > targetEnd)
  {
    actualEnd = targetEnd;
  }
}

__device__ bool CheckLineSegmentIntersect(glm::vec2 startPoint, glm::vec2 endPoint, int c, float slope, float* xCoord)
{
  /*----------  Slope 0 Check  ----------*/
  if (slope > -EPSILON && slope < EPSILON)
  {
    return false;
  }

  float yIntercept = static_cast<float>(c);

  // Incoming Line: y = c
  float y1 = startPoint.y;
  float y2 = endPoint.y;

  float maxY = y1 > y2 ? y1 : y2;
  float minY = y1 > y2 ? y2 : y1;

  if (yIntercept <= minY)
  {
    return false;
  }

  if (yIntercept > maxY)
  {
    return false;
  }

  if (slope == INFINITY)
  {
    (*xCoord) = startPoint.x;
    return true;
  }

  // y = m(x - p1.x) + p1.y
  //  Solve for y = c
  float x = ((yIntercept - startPoint.y) / slope) + startPoint.x;
  (*xCoord) = x;
  return true;
}

__device__ float GetLineSegmentSlope(const glm::vec2& startPoint, const glm::vec2& endPoint)
{
  // x2 - x1
  const float denom = endPoint[0] - startPoint[0];

  // y2 - y1
  const float num = endPoint[1] - startPoint[1];

  if (denom > -EPSILON && denom < EPSILON)
  {
    return INFINITY;
  }

  const float slope = num / denom;
  return slope;
}

__device__ bool BoundingBoxContains(const BoundingBox& box, float x, float y)
{
  if (x < box.min.x - EPSILON || x > box.max.x + EPSILON)
  {
    return false;
  }

  if (y < box.min.y - EPSILON || y > box.max.y + EPSILON)
  {
    return false;
  }

  return true;
}


__device__ bool CalculateIntersection(const glm::vec2& p0,
                                      const glm::vec2& p1,
                                      const glm::vec2& p2,
                                      float slope0,
                                      float slope1,
                                      float slope2,
                                      const BoundingBox& box,
                                      float& startX,
                                      float& endX,
                                      int yIntercept
)
{
  float xResult1 = 0.0f;
  float xResult2 = 0.0f;

  float x1 = 0.0f;
  float x2 = 0.0f;
  float x3 = 0.0f;

  const bool result1 = CheckLineSegmentIntersect(p0, p1, yIntercept, slope0, &x1);
  const bool result2 = CheckLineSegmentIntersect(p1, p2, yIntercept, slope1, &x2);
  const bool result3 = CheckLineSegmentIntersect(p2, p0, yIntercept, slope2, &x3);

  int pointsCount = 0;

  if (result1 && BoundingBoxContains(box, x1, yIntercept))
  {
    pointsCount++;
    xResult1 = x1;
  }

  if (result2 && BoundingBoxContains(box, x2, yIntercept))
  {
    pointsCount++;

    if (pointsCount == 2)
    {
      xResult2 = x2;
    }
    else
    {
      xResult1 = x2;
    }
  }

  if (result3 && BoundingBoxContains(box, x3, yIntercept))
  {
    pointsCount++;
    xResult2 = x3;
  }

  if (pointsCount == 2)
  {
    startX = xResult1 > xResult2 ? xResult2 : xResult1;
    endX = xResult1 > xResult2 ? xResult1 : xResult2;

    startX = ceil(startX);
    endX = floor(endX);

    return true;
  }

  return false;
}

__device__ void TryStoreFragment(const Primitive& target, float xCoord, int yCoord, int screenWidth, int screenHeight,
                                 const VertexOut& v1, const VertexOut& v2, const VertexOut& v3,
                                 const glm::vec3& baryCoordinates, int pixelIndex, int* depth, Fragment* fragmentBuffer)
{
  const float ratio1 = baryCoordinates.x;
  const float ratio2 = baryCoordinates.y;
  const float ratio3 = baryCoordinates.z;

  // pos[2] holds NDC Z [0,1]
  const float fragmentDepth = 1.0f / ((ratio1 * (1.0f / v1.pos[2])) + (ratio2 * (1.0f / v2.pos[2])) + (ratio3 * (1.0f /
    v3.pos[2])));
  const int fragmentIntegerDepth = fragmentDepth * INT_MAX;

#ifdef DO_COLOR_LERP
  const glm::vec3 interpolatedColor = fragmentDepth * ((ratio1 * (glm::vec3(1,0,0) / v1.pos[2])) + (ratio2 * (glm::vec3(0,1,0) / v2.
    pos[2])) + (ratio3 * (glm::vec3(0,0,1) / v3.pos[2])));
#else
  const glm::vec3 interpolatedColor = fragmentDepth * ((ratio1 * (v1.col / v1.pos[2])) + (ratio2 * (v2.col / v2.
    pos[2])) + (ratio3 * (v3.col / v3.pos[2])));
#endif

  const glm::vec2 interpolatedUV = fragmentDepth * ((ratio1 * (v1.texcoord0 / v1.pos[2])) + (ratio2 * (v2.texcoord0 / v2
    .pos[2])) + (ratio3 * (v3.texcoord0 / v3.pos[2])));
  const glm::vec3 interpolatedEyeNormal = fragmentDepth * ((ratio1 * (v1.eyeNor / v1.pos[2])) + (ratio2 * (v2.eyeNor /
    v2.pos[2])) + (ratio3 * (v3.eyeNor / v3.pos[2])));
  const glm::vec3 interpolatedEyePos = fragmentDepth * ((ratio1 * (v1.eyePos / v1.pos[2])) + (ratio2 * (v2.eyePos / v2.
    pos[2])) + (ratio3 * (v3.eyePos / v3.pos[2])));

  Fragment targetFragment;
  targetFragment.color = interpolatedColor;
  targetFragment.uv = interpolatedUV;
  targetFragment.normal = interpolatedEyeNormal;
  targetFragment.pos = interpolatedEyePos;
  targetFragment.dev_diffuseTex = v1.dev_diffuseTex;
  targetFragment.diffuseTexWidth = v1.diffuseTexWidth;
  targetFragment.diffuseTexHeight = v1.diffuseTexHeight;

  const int minDepth = atomicMin(&depth[pixelIndex], fragmentIntegerDepth);

  if (minDepth > fragmentIntegerDepth)
  {
    depth[pixelIndex] = fragmentIntegerDepth;
    fragmentBuffer[pixelIndex] = targetFragment;
  }
}

__device__ void TryStoreFragmentLine(const Primitive& target, float xCoord, int yCoord, int screenWidth, int screenHeight,
  const VertexOut& v1, const VertexOut& v2, int pixelIndex, int* depth, Fragment* fragmentBuffer)
{
  // // pos[2] holds NDC Z [0,1]
  // const float fragmentDepth = 1.0f / ((ratio1 * (1.0f / v1.pos[2])) + (ratio2 * (1.0f / v2.pos[2])) + (ratio3 * (1.0f /
  //   v3.pos[2])));
  // const int fragmentIntegerDepth = fragmentDepth * INT_MAX;
  //
  // const glm::vec2 interpolatedUV = fragmentDepth * ((ratio1 * (v1.texcoord0 / v1.pos[2])) + (ratio2 * (v2.texcoord0 / v2
  //   .pos[2])) + (ratio3 * (v3.texcoord0 / v3.pos[2])));
  // const glm::vec3 interpolatedEyeNormal = fragmentDepth * ((ratio1 * (v1.eyeNor / v1.pos[2])) + (ratio2 * (v2.eyeNor /
  //   v2.pos[2])) + (ratio3 * (v3.eyeNor / v3.pos[2])));
  // const glm::vec3 interpolatedEyePos = fragmentDepth * ((ratio1 * (v1.eyePos / v1.pos[2])) + (ratio2 * (v2.eyePos / v2.
  //   pos[2])) + (ratio3 * (v3.eyePos / v3.pos[2])));
  //
  // Fragment targetFragment;
  // targetFragment.color = glm::vec3(1, 0, 0);
  // targetFragment.uv = interpolatedUV;
  // targetFragment.normal = interpolatedEyeNormal;
  // targetFragment.pos = interpolatedEyePos;
  // targetFragment.dev_diffuseTex = v1.dev_diffuseTex;
  // targetFragment.diffuseTexWidth = v1.diffuseTexWidth;
  // targetFragment.diffuseTexHeight = v1.diffuseTexHeight;
  //
  // const int minDepth = atomicMin(&depth[pixelIndex], fragmentIntegerDepth);
  //
  // if (minDepth > fragmentIntegerDepth)
  // {
  //   depth[pixelIndex] = fragmentIntegerDepth;
  //   fragmentBuffer[pixelIndex] = targetFragment;
  // }
}

__device__ void TryStoreFragmentPoint(const Primitive& target, float xCoord, int yCoord, int screenWidth, int screenHeight,
  const VertexOut& v1, int pixelIndex, int* depth, Fragment* fragmentBuffer)
{
  // pos[2] holds NDC Z [0,1]
  const float fragmentDepth = v1.pos[2];
  const int fragmentIntegerDepth = fragmentDepth * INT_MAX;
  
  const glm::vec2 interpolatedUV = v1.texcoord0;
  const glm::vec3 interpolatedEyeNormal = v1.eyeNor;
  const glm::vec3 interpolatedEyePos = v1.eyePos;
  const glm::vec3 interpolatedColor = v1.col;
  
  Fragment targetFragment;
  targetFragment.color = interpolatedColor;
  targetFragment.uv = interpolatedUV;
  targetFragment.normal = interpolatedEyeNormal;
  targetFragment.pos = interpolatedEyePos;
  targetFragment.dev_diffuseTex = v1.dev_diffuseTex;
  targetFragment.diffuseTexWidth = v1.diffuseTexWidth;
  targetFragment.diffuseTexHeight = v1.diffuseTexHeight;
  
  const int minDepth = atomicMin(&depth[pixelIndex], fragmentIntegerDepth);
  
  if (minDepth > fragmentIntegerDepth)
  {
    depth[pixelIndex] = fragmentIntegerDepth;
    fragmentBuffer[pixelIndex] = targetFragment;
  }
}

__global__ void _rasterizeTriangles(int numPrimitives, Primitive* dev_primitives, int screenWidth, int screenHeight, int* depth,
                            Fragment* fragmentBuffer)
{
  // primitive id
  int primtiveId = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (primtiveId >= numPrimitives)
  {
    return;
  }

  const Primitive& target = dev_primitives[primtiveId];

  const glm::vec2 p0 = glm::vec2(target.v[0].pos[0], target.v[0].pos[1]);
  const glm::vec2 p1 = glm::vec2(target.v[1].pos[0], target.v[1].pos[1]);
  const glm::vec2 p2 = glm::vec2(target.v[2].pos[0], target.v[2].pos[1]);

  const BoundingBox boundingBox = getBoundingBoxForTriangle(p0, p1, p2);

#ifdef USE_LINE_SEGMENT_CHECK
int rasterStartY = floor(boundingBox.min.y);
int rasterEndY = ceil(boundingBox.max.y);
ClampRangeInt(rasterStartY, rasterEndY, 0, screenHeight - 1);

const float slope0 = GetLineSegmentSlope(p0, p1);
const float slope1 = GetLineSegmentSlope(p1, p2);
const float slope2 = GetLineSegmentSlope(p2, p0);

for (int yValue = rasterStartY; yValue <= rasterEndY; yValue += 1) {


  float rasterStartX = 0;
  float rasterEndX = 0;

  const bool result = CalculateIntersection(p0, p1, p2, slope0, slope1, slope2, boundingBox, rasterStartX, rasterEndX, yValue);

  if (!result) {
    continue;
  }

  ClampRange(rasterStartX, rasterEndX, 0, screenWidth - 1);

  for (int xValue = rasterStartX; xValue <= rasterEndX; ++xValue) {
    const glm::vec3 baryCoordinates = calculateBarycentricCoordinate(p0, p1, p2, glm::vec2(xValue, yValue));
    const int pixelIndex = xValue + (yValue * screenWidth);
    TryStoreFragment(target, xValue, yValue, screenWidth, screenHeight, target.v[0], target.v[1], target.v[2], baryCoordinates, pixelIndex, depth, fragmentBuffer);
  }
}
#endif

#ifdef USE_BARY_CHECK
  int rasterStartX = floor(boundingBox.min.x);
  int rasterEndX = ceil(boundingBox.max.x);

  ClampRangeInt(rasterStartX, rasterEndX, 0, screenWidth - 1);

  for (int xValue = rasterStartX; xValue <= rasterEndX; ++xValue)
  {
    int rasterStartY = floor(boundingBox.min.y);
    int rasterEndY = ceil(boundingBox.max.y);
    ClampRangeInt(rasterStartY, rasterEndY, 0, screenHeight - 1);

    for (int yValue = rasterStartY; yValue <= rasterEndY; yValue += 1)
    {
      const glm::vec3 baryCoordinates = calculateBarycentricCoordinate(p0, p1, p2, glm::vec2(xValue, yValue));

      if (!isBarycentricCoordInBounds(baryCoordinates))
      {
        continue;
      }

      const int pixelIndex = xValue + (yValue * screenWidth);
      TryStoreFragment(target, xValue, yValue, screenWidth, screenHeight, target.v[0], target.v[1], target.v[2],
                       baryCoordinates, pixelIndex, depth, fragmentBuffer);
    }
  }
#endif
}

__global__ void _rasterizeLines(int numPrimitives, Primitive* dev_primitives, int screenWidth, int screenHeight, int* depth,
  Fragment* fragmentBuffer)
{
  // primitive id
  int primtiveId = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (primtiveId >= numPrimitives)
  {
    return;
  }

  const Primitive& target = dev_primitives[primtiveId];

  const glm::vec2 p0 = glm::vec2(target.v[0].pos[0], target.v[0].pos[1]);
  const glm::vec2 p1 = glm::vec2(target.v[1].pos[0], target.v[1].pos[1]);

  const BoundingBox boundingBox = getBoundingBoxForLine(p0, p1);

  int rasterStartY = floor(boundingBox.min.y);
  int rasterEndY = ceil(boundingBox.max.y);
  ClampRangeInt(rasterStartY, rasterEndY, 0, screenHeight - 1);

  const float slope0 = GetLineSegmentSlope(p0, p1);

  for (int yValue = rasterStartY; yValue <= rasterEndY; yValue += 1) {
    float xIntercept;

    const bool doesIntersect = CheckLineSegmentIntersect(p0, p1, yValue, slope0, &xIntercept);

    if (!doesIntersect)
    {
      continue;
    }

    const int xValue = (int)glm::clamp(xIntercept, 0.0f, float(screenWidth - 1));

    const int pixelIndex = xValue + (yValue * screenWidth);
    TryStoreFragmentLine(target, xValue, yValue, screenWidth, screenHeight, target.v[0], target.v[1], pixelIndex, depth, fragmentBuffer);
  }
}

__global__ void _rasterizePoints(int numPrimitives, Primitive* dev_primitives, int screenWidth, int screenHeight, int* depth,
  Fragment* fragmentBuffer)
{
  // primitive id
  int primtiveId = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (primtiveId >= numPrimitives)
  {
    return;
  }

  const Primitive& target = dev_primitives[primtiveId];

  const glm::vec2 p0 = glm::vec2(target.v[0].pos[0], target.v[0].pos[1]);

  const int xValue = glm::clamp((int)glm::round(p0.x), 0, screenWidth - 1);
  const int yValue = glm::clamp((int)glm::round(p0.y), 0, screenHeight - 1);

  const int pixelIndex = xValue + (yValue * screenWidth);
  TryStoreFragmentPoint(target, xValue, yValue, screenWidth, screenHeight, target.v[0], pixelIndex, depth, fragmentBuffer);
}

/**
 * Perform rasterization.
 */
void rasterize(uchar4* pbo, const glm::mat4& MVP, const glm::mat4& MV, const glm::mat3 MV_normal)
{
  int sideLength2d = 8;
  dim3 blockSize2d(sideLength2d, sideLength2d);
  dim3 blockCount2d((width - 1) / blockSize2d.x + 1,
                    (height - 1) / blockSize2d.y + 1);

  // Execute your rasterization pipeline here
  // (See README for rasterization pipeline outline.)

  // Vertex Process & primitive assembly
  curPrimitiveBeginId = 0;
  dim3 numThreadsPerBlock(128);

  auto it = mesh2PrimitivesMap.begin();
  auto itEnd = mesh2PrimitivesMap.end();

  for (; it != itEnd; ++it)
  {
    auto p = (it->second).begin(); // each primitive
    auto pEnd = (it->second).end();
    for (; p != pEnd; ++p)
    {
      dim3 numBlocksForVertices((p->numVertices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
      dim3 numBlocksForIndices((p->numIndices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);

      _vertexTransformAndAssembly <<< numBlocksForVertices, numThreadsPerBlock >>>(
        p->numVertices, *p, MVP, MV, MV_normal, width, height);
      checkCUDAError("Vertex Processing");
      cudaDeviceSynchronize();
      _primitiveAssembly <<< numBlocksForIndices, numThreadsPerBlock >>>
      (p->numIndices,
       curPrimitiveBeginId,
       dev_primitives,
       *p,
        GLOBAL_DRAW_MODE);
      checkCUDAError("Primitive Assembly");

      curPrimitiveBeginId += p->numPrimitives;
    }
  }

  checkCUDAError("Vertex Processing and Primitive Assembly");

  cudaMemset(dev_fragmentBuffer, 0, width * height * sizeof(Fragment));
  initDepth <<<blockCount2d, blockSize2d >>>(width, height, dev_depth);

  // TODO: rasterize
  const int blockSize1d = 512;
  dim3 numRasterizeBlocks = (curPrimitiveBeginId + blockSize1d - 1) / blockSize1d;

  if (GLOBAL_DRAW_MODE == Triangle) {
    _rasterizeTriangles <<< numRasterizeBlocks, blockSize1d >>> (curPrimitiveBeginId, dev_primitives, width, height, dev_depth,
      dev_fragmentBuffer);
  }
  else if (GLOBAL_DRAW_MODE == Line) {
    _rasterizeLines <<< numRasterizeBlocks, blockSize1d >>> (curPrimitiveBeginId, dev_primitives, width, height, dev_depth,
      dev_fragmentBuffer);
  }
  else if (GLOBAL_DRAW_MODE == Point) {
    _rasterizePoints <<< numRasterizeBlocks, blockSize1d >>> (curPrimitiveBeginId, dev_primitives, width, height, dev_depth,
      dev_fragmentBuffer);
  }

  // Copy depthbuffer colors into framebuffer
  render <<< blockCount2d, blockSize2d >>>(width, height, dev_fragmentBuffer, dev_framebuffer);
  checkCUDAError("fragment shader");
  // Copy framebuffer into OpenGL buffer for OpenGL previewing
  sendImageToPBO<<<blockCount2d, blockSize2d>>>(pbo, baseWidth, baseHeight, ALIASING_VALUE, dev_framebuffer);
  checkCUDAError("copy render result to pbo");
}

/**
 * Called once at the end of the program to free CUDA memory.
 */
void rasterizeFree()
{
  // deconstruct primitives attribute/indices device buffer

  auto it(mesh2PrimitivesMap.begin());
  auto itEnd(mesh2PrimitivesMap.end());
  for (; it != itEnd; ++it)
  {
    for (auto p = it->second.begin(); p != it->second.end(); ++p)
    {
      cudaFree(p->dev_indices);
      cudaFree(p->dev_position);
      cudaFree(p->dev_normal);
      cudaFree(p->dev_texcoord0);
      cudaFree(p->dev_diffuseTex);

      cudaFree(p->dev_verticesOut);


      //TODO: release other attributes and materials
    }
  }

  ////////////

  cudaFree(dev_primitives);
  dev_primitives = NULL;

  cudaFree(dev_fragmentBuffer);
  dev_fragmentBuffer = NULL;

  cudaFree(dev_framebuffer);
  dev_framebuffer = NULL;

  cudaFree(dev_depth);
  dev_depth = NULL;

  checkCUDAError("rasterize Free");
}
