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
#include <thrust/device_ptr.h>
#include <thrust/partition.h>
#include <util/checkCUDAError.h>
#include <util/tiny_gltf_loader.h>
#include "rasterizeTools.h"
#include "rasterize.h"
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>


#define LAMBERT_SHADING 1
#define BLINN_PHONG_SHADING 1
//#define BACKFACE_CULLING 1
#define BILINEAR_FILTERING 1

//happens by default now since added check
//#define COLOR_TRIANGLE_INTERPOLATION 1

static std::map<std::string, std::vector<PrimitiveDevBufPointers>> mesh2PrimitivesMap;


static int width = 0;
static int height = 0;

static int totalNumPrimitives = 0;
static Primitive* dev_primitives = NULL;
static Fragment* dev_fragmentBuffer = NULL;
static glm::vec3* dev_framebuffer = NULL;

static int* dev_depth = NULL; // you might need this buffer when doing depth test

//lights in scene
static glm::vec3* dev_lights = NULL; 
const int num_lights = 2;

//array of objects
std::vector<ObjectData> objects;

/**
 * Kernel that writes the image to the OpenGL PBO directly.
 */
__global__

void sendImageToPBO(uchar4* pbo, int w, int h, glm::vec3* image)
{
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * w);
  if (x < w && y < h)
  {
    glm::vec3 color;
    color.x = glm::clamp(image[index].x, 0.0f, 1.0f) * 255.0;
    color.y = glm::clamp(image[index].y, 0.0f, 1.0f) * 255.0;
    color.z = glm::clamp(image[index].z, 0.0f, 1.0f) * 255.0;
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

void render(int w, int h, Fragment* fragmentBuffer, glm::vec3* framebuffer, glm::vec3* lights, int num_lights, glm::vec3 camera_pos)
{
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * w);
  if (x < w && y < h)
  {
    glm::vec3 eye_pos = fragmentBuffer[index].eyePos;
    glm::vec3 eye_normal = fragmentBuffer[index].eyeNor;

    glm::vec3 pixel_color = fragmentBuffer[index].color;

    // TODO: adsd your fragment shader code here
#ifdef LAMBERT_SHADING
    for(int i = 0; i < num_lights; i++)
    {
      glm::vec3& light_source = lights[i];
      glm::vec3 light_direction = glm::normalize(light_source - eye_pos);
      float amount_of_light = glm::max(glm::dot(light_direction, eye_normal), 0.0f);
#ifdef BLINN_PHONG_SHADING
      glm::vec3 eye_direction = glm::normalize(camera_pos - eye_pos);
      glm::vec3 half_direction = glm::normalize(light_direction + eye_direction);
      amount_of_light = glm::pow(glm::max(glm::dot(light_direction, half_direction), 0.0f), 8.0f);
#endif
      pixel_color += fragmentBuffer[index].color * amount_of_light;
    }
#endif

    //hack to get multiple objects to work (check if don't overwrite if not black) DOESN"T CHECK FOR DEPTH BUFFER...
    if(framebuffer[index] == glm::vec3(0.0f))
    {
      framebuffer[index] = pixel_color;
    }
  }
}

/**
 * Called once at the beginning of the program to allocate memory.
 */
void rasterizeInit(int w, int h)
{
  width = w;
  height = h;
  cudaFree(dev_fragmentBuffer);
  cudaMalloc(&dev_fragmentBuffer, width * height * sizeof(Fragment));
  cudaMemset(dev_fragmentBuffer, 0, width * height * sizeof(Fragment));
  cudaFree(dev_framebuffer);
  cudaMalloc(&dev_framebuffer, width * height * sizeof(glm::vec3));
  cudaMemset(dev_framebuffer, 0, width * height * sizeof(glm::vec3));
  cudaFree(dev_depth);
  cudaMalloc(&dev_depth, width * height * sizeof(int));
  cudaFree(dev_lights);
  cudaMalloc(&dev_lights, num_lights * sizeof(glm::vec3));
  cudaMemset(dev_lights, 0, num_lights * sizeof(glm::vec3));

  //init lights here
  glm::vec3 cpu_lights[num_lights] =
  {
    { 2.0f, 2.0f, 2.0f },
    { -2.0f, 2.0f, 2.0f },
  };

  cudaMemcpy(dev_lights, cpu_lights, num_lights * sizeof(glm::vec3), cudaMemcpyHostToDevice);

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

void set_scene(int index)
{
  if(index >= 0 && index < objects.size())
  {
    dev_primitives = objects[index].dev_primitives;
    totalNumPrimitives = objects[index].totalNumPrimitives;
  }
}

void copy_object(int index)
{
  if (index >= 0 && index < objects.size())
  {
    //copy over pointer and primitives
    ObjectData object_data;
    object_data.dev_primitives = dev_primitives;
    object_data.totalNumPrimitives = totalNumPrimitives;
    object_data.is_copy = true;
    objects.push_back(object_data);
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
  //    for each primitive: 
  //      build device buffer of indices, materail, and each attributes
  //      and store these pointers in a map
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
          std::pair<std::string, std::vector<PrimitiveDevBufPointers>>(mesh.name, std::vector<PrimitiveDevBufPointers>()));
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

    //copy over pointer and primitives
    ObjectData object_data;
    object_data.dev_primitives = dev_primitives;
    object_data.totalNumPrimitives = totalNumPrimitives;
    objects.push_back(object_data);
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
    //textures
    primitive.dev_verticesOut[vid].dev_diffuseTex = 0;

    //check if textures exist
    if (primitive.dev_diffuseTex)
    {
      primitive.dev_verticesOut[vid].dev_diffuseTex = primitive.dev_diffuseTex;
      primitive.dev_verticesOut[vid].texWidth = primitive.diffuseTexWidth;
      primitive.dev_verticesOut[vid].texHeight = primitive.diffuseTexHeight;
      primitive.dev_verticesOut[vid].texcoord0 = primitive.dev_texcoord0[vid];
    }

    // TODO: Apply vertex transformation here
    // Multiply the MVP matrix for each vertex position, this will transform everything into clipping space
    // Then divide the pos by its w element to transform into NDC space
    // Finally transform x and y to viewport space

    //clip
    primitive.dev_verticesOut[vid].pos = MVP * glm::vec4(primitive.dev_position[vid], 1.0f);

    //ndc
    primitive.dev_verticesOut[vid].pos /= primitive.dev_verticesOut[vid].pos.w;

    //screen space
    const float width_ndc = static_cast<float>(width) * 0.5f;
    const float height_ndc = static_cast<float>(height) * 0.5f;
    primitive.dev_verticesOut[vid].pos.x = width_ndc * (primitive.dev_verticesOut[vid].pos.x + 1.0f);
    primitive.dev_verticesOut[vid].pos.y = height_ndc * (1.0f - primitive.dev_verticesOut[vid].pos.y);
    primitive.dev_verticesOut[vid].pos.z = 0.5f * (1.0f + primitive.dev_verticesOut[vid].pos.z);

    // TODO: Apply vertex assembly here
    // Assemble all attribute arraies into the primitive array
    primitive.dev_verticesOut[vid].eyeNor = MV_normal * primitive.dev_normal[vid];
    primitive.dev_verticesOut[vid].eyeNor = glm::normalize(primitive.dev_verticesOut[vid].eyeNor);
    primitive.dev_verticesOut[vid].eyePos = glm::vec3(MV * glm::vec4(primitive.dev_position[vid], 1.0f));
  }
}


static int curPrimitiveBeginId = 0;

__global__

void _primitiveAssembly(int numIndices, int curPrimitiveBeginId, Primitive* dev_primitives,
                        PrimitiveDevBufPointers primitive)
{
  // index id
  int iid = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (iid < numIndices)
  {
    // TODO: uncomment the following code for a start
    // This is primitive assembly for triangles
    int pid; // id for cur primitives vector
    if (primitive.primitiveMode == TINYGLTF_MODE_TRIANGLES)
    {
      pid = iid / (int)primitive.primitiveType;
      dev_primitives[pid + curPrimitiveBeginId].v[iid % (int)primitive.primitiveType]
        = primitive.dev_verticesOut[primitive.dev_indices[iid]];
    }
    // TODO: other primitive types (point, line)
  }
}

__global__ void backface_cull(int totalPrimitives, glm::vec3 camera, Primitive* primitives)
{
  int vid = blockIdx.x * blockDim.x + threadIdx.x;
  if (vid < totalPrimitives)
  {
    glm::vec3 triangle_pos[3] =
    {
      glm::vec3(primitives[vid].v[0].pos),
      glm::vec3(primitives[vid].v[1].pos),
      glm::vec3(primitives[vid].v[2].pos)
    };

    glm::vec3 dir_1 = triangle_pos[0] - triangle_pos[1];
    glm::vec3 dir_2 = triangle_pos[2] - triangle_pos[2];
    glm::vec3 triangle_normal = glm::cross(dir_1, dir_2);

    primitives[vid].backface_culled = false;

    if(glm::dot(camera, triangle_normal) < 0.0f)
    {
      primitives[vid].backface_culled = true;
    }
  }
}

//stream compaction for backface culling
struct HostDeviceSteamCompactionCallback {
  __host__ __device__ bool operator()(const Primitive &p) {
    return !p.backface_culled;
  };
};

__global__ void rasterize_triangles(int totalPrimitives, int width, int height, int* depths,
                                    Primitive* primitives, Fragment* fragments)
{
  int vid = blockIdx.x * blockDim.x + threadIdx.x;
  if (vid < totalPrimitives)
  {
    glm::vec3 triangle_pos[3] =
    {
      glm::vec3(primitives[vid].v[0].pos),
      glm::vec3(primitives[vid].v[1].pos),
      glm::vec3(primitives[vid].v[2].pos)
    };

    glm::vec2 triangle_texcoords[3] =
    {
      primitives[vid].v[0].texcoord0,
      primitives[vid].v[1].texcoord0,
      primitives[vid].v[2].texcoord0,
    };

    int texture_width = primitives[vid].v[0].texWidth;
    int texture_height = primitives[vid].v[0].texHeight;

    //for correct color interpolation
    glm::vec3 triangle_colors[3] =
    {
      glm::vec3(1.0f, 0.0f, 0.0f),
      glm::vec3(0.0f, 1.0f, 0.0f),
      glm::vec3(0.0f, 0.0f, 1.0f),
    };

    glm::vec3 eye_pos[3] =
    {
      primitives[vid].v[0].eyePos,
      primitives[vid].v[1].eyePos,
      primitives[vid].v[2].eyePos,
    };
    
    glm::vec3 eye_normal[3] =
    {
      primitives[vid].v[0].eyeNor,
      primitives[vid].v[1].eyeNor,
      primitives[vid].v[2].eyeNor,
    };

    //get aabb
    AABB triangle_aabb = getAABBForTriangle(triangle_pos);

    //clamp between screen size
    triangle_aabb = [width, height](int min_x, int max_x, int min_y, int max_y)
    {
      AABB result{};
      result.min.x = glm::clamp(min_x, 0, width - 1);
      result.max.x = glm::clamp(max_x, 0, width - 1);
      result.min.y = glm::clamp(min_y, 0, height - 1);
      result.max.y = glm::clamp(max_y, 0, height - 1);
      return result;
    }(triangle_aabb.min.x, triangle_aabb.max.x,
      triangle_aabb.min.y, triangle_aabb.max.y);

    //scanline using baycentric
    for (int x = triangle_aabb.min.x; x <= triangle_aabb.max.x; x++)
    {
      for (int y = triangle_aabb.min.y; y <= triangle_aabb.max.y; y++)
      {
        //caclulate baycentric (if pixel is on triangle)
        const glm::vec2 pixel_space{x, y};
        const glm::vec3 barycentric_coordinate = calculateBarycentricCoordinate(triangle_pos, pixel_space);

        if(isBarycentricCoordInBounds(barycentric_coordinate))
        {
          float depth = -getZAtCoordinate(barycentric_coordinate, triangle_pos);
          float depth_in_int = depth * 1000.0f;
          int pixel = y * width + x;

          //depth test (get the pixel closest)
          const int old_depth = atomicMin(&depths[pixel], depth_in_int);

          //fragment shading

          //check if depth was closer (draw pixel on top)
          if(old_depth != depths[pixel])
          {
            float eye_pos1_z = eye_pos[0].z;
            float eye_pos2_z = eye_pos[1].z;
            float eye_pos3_z = eye_pos[2].z;
            float bary_correct_x = barycentric_coordinate.x / eye_pos1_z;
            float bary_correct_y = barycentric_coordinate.y / eye_pos2_z;
            float bary_correct_z = barycentric_coordinate.z / eye_pos3_z;
            float perspective_correct_z = 1.0f / (bary_correct_x + bary_correct_y + bary_correct_z);

            //debugging depth
            //fragments[pixel].color = glm::vec3(depth);

            //normals
            //fragments[pixel].color = ;

            //perspective correct normal
            const glm::vec3 perspective_correct_eye_normal = 
              (
              barycentric_coordinate.x * (eye_normal[0] / eye_pos1_z) +
              barycentric_coordinate.y * (eye_normal[1] / eye_pos2_z) +
              barycentric_coordinate.z * (eye_normal[2] / eye_pos3_z)
              ) * perspective_correct_z;

            fragments[pixel].eyeNor = perspective_correct_eye_normal;

            //textures 

            //perspective correct texture coordinate
            const glm::vec2 perspective_correct_texcoord = 
              (
              barycentric_coordinate.x * (triangle_texcoords[0] / eye_pos1_z) +
              barycentric_coordinate.y * (triangle_texcoords[1] / eye_pos2_z) +
              barycentric_coordinate.z * (triangle_texcoords[2] / eye_pos3_z)
              ) * perspective_correct_z;

            fragments[pixel].texcoord0 = perspective_correct_texcoord;

            TextureData* diffuse_texture = primitives[vid].v->dev_diffuseTex;
            fragments[pixel].dev_diffuseTex = diffuse_texture;
            if(diffuse_texture)
            {
              auto sample_texture = [&](int u, int v)
              {
                int v_height = v * texture_width;
                int u_v_index = 3 * (u + v_height);
                glm::vec3 texture_color =
                {
                  diffuse_texture[u_v_index],
                  diffuse_texture[u_v_index + 1],
                  diffuse_texture[u_v_index + 2]
                };
                //put in range 0 -> 1
                texture_color /= 255.0f;
                return texture_color;
              };

              //bilinear
#ifdef BILINEAR_FILTERING
              float u_float = static_cast<float>(texture_width) * perspective_correct_texcoord[0];
              float v_float = static_cast<float>(texture_height) * perspective_correct_texcoord[1];

              //4 points
              int u_int = static_cast<int>(glm::floor(u_float));
              int v_int = static_cast<int>(glm::floor(v_float));
              int u_int_plus_one = glm::clamp(u_int + 1, 0, texture_width - 1);
              int v_int_plus_one = glm::clamp(v_int + 1, 0, texture_height - 1);

              //calculate difference (will be used in mixing
              float u_diff = u_float - static_cast<float>(u_int);
              float v_diff = v_float - static_cast<float>(v_int);

              //sample 4 points (bilinear mix between them)
              const auto sample_mix_1 = glm::mix(sample_texture(u_int, v_int), sample_texture(u_int, v_int_plus_one), v_diff);
              const auto sample_mix_2 = glm::mix(sample_texture(u_int_plus_one, v_int), sample_texture(u_int_plus_one, v_int_plus_one), v_diff);
              const auto sample_mix_final = glm::mix(sample_mix_1, sample_mix_2, u_diff);

              fragments[pixel].color = sample_mix_final;
#else
              //not bilinear
              int u = texture_width * perspective_correct_texcoord[0];
              int v = texture_height * perspective_correct_texcoord[1];
              fragments[pixel].color = sample_texture(u, v);
#endif
            } 
            //force color triangle interpolation (no texture)
            else
            {
              //perspective correct color
              const glm::vec3 perspective_correct_color =
              (
                barycentric_coordinate.x * (triangle_colors[0] / eye_pos1_z) +
                barycentric_coordinate.y * (triangle_colors[1] / eye_pos2_z) +
                barycentric_coordinate.z * (triangle_colors[2] / eye_pos3_z)
              ) * perspective_correct_z;
              fragments[pixel].color = perspective_correct_color;
            }
#ifdef COLOR_TRIANGLE_INTERPOLATION
            //perspective correct color
            const glm::vec3 perspective_correct_color = 
              (
              barycentric_coordinate.x * (triangle_colors[0] / eye_pos1_z) +
              barycentric_coordinate.y * (triangle_colors[1] / eye_pos2_z) +
              barycentric_coordinate.z * (triangle_colors[2] / eye_pos3_z)
              ) * perspective_correct_z;

            fragments[pixel].color = perspective_correct_color;
#endif
          }
        }
      }
    }
  }
}

int sideLength2d = 8;
dim3 blockSize2d(sideLength2d, sideLength2d);
dim3 blockCount2d((width - 1) / blockSize2d.x + 1,
                  (height - 1) / blockSize2d.y + 1);

/**
 * Perform rasterization.
 */
void rasterize(uchar4* pbo, const glm::mat4& MVP, const glm::mat4& MV, const glm::mat3 MV_normal, glm::vec3& camera_pos)
{
  blockCount2d = dim3((width - 1) / blockSize2d.x + 1,
                  (height - 1) / blockSize2d.y + 1);

  // Execute your rasterization pipeline here
  // (See README for rasterization pipeline outline.)
  // Vertex Process & primitive assembly
  {
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
        _vertexTransformAndAssembly << < numBlocksForVertices, numThreadsPerBlock >> >(
          p->numVertices, *p, MVP, MV, MV_normal, width, height);
        checkCUDAError("Vertex Processing");
        cudaDeviceSynchronize();
        _primitiveAssembly << < numBlocksForIndices, numThreadsPerBlock >> >
        (p->numIndices,
         curPrimitiveBeginId,
         dev_primitives,
         *p);
        checkCUDAError("Primitive Assembly");
        curPrimitiveBeginId += p->numPrimitives;
      }
    }
    checkCUDAError("Vertex Processing and Primitive Assembly");
  }
  cudaMemset(dev_fragmentBuffer, 0, width * height * sizeof(Fragment));
  initDepth << <blockCount2d, blockSize2d >> >(width, height, dev_depth);

  const int blockSize1d = 128;
  int remaining_primitives = totalNumPrimitives;
  dim3 num_triangles((remaining_primitives + blockSize1d - 1) / blockSize1d);

  //backface culling
#ifdef BACKFACE_CULLING
  backface_cull<<<num_triangles, blockSize1d>>>(remaining_primitives, camera_pos, dev_primitives);
  
  //stream compact away backface culled triangled using thrust
  thrust::device_ptr<Primitive> dev_primitive_ptr_start = thrust::device_pointer_cast(dev_primitives);
  thrust::device_ptr<Primitive> dev_primitive_ptr_end = thrust::device_pointer_cast(
    dev_primitives + remaining_primitives);

  //perform stream compaction  
  thrust::device_ptr<Primitive> new_dev_primitive_end = thrust::partition(
    dev_primitive_ptr_start, dev_primitive_ptr_end, HostDeviceSteamCompactionCallback());
  
  Primitive* dev_primitive_end = thrust::raw_pointer_cast(new_dev_primitive_end);

  //update the primitive counts
  remaining_primitives = dev_primitive_end - dev_primitives;
#endif

  // TODO: rasterize
  rasterize_triangles<<<num_triangles, blockSize1d>>>(remaining_primitives, width, height, dev_depth, dev_primitives, dev_fragmentBuffer);

  // Copy depthbuffer colors into framebuffer
  glm::vec3 camera_pos_in_MV = glm::vec3(MV * glm::vec4(camera_pos, 1.0f));
  render << <blockCount2d, blockSize2d >> >(width, height, dev_fragmentBuffer, dev_framebuffer, dev_lights, num_lights, camera_pos_in_MV);
  checkCUDAError("fragment shader");

}

void zero_frame_buffer()
{
  cudaMemset(dev_framebuffer, 0, width * height * sizeof(glm::vec3));
}

void write_to_pbo(uchar4* pbo)
{
  // Copy framebuffer into OpenGL buffer for OpenGL previewing
  sendImageToPBO<<<blockCount2d, blockSize2d>>>(pbo, width, height, dev_framebuffer);
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
  // cudaFree(dev_primitives);
  // dev_primitives = NULL;
  for(auto& object : objects)
  {
    free(object.dev_primitives);
  }
  cudaFree(dev_fragmentBuffer);
  dev_fragmentBuffer = NULL;
  cudaFree(dev_framebuffer);
  dev_framebuffer = NULL;
  cudaFree(dev_depth);
  dev_depth = NULL;
  cudaFree(dev_lights);
  dev_lights = NULL;
  checkCUDAError("rasterize Free");
}
