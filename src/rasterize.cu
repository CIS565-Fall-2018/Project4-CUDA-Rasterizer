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
#include <chrono>

#define BACKFACE_CULL 1
#define MUTEX 1
#define NORMALS 0
//#define TRIANGLES --> (lines||points) == 0;
#define LINES 1
#define POINTS 0
#define BILINEAR 1

#define TIME 1

namespace {

	typedef unsigned short VertexIndex;
	typedef glm::vec3 VertexAttributePosition;
	typedef glm::vec3 VertexAttributeNormal;
	typedef glm::vec2 VertexAttributeTexcoord;
	typedef unsigned char TextureData;

	typedef unsigned char BufferByte;

	enum PrimitiveType{
		Point = 1,
		Line = 2,
		Triangle = 3
	};

	struct VertexOut {
		glm::vec4 pos;
		glm::vec3 eyePos;	// eye space position used for shading
		glm::vec3 eyeNor;	// eye space normal used for shading, cuz normal will go wrong after perspective transformation
		glm::vec3 col;
		glm::vec2 texcoord0;
		TextureData* dev_diffuseTex = NULL;
    int texture_width;
    int texture_height;
	};

	struct Primitive {
		PrimitiveType primitiveType = Triangle;	// C++ 11 init
		VertexOut v[3];
	};

	struct Fragment {
		glm::vec3 color;
		glm::vec3 eyePos;	// eye space position used for shading
		glm::vec3 eyeNor;
		VertexAttributeTexcoord texcoord0;
		TextureData* dev_diffuseTex;
    int texture_width;
    int texture_height;
	};

	struct PrimitiveDevBufPointers {
		int primitiveMode;	//from tinygltfloader macro
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

		// Vertex Out, vertex used for rasterization, this is changing every frame
		VertexOut* dev_verticesOut;
	};

}

static std::map<std::string, std::vector<PrimitiveDevBufPointers>> mesh2PrimitivesMap;

static int width = 0;
static int height = 0;

static int totalNumPrimitives = 0;
static Primitive *dev_primitives = NULL;
static Fragment *dev_fragmentBuffer = NULL;
static glm::vec3 *dev_framebuffer = NULL;

static int * dev_depth = NULL;
static int * dev_mutex = NULL;

/**
 * Kernel that writes the image to the OpenGL PBO directly.
 */
__global__ 
void sendImageToPBO(uchar4 *pbo, int w, int h, glm::vec3 *image) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * w);

    if (x < w && y < h) {
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
*
* Fragment shader code
*/
__global__
void render(int w, int h, Fragment *fragmentBuffer, glm::vec3 *framebuffer) {
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  if (x >= w || y >= h) {
    return;
  }
  int index = x + (y * w);

  auto col = fragmentBuffer[index].color;
  Fragment f = fragmentBuffer[index];
  framebuffer[index] = fragmentBuffer[index].color;

  glm::vec3 dir_to_light(glm::normalize(glm::vec3(10.f, 10.f, 20.f) - f.eyePos));
  glm::vec3 nor = f.eyeNor;
  glm::vec3 dir_to_eye(glm::normalize(-f.eyePos));
  glm::vec3 state(dir_to_light + dir_to_eye);
  glm::vec3 texture_coloring(255.f);

  // display based on visualization type

  // shading for lines and points
#if LINES || POINTS
  framebuffer[index] = f.color;
  return;
#endif

  // shading triangles
  if (fragmentBuffer[index].dev_diffuseTex) {
    int width = f.texture_width;
    int height = f.texture_height;
    auto& tex = f.dev_diffuseTex;

#if BILINEAR
    float u = f.texcoord0.x * width;  int x = glm::floor(u);
    float v = f.texcoord0.y * height; int y = glm::floor(v);

    float uRatio = u - x;
    float vRatio = v - y;

    int uv_0 = 3 * (x + y * width);
    int uv_1 = 3 * (x + 1 + y * width);
    int uv_2 = 3 * (x + (y + 1) * width);
    int uv_3 = 3 * (x + 1 + (y + 1) * width);

    // bilinear interp: lerp in one direction twice, then lerp between those results
    glm::vec3 col(0.f);
    for (int i = 0; i < 3; ++i) {
      float val_1 = glm::mix(tex[uv_0 + i], tex[uv_1 + i], uRatio);
      float val_2 = glm::mix(tex[uv_2 + i], tex[uv_3 + i], uRatio);
      col[i] = glm::mix(val_1, val_2, vRatio);
    }

    framebuffer[index] = col / 255.f;
#else 
    int u = f.texcoord0.x * width;
    int v = f.texcoord0.y * height;

    int uvIndex = 3 * (u + (v * width));
    framebuffer[index] = glm::vec3(tex[uvIndex] / 255.f, tex[uvIndex + 1] / 255.f, tex[uvIndex + 2] / 255.f);
#endif
  } else {
    // lambert
    float absDot = glm::abs(glm::dot(f.eyeNor, dir_to_light));
    glm::vec3 intensity(1.5f);
    framebuffer[index] = glm::clamp(f.color * intensity * absDot, 0.f, 1.f);
  }

#if NORMALS
  framebuffer[index] = nor;
#endif
}

/**
 * Called once at the beginning of the program to allocate memory.
 */
void rasterizeInit(int w, int h) {
    width = w;
    height = h;
	cudaFree(dev_fragmentBuffer);
	cudaMalloc(&dev_fragmentBuffer, width * height * sizeof(Fragment));
	cudaMemset(dev_fragmentBuffer, 0, width * height * sizeof(Fragment));
    cudaFree(dev_framebuffer);
    cudaMalloc(&dev_framebuffer,   width * height * sizeof(glm::vec3));
    cudaMemset(dev_framebuffer, 0, width * height * sizeof(glm::vec3));
    
	cudaFree(dev_depth);
	cudaMalloc(&dev_depth, width * height * sizeof(int));

  cudaFree(dev_mutex);
  cudaMalloc(&dev_mutex, width * height * sizeof(int));
  // zero out the mutex array so no random mem when doing call checks
  cudaMemset(dev_mutex, 0, width * height * sizeof(int)); 

	checkCUDAError("rasterizeInit");
}

__global__
void initDepth(int w, int h, int * depth)
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
void _deviceBufferCopy(int N, BufferByte* dev_dst, const BufferByte* dev_src, int n, int byteStride, int byteOffset, int componentTypeByteSize) {
	
	// Attribute (vec3 position)
	// component (3 * float)
	// byte (4 * byte)

	// id of component
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (i < N) {
		int count = i / n;
		int offset = i - count * n;	// which component of the attribute

		for (int j = 0; j < componentTypeByteSize; j++) {
			
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
	glm::mat4 MV, glm::mat3 MV_normal) {

	// vertex id
	int vid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (vid < numVertices) {
		position[vid] = glm::vec3(MV * glm::vec4(position[vid], 1.0f));
		normal[vid] = glm::normalize(MV_normal * normal[vid]);
	}
}

glm::mat4 getMatrixFromNodeMatrixVector(const tinygltf::Node & n) {
	
	glm::mat4 curMatrix(1.0);

	const std::vector<double> &m = n.matrix;
	if (m.size() > 0) {
		// matrix, copy it

		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				curMatrix[i][j] = (float)m.at(4 * i + j);
			}
		}
	} else {
		// no matrix, use rotation, scale, translation

		if (n.translation.size() > 0) {
			curMatrix[3][0] = n.translation[0];
			curMatrix[3][1] = n.translation[1];
			curMatrix[3][2] = n.translation[2];
		}

		if (n.rotation.size() > 0) {
			glm::mat4 R;
			glm::quat q;
			q[0] = n.rotation[0];
			q[1] = n.rotation[1];
			q[2] = n.rotation[2];

			R = glm::mat4_cast(q);
			curMatrix = curMatrix * R;
		}

		if (n.scale.size() > 0) {
			curMatrix = curMatrix * glm::scale(glm::vec3(n.scale[0], n.scale[1], n.scale[2]));
		}
	}

	return curMatrix;
}

void traverseNode (
	std::map<std::string, glm::mat4> & n2m,
	const tinygltf::Scene & scene,
	const std::string & nodeString,
	const glm::mat4 & parentMatrix
	) 
{
	const tinygltf::Node & n = scene.nodes.at(nodeString);
	glm::mat4 M = parentMatrix * getMatrixFromNodeMatrixVector(n);
	n2m.insert(std::pair<std::string, glm::mat4>(nodeString, M));

	auto it = n.children.begin();
	auto itEnd = n.children.end();

	for (; it != itEnd; ++it) {
		traverseNode(n2m, scene, *it, M);
	}
}

void rasterizeSetBuffers(const tinygltf::Scene & scene) {

	totalNumPrimitives = 0;

	std::map<std::string, BufferByte*> bufferViewDevPointers;

	// 1. copy all `bufferViews` to device memory
	{
		std::map<std::string, tinygltf::BufferView>::const_iterator it(
			scene.bufferViews.begin());
		std::map<std::string, tinygltf::BufferView>::const_iterator itEnd(
			scene.bufferViews.end());

		for (; it != itEnd; it++) {
			const std::string key = it->first;
			const tinygltf::BufferView &bufferView = it->second;
			if (bufferView.target == 0) {
				continue; // Unsupported bufferView.
			}

			const tinygltf::Buffer &buffer = scene.buffers.at(bufferView.buffer);

			BufferByte* dev_bufferView;
			cudaMalloc(&dev_bufferView, bufferView.byteLength);
			cudaMemcpy(dev_bufferView, &buffer.data.front() + bufferView.byteOffset, bufferView.byteLength, cudaMemcpyHostToDevice);

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
			for (; it != itEnd; ++it) {
				traverseNode(nodeString2Matrix, scene, *it, glm::mat4(1.0f));
			}
		}

		// parse through node to access mesh

		auto itNode = nodeString2Matrix.begin();
		auto itEndNode = nodeString2Matrix.end();
		for (; itNode != itEndNode; ++itNode) {

			const tinygltf::Node & N = scene.nodes.at(itNode->first);
			const glm::mat4 & matrix = itNode->second;
			const glm::mat3 & matrixNormal = glm::transpose(glm::inverse(glm::mat3(matrix)));

			auto itMeshName = N.meshes.begin();
			auto itEndMeshName = N.meshes.end();

			for (; itMeshName != itEndMeshName; ++itMeshName) {

				const tinygltf::Mesh & mesh = scene.meshes.at(*itMeshName);

				auto res = mesh2PrimitivesMap.insert(std::pair<std::string, std::vector<PrimitiveDevBufPointers>>(mesh.name, std::vector<PrimitiveDevBufPointers>()));
				std::vector<PrimitiveDevBufPointers> & primitiveVector = (res.first)->second;

				// for each primitive
				for (size_t i = 0; i < mesh.primitives.size(); i++) {
					const tinygltf::Primitive &primitive = mesh.primitives[i];

					if (primitive.indices.empty())
						return;

					// add new attributes here for your PrimitiveDevBufPointers when you add new attributes
					VertexIndex* dev_indices = NULL;
					VertexAttributePosition* dev_position = NULL;
					VertexAttributeNormal* dev_normal = NULL;
					VertexAttributeTexcoord* dev_texcoord0 = NULL;

					// ----------Indices-------------

					const tinygltf::Accessor &indexAccessor = scene.accessors.at(primitive.indices);
					const tinygltf::BufferView &bufferView = scene.bufferViews.at(indexAccessor.bufferView);
					BufferByte* dev_bufferView = bufferViewDevPointers.at(indexAccessor.bufferView);

					// assume type is SCALAR for indices
					int n = 1;
					int numIndices = indexAccessor.count;
					int componentTypeByteSize = sizeof(VertexIndex);
					int byteLength = numIndices * n * componentTypeByteSize;

					dim3 numThreadsPerBlock(128);
					dim3 numBlocks((numIndices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
					cudaMalloc(&dev_indices, byteLength);
					_deviceBufferCopy << <numBlocks, numThreadsPerBlock >> > (
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
					switch (primitive.mode) {
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
					for (; it != itEnd; it++) {
						const tinygltf::Accessor &accessor = scene.accessors.at(it->second);
						const tinygltf::BufferView &bufferView = scene.bufferViews.at(accessor.bufferView);

						int n = 1;
						if (accessor.type == TINYGLTF_TYPE_SCALAR) {
							n = 1;
						}
						else if (accessor.type == TINYGLTF_TYPE_VEC2) {
							n = 2;
						}
						else if (accessor.type == TINYGLTF_TYPE_VEC3) {
							n = 3;
						}
						else if (accessor.type == TINYGLTF_TYPE_VEC4) {
							n = 4;
						}

						BufferByte * dev_bufferView = bufferViewDevPointers.at(accessor.bufferView);
						BufferByte ** dev_attribute = NULL;

						numVertices = accessor.count;
						int componentTypeByteSize;

						// Note: since the type of our attribute array (dev_position) is static (float32)
						// We assume the glTF model attribute type are 5126(FLOAT) here

						if (it->first.compare("POSITION") == 0) {
							componentTypeByteSize = sizeof(VertexAttributePosition) / n;
							dev_attribute = (BufferByte**)&dev_position;
						}
						else if (it->first.compare("NORMAL") == 0) {
							componentTypeByteSize = sizeof(VertexAttributeNormal) / n;
							dev_attribute = (BufferByte**)&dev_normal;
						}
						else if (it->first.compare("TEXCOORD_0") == 0) {
							componentTypeByteSize = sizeof(VertexAttributeTexcoord) / n;
							dev_attribute = (BufferByte**)&dev_texcoord0;
						}

						std::cout << accessor.bufferView << "  -  " << it->second << "  -  " << it->first << '\n';

						dim3 numThreadsPerBlock(128);
						dim3 numBlocks((n * numVertices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
						int byteLength = numVertices * n * componentTypeByteSize;
						cudaMalloc(dev_attribute, byteLength);

						_deviceBufferCopy << <numBlocks, numThreadsPerBlock >> > (
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
					if (!primitive.material.empty()) {
						const tinygltf::Material &mat = scene.materials.at(primitive.material);
						printf("material.name = %s\n", mat.name.c_str());

						if (mat.values.find("diffuse") != mat.values.end()) {
							std::string diffuseTexName = mat.values.at("diffuse").string_value;
							if (scene.textures.find(diffuseTexName) != scene.textures.end()) {
								const tinygltf::Texture &tex = scene.textures.at(diffuseTexName);
								if (scene.images.find(tex.source) != scene.images.end()) {
									const tinygltf::Image &image = scene.images.at(tex.source);

									size_t s = image.image.size() * sizeof(TextureData);
									cudaMalloc(&dev_diffuseTex, s);
									cudaMemcpy(dev_diffuseTex, &image.image.at(0), s, cudaMemcpyHostToDevice);
									
									diffuseTexWidth = image.width;
									diffuseTexHeight = image.height;

									checkCUDAError("Set Texture Image data");
								}
							}
						}

						// TODO: write your code for other materials
						// You may have to take a look at tinygltfloader
						// You can also use the above code loading diffuse material as a start point 
					}

					// ---------Node hierarchy transform--------
					cudaDeviceSynchronize();
					
					dim3 numBlocksNodeTransform((numVertices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
					_nodeMatrixTransform << <numBlocksNodeTransform, numThreadsPerBlock >> > (
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

						dev_vertexOut	//VertexOut
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

		for (; it != itEnd; it++) {
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
	int width, int height) {

	// vertex id
	int v_id = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (v_id < numVertices) {

    glm::vec4 v_world_space = glm::vec4(primitive.dev_position[v_id], 1.f);

		// Apply vertex transformation here
		// Transform to clipping space
    glm::vec4 v_pos = MVP * v_world_space;
    // Transform into NDC space
    v_pos /= v_pos.w;
    // Transform to screen space
    v_pos.x = (v_pos.x + 1) * width / 2.f;
    v_pos.y = (1 - v_pos.y) * height / 2.f;
    v_pos.z = (1 + v_pos.z) / 2.f;

		// Apply vertex assembly here
		// Assemble all attribute arrays into the primitive array
    primitive.dev_verticesOut[v_id].pos = v_pos;
    primitive.dev_verticesOut[v_id].eyeNor = glm::normalize(MV_normal * primitive.dev_normal[v_id]);
    primitive.dev_verticesOut[v_id].eyePos = glm::vec3(MV * v_world_space);
    primitive.dev_verticesOut[v_id].dev_diffuseTex = primitive.dev_diffuseTex;
    primitive.dev_verticesOut[v_id].texture_width = primitive.diffuseTexWidth;
    primitive.dev_verticesOut[v_id].texture_height = primitive.diffuseTexHeight;
    if (primitive.dev_texcoord0 != NULL) {
      primitive.dev_verticesOut[v_id].texcoord0 = primitive.dev_texcoord0[v_id];
    }
	}
}

static int curPrimitiveBeginId = 0;

__global__ 
void _primitiveAssembly(int numIndices, int curPrimitiveBeginId, Primitive* dev_primitives, PrimitiveDevBufPointers primitive) {

	int index_id = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index_id >= numIndices) {
    return;
  }

  // id for current primitives vector
	int primitive_id = 0;	

	if (primitive.primitiveMode == TINYGLTF_MODE_TRIANGLES) {
    primitive_id = index_id / (int)primitive.primitiveType;
		dev_primitives[primitive_id + curPrimitiveBeginId].v[index_id % (int)primitive.primitiveType]
			= primitive.dev_verticesOut[primitive.dev_indices[index_id]];
	} else if (primitive.primitiveMode == TINYGLTF_MODE_LINE) {
    primitive_id = index_id / (int)primitive.primitiveType;
    dev_primitives[primitive_id + curPrimitiveBeginId].primitiveType = Line;
    dev_primitives[primitive_id + curPrimitiveBeginId].v[index_id % (int)primitive.primitiveType]
      = primitive.dev_verticesOut[primitive.dev_indices[index_id]];
  } else if (primitive.primitiveMode == TINYGLTF_MODE_POINTS) {
    primitive_id = index_id / (int)primitive.primitiveType;
    dev_primitives[primitive_id + curPrimitiveBeginId].primitiveType = Point;
    dev_primitives[primitive_id + curPrimitiveBeginId].v[index_id % (int)primitive.primitiveType]
      = primitive.dev_verticesOut[primitive.dev_indices[index_id]];
  }
}

__global__
void computePointsRasterization(const int width, const int height, const int num_inputs,
  const Primitive *primitives,
  Fragment *fragments) {

  int input_index = blockDim.x * blockIdx.x + threadIdx.x;
  if (input_index >= num_inputs) {
    return;
  }

  Primitive primitive = primitives[input_index];
  glm::vec2 a(primitive.v[0].pos);
  glm::vec2 b(primitive.v[1].pos);
  glm::vec2 c(primitive.v[2].pos);

  glm::vec3 color(1.f);

  int index = 0;
  if (0 <= a.x && a.x < width && 0 <= a.y && a.y < height) {
    index = (int)a.x + (int)a.y * width;
    fragments[index].color = color;
  }
  if (0 <= b.x && b.x < width && 0 <= b.y && b.y < height) {
    index = (int)b.x + (int)b.y * width;
    fragments[index].color = color;
  }
  if (0 <= c.x && c.x < width && 0 <= c.y && c.y < height) {
    index = (int)c.x + (int)c.y * width;
    fragments[index].color = color;
  }
}

__host__ __device__
void lineCalculations(const int width, const int height,
  const glm::vec2& a, const glm::vec2& b,
  Fragment *fragments) {

  glm::vec2 min(
    glm::max(0.f, glm::min(a.x, b.x)),
    glm::max(0.f, glm::min(a.y, b.y)));
  glm::vec2 max(
    glm::min((float)width, glm::max(a.x, b.x)),
    glm::min((float)height, glm::max(a.y, b.y)));
  min = glm::floor(min);
  max = glm::ceil(max);

  // invalid config
  if (min.x > max.x && min.y > max.y) {
    return;
  }

  // optimize runtime calcs, iterate over longer dir (fewer var creations)
  glm::vec2 diff(max - min);
  int start_to_end = (diff.x >= diff.y) ? diff.x : diff.y;

  glm::vec3 color(1.f);
  // fill values
  for (float i = 0; i <= start_to_end; ++i) {
    glm::vec2 p = glm::mix(a, b, i / start_to_end);
    fragments[(int)p.x + (int)p.y * width].color = color;
  }
}

__global__
void computeLinesRasterization(const int width, const int height, const int num_inputs,
  const Primitive *primitives,
  Fragment *fragments) {

  int input_index = blockDim.x * blockIdx.x + threadIdx.x;
  if (input_index >= num_inputs) {
    return;
  }

  Primitive primitive = primitives[input_index];
  glm::vec2 a(primitive.v[0].pos);
  glm::vec2 b(primitive.v[1].pos);
  glm::vec2 c(primitive.v[2].pos);

  lineCalculations(width, height, a, b, fragments);
  lineCalculations(width, height, b, c, fragments);
  lineCalculations(width, height, c, a, fragments);
}

__global__ void computeRasterization(const int width, const int height, const int num_triangles,
  const Primitive* primitives,
  Fragment* fragments,
  int* depth, int* mutex) {

  int triangle_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (triangle_index >= num_triangles) {
    return;
  }

  // grabbing vertex information
  Primitive triangle = primitives[triangle_index];
  VertexOut v0 = triangle.v[0]; glm::vec3 a(v0.pos);
  VertexOut v1 = triangle.v[1]; glm::vec3 b(v1.pos);
  VertexOut v2 = triangle.v[2]; glm::vec3 c(v2.pos);
  glm::vec3 tri[3] = { a, b, c };

#if BACKFACE_CULL
  // ignore triangles facing the other direction
  if (calculateSignedArea(tri) < 0.f) {
    return;
  }
#endif

  // find bounding box for this triangle - clamping between screenspace bounds [0, width], [0, height])
  glm::vec2 min_xy(
    glm::min(a.x, glm::min(b.x, glm::min(c.x, 1.f * width))),
    glm::min(a.y, glm::min(b.y, glm::min(c.y, 1.f * height))));
  glm::vec2 max_xy(
    glm::max(a.x, glm::max(b.x, glm::max(c.x, 0.f))),
    glm::max(a.y, glm::max(b.y, glm::max(c.y, 0.f))));

  // create bounds
  int x_min = glm::floor(min_xy.x); int x_max = glm::ceil(max_xy.x);
  int y_min = glm::floor(min_xy.y); int y_max = glm::ceil(max_xy.y);

  // check bounds
  if (x_min > x_max || y_min > y_max) {
    return;
  }

  // beg load tex data for looping so dont need to reload
  TextureData *pDiffuseTexData = v0.dev_diffuseTex;
  int diffuse_texture_width = 0;
  int diffuse_texture_height = 0;
  if (pDiffuseTexData) {
    diffuse_texture_width = v0.texture_width;
    diffuse_texture_height = v0.texture_height;
  }

  // doing scanline rendering based on y-value and x-location depths
  for (int y = y_min; y <= y_max; ++y) {
    for (int x = x_min; x <= x_max; ++x) {
      // 2D x,y to 1D indexing --> x + width * y;
      int fragment_idx = (int)(x + width * y);
      glm::vec3 p(x, y, 0);

      // setup barycentric weights for future calculations
      glm::vec3 b_weights = calculateBarycentricCoordinate(tri, glm::vec2(x, y));
      if (!isBarycentricCoordInBounds(b_weights)) {
        // not part of this triangle
        continue;
      }

      // calculate attributes based on barycentric values
      // fake [0, int_max] depth buffer range so that can compare int values - easier for use with atomicCAS
      int z_depth = (int)(INT_MAX * (b_weights.x * a.z + b_weights.y * b.z + b_weights.z * c.z));
      // update for perspective
      b_weights /= glm::vec3(a.z, b.z, c.z);
      // calculate remaining attributes with perspective divide
      float z_perspective = 1.f / (b_weights.x + b_weights.y + b_weights.z);
      glm::vec3 pos = z_perspective * (b_weights.x * v0.eyePos + b_weights.y * v1.eyePos + b_weights.z * v2.eyePos);
      glm::vec3 nor = z_perspective * (b_weights.x * v0.eyeNor + b_weights.y * v1.eyeNor + b_weights.z * v2.eyeNor);
      nor = glm::normalize(nor);
      glm::vec2 uvs = z_perspective * (b_weights.x * v0.texcoord0 + b_weights.y * v1.texcoord0 + b_weights.z * v2.texcoord0);

#if MUTEX
      // use mutex check to avoid race conditions for writing to fragment buffer
      // Waiting for fragment to unlock
      bool isSet;
      do {
        isSet = (atomicCAS(&mutex[fragment_idx], 0, 1) == 0);
        if (isSet) {
          // fragment available - do depth check
          if (z_depth < depth[fragment_idx]) {
            depth[fragment_idx] = z_depth;
            fragments[fragment_idx].color = nor;
            fragments[fragment_idx].eyeNor = nor;
            fragments[fragment_idx].eyePos = pos;
            fragments[fragment_idx].texcoord0 = uvs;
            if (pDiffuseTexData) {
              fragments[fragment_idx].dev_diffuseTex = pDiffuseTexData;
              fragments[fragment_idx].texture_width = diffuse_texture_width;
              fragments[fragment_idx].texture_height = diffuse_texture_height;
            }
          }
          mutex[fragment_idx] = 0;
        }
      } while (!isSet);

# else 
      if (z_depth < depth[fragment_idx]) {
        depth[fragment_idx] = z_depth;
        fragments[fragment_idx].color = nor;
        fragments[fragment_idx].eyeNor = nor;
        fragments[fragment_idx].eyePos = pos;
        fragments[fragment_idx].texcoord0 = uvs;
        if (pDiffuseTexData) {
          fragments[fragment_idx].dev_diffuseTex = pDiffuseTexData;
          fragments[fragment_idx].texture_width = diffuse_texture_width;
          fragments[fragment_idx].texture_height = diffuse_texture_height;
        }
      } else {
        // occluded
        continue;
      }
# endif

    } // end: from x_min to x_max
  } // end: from y_min to y_max
}

// For Runtime Comparison - taken and modified from my own CUDA Pathtracer code
void startTimer(std::chrono::high_resolution_clock::time_point& timer) {
  timer = std::chrono::high_resolution_clock::now();
}

// returns duration
float endTimer(std::chrono::high_resolution_clock::time_point& timer) {
  auto time_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> duro = time_end - timer;
  return (float)(duro.count());
}

/**
 * Perform rasterization.
 */
void rasterize(uchar4 *pbo, const glm::mat4 & MVP, const glm::mat4 & MV, const glm::mat3 MV_normal) {
    int sideLength2d = 8;
    dim3 blockSize2d(sideLength2d, sideLength2d);
    dim3 blockCount2d((width  - 1) / blockSize2d.x + 1,
		(height - 1) / blockSize2d.y + 1);

	// Execute the rasterization pipeline
	// (See README for rasterization pipeline outline.)

  dim3 numThreadsPerBlock(128);

  // create timers for runtime checking
  std::chrono::high_resolution_clock::time_point vert_time;
  std::chrono::high_resolution_clock::time_point primitive_assembly_time;
  std::chrono::high_resolution_clock::time_point rasterization_time;
  std::chrono::high_resolution_clock::time_point render_time;
  float vert_duration = 0.f;
  float primitive_duration = 0.f;
  float rasterization_duration = 0.f;
  float render_duration = 0.f;

	// Vertex Process & primitive assembly
	{
    curPrimitiveBeginId = 0;

		auto it = mesh2PrimitivesMap.begin();
		auto itEnd = mesh2PrimitivesMap.end();

		for (; it != itEnd; ++it) {
			auto p = (it->second).begin();	// each primitive
			auto pEnd = (it->second).end();
			for (; p != pEnd; ++p) {
				dim3 numBlocksForVertices((p->numVertices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
				dim3 numBlocksForIndices((p->numIndices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);

        startTimer(vert_time);
				_vertexTransformAndAssembly << < numBlocksForVertices, numThreadsPerBlock >> >(p->numVertices, *p, MVP, MV, MV_normal, width, height);
        vert_duration = endTimer(vert_time);
        checkCUDAError("Vertex Processing");
				
        cudaDeviceSynchronize();
				
        startTimer(primitive_assembly_time);
        _primitiveAssembly << < numBlocksForIndices, numThreadsPerBlock >> >
					(p->numIndices, 
					curPrimitiveBeginId, 
					dev_primitives, 
					*p);
        primitive_duration = endTimer(primitive_assembly_time);
				checkCUDAError("Primitive Assembly");

				curPrimitiveBeginId += p->numPrimitives;
			}
		}

		checkCUDAError("Vertex Processing and Primitive Assembly");
	}
	
	cudaMemset(dev_fragmentBuffer, 0, width * height * sizeof(Fragment));
	initDepth << <blockCount2d, blockSize2d >> >(width, height, dev_depth);
	
	// rasterize
  startTimer(rasterization_time);
  int numBlocks = (totalNumPrimitives + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x;
#if POINTS
  computePointsRasterization << <numBlocks, numThreadsPerBlock >> > (
    width, height, totalNumPrimitives, dev_primitives, dev_fragmentBuffer);
#elif LINES
  computeLinesRasterization << <numBlocks, numThreadsPerBlock >> > (
    width, height, totalNumPrimitives, dev_primitives, dev_fragmentBuffer);
#else // TRIANGLES
  computeRasterization << <numBlocks, numThreadsPerBlock >> > (
    width, height, totalNumPrimitives, dev_primitives, dev_fragmentBuffer,
    dev_depth, dev_mutex);
#endif
  rasterization_duration = endTimer(rasterization_time);

  // Copy depthbuffer colors into framebuffer
  startTimer(rasterization_time);
	render << <blockCount2d, blockSize2d >> >(width, height, dev_fragmentBuffer, dev_framebuffer);
  rasterization_duration = endTimer(rasterization_time);
	checkCUDAError("fragment shader");
  // Copy framebuffer into OpenGL buffer for OpenGL previewing
  sendImageToPBO<<<blockCount2d, blockSize2d>>>(pbo, width, height, dev_framebuffer);
  checkCUDAError("copy render result to pbo");

#if TIME
  printf("Time in milliseconds for vert transform: %f\n", vert_duration);
  printf("Time in milliseconds for primitive assembly: %f\n", primitive_duration);
  printf("Time in milliseconds for rasterization: %f\n", rasterization_duration);
  printf("Time in milliseconds for rendering: %f\n", render_duration);
#endif
}

/**
 * Called once at the end of the program to free CUDA memory.
 */
void rasterizeFree() {

    // deconstruct primitives attribute/indices device buffer

	auto it(mesh2PrimitivesMap.begin());
	auto itEnd(mesh2PrimitivesMap.end());
	for (; it != itEnd; ++it) {
		for (auto p = it->second.begin(); p != it->second.end(); ++p) {
			cudaFree(p->dev_indices);
			cudaFree(p->dev_position);
			cudaFree(p->dev_normal);
			cudaFree(p->dev_texcoord0);
			cudaFree(p->dev_diffuseTex);
			cudaFree(p->dev_verticesOut);
		}
	}

  cudaFree(dev_primitives);
  dev_primitives = NULL;

	cudaFree(dev_fragmentBuffer);
	dev_fragmentBuffer = NULL;

  cudaFree(dev_framebuffer);
  dev_framebuffer = NULL;

	cudaFree(dev_depth);
	dev_depth = NULL;

  cudaFree(dev_mutex);
  dev_mutex = NULL;

  checkCUDAError("rasterize Free");
}
