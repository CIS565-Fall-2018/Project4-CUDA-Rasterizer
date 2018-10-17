/**
 * @file      rasterize.cu
 * @brief     CUDA-accelerated rasterization pipeline.
 * @authors   Skeleton code: Yining Karl Li, Kai Ninomiya, Shuai Shao (Shrek)
 * @date      2012-2018
 * @copyright University of Pennsylvania & Edward Atter
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

#include <math.h>

#define MODE_TRIANGLE	0
#define MODE_POINT		1
#define MODE_LINE		2

#define OPTION_ENABLE_LAMBERT			0
#define OPTION_ENABLE_SSAA				0
#define OPTION_SSAA_GRID_SIZE 			2
#define OPTION_ENABLE_BACK_FACE_CULLING	0
#define OPTION_SELECT_MODE				MODE_TRIANGLE

#define SSAA_GRID_AREA				OPTION_SSAA_GRID_SIZE * OPTION_SSAA_GRID_SIZE

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

	// TODO: add new attributes to your VertexOut
	// The attributes listed below might be useful,
	// but always feel free to modify on your own

	glm::vec3 eyePos;	// eye space position used for shading
	glm::vec3 eyeNor;	// eye space normal used for shading, cuz normal will go wrong after perspective transformation
	// glm::vec3 col;
	glm::vec2 texcoord0;
	TextureData* dev_diffuseTex = NULL;
	// int texWidth, texHeight;
	// ...
};

struct Primitive {
	PrimitiveType primitiveType = Triangle;	// C++ 11 init
	VertexOut v[3];
};

struct Fragment {
	glm::vec3 color;

	// TODO: add new attributes to your Fragment
	// The attributes listed below might be useful,
	// but always feel free to modify on your own

	glm::vec3 eyePos;	// eye space position used for shading
	glm::vec3 eyeNor;
	// VertexAttributeTexcoord texcoord0;
	// TextureData* dev_diffuseTex;
	// ...
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
	// TextureData* dev_specularTex;
	// TextureData* dev_normalTex;
	// ...

	// Vertex Out, vertex used for rasterization, this is changing every frame
	VertexOut* dev_verticesOut;

	// TODO: add more attributes when needed
};

}

static std::map<std::string, std::vector<PrimitiveDevBufPointers>> mesh2PrimitivesMap;


static int width = 0;
static int height = 0;
static int trueWidth = 0;
static int trueHeight = 0;

static int totalNumPrimitives = 0;
static Primitive *dev_primitives = NULL;
static Fragment *dev_fragmentBuffer = NULL;
static glm::vec3 *dev_framebuffer = NULL;

static int * dev_depth = NULL;	// you might need this buffer when doing depth test

//Additional globals
static int * dev_mutex = NULL; //int []

// Generates a random float between A and B
// From https://stackoverflow.com/a/24537113/3421536
// See also: https://stackoverflow.com/a/25034092/3421536
__device__
int generateRandomInt(int A, int B, float randu_f) {
	//float randu_f = curand_uniform(state);
	randu_f *= (B-A+0.999999); // You should not use (B-A+1)*
	randu_f += A;
	int randu_int = __float2int_rz(randu_f);
	//printf("RAND: %i <--%f \n", randu_int, randu_f);
	if (randu_int > B || randu_int < A) {
		printf("WARN: generateRandomInt out of bounds! %i -> [%i, %i]\n", randu_int, A, B);
	}
	return randu_int;
}

/**
 * From https://github.com/CIS565-Fall-2018/Project3-CUDA-Path-Tracer
 * Handy-dandy hash function that provides seeds for random number generation.
 */
__host__ __device__ inline unsigned int utilhash(unsigned int a) {
	a = (a + 0x7ed55d16) + (a << 12);
	a = (a ^ 0xc761c23c) ^ (a >> 19);
	a = (a + 0x165667b1) + (a << 5);
	a = (a + 0xd3a2646c) ^ (a << 9);
	a = (a + 0xfd7046c5) + (a << 3);
	a = (a ^ 0xb55a4f09) ^ (a >> 16);
	return a;
}

// From https://github.com/CIS565-Fall-2018/Project3-CUDA-Path-Tracer
__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth) {
	int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
	return thrust::default_random_engine(h);
}

/**
 * Kernel that writes the image to the OpenGL PBO directly.
 */
__global__ 
void sendImageToPBO(uchar4 *pbo, int w, int h, int trueWidth, int trueHeight, glm::vec3 *image) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * trueWidth);

	if (x >= trueWidth || y >= trueHeight) { return; }

	glm::vec3 color;
	color.x = 0;
	color.y = 0;
	color.z = 0;
#if OPTION_ENABLE_SSAA
	//Random SSAA
	thrust::default_random_engine rng =
			makeSeededRandomEngine(0, index, 1);
	thrust::uniform_real_distribution<float> u01(0, 1);

	//Possible samples
	glm::vec3 samples[SSAA_GRID_AREA];
	int i = 0;
	for (int xOffset = 0; xOffset < OPTION_SSAA_GRID_SIZE; xOffset ++) {
		for (int yOffset = 0; yOffset < OPTION_SSAA_GRID_SIZE; yOffset ++) {
			int xIdx = xOffset + x * OPTION_SSAA_GRID_SIZE;
			int yIdx = (yOffset + y * OPTION_SSAA_GRID_SIZE) * w;
			int imageColorIdx = xIdx + yIdx;
			samples[i] = image[imageColorIdx];
			i++;
		}
	}

	// Generate random samples and add randomly selected pixels
	for (int i = 0; i < SSAA_GRID_AREA; i++){
		int randIdx = generateRandomInt(0, SSAA_GRID_AREA - 1, u01(rng));
		color.x += glm::clamp(samples[randIdx].x, 0.0f, 1.0f) * 255.0;
		color.y += glm::clamp(samples[randIdx].y, 0.0f, 1.0f) * 255.0;
		color.z += glm::clamp(samples[randIdx].z, 0.0f, 1.0f) * 255.0;
	}

	//Take the average
	color /= (float)SSAA_GRID_AREA;

#else
	color.x = glm::clamp(image[index].x, 0.0f, 1.0f) * 255.0;
	color.y = glm::clamp(image[index].y, 0.0f, 1.0f) * 255.0;
	color.z = glm::clamp(image[index].z, 0.0f, 1.0f) * 255.0;
#endif
	// Each thread writes one pixel location in the texture (textel)
	//if(color.x != 0 || color.y != 0 || color.z != 0) printf("COLOR: %f, %f, %f\n", color.x, color.y, color.z);
	pbo[index].w = 0;
	pbo[index].x = color.x;
	pbo[index].y = color.y;
	pbo[index].z = color.z;
}

/** 
 * Writes fragment colors to the framebuffer
 */
__global__
void render(int w, int h, Fragment *fragmentBuffer, glm::vec3 *frameBuffer) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * w);

	if (x < w && y < h) {
		frameBuffer[index] = fragmentBuffer[index].color;
#if OPTION_SELECT_MODE == MODE_TRIANGLE
		// TODO: add your fragment shader code here
#if OPTION_ENABLE_LAMBERT
		// Adapted from https://www.opengl.org/sdk/docs/tutorials/ClockworkCoders/lighting.php
		glm::vec3 v = fragmentBuffer[index].eyePos;
		glm::vec3 N = fragmentBuffer[index].eyeNor;
		glm::vec3 lightSource(1, 1, 1);
		glm::vec3 L = glm::normalize(lightSource - v);
		float Idiff = glm::dot(L, N);
		Idiff = glm::clamp(Idiff, 0.0f, 1.0f);
		frameBuffer[index] = Idiff * fragmentBuffer[index].color;
#endif
//End MODE_TRIANGLE
#endif
	}
}

/**
 * Called once at the beginning of the program to allocate memory.
 */
void rasterizeInit(int w, int h) {
	width = w;
	height = h;
	trueWidth = width;
	trueHeight = height;

#if OPTION_ENABLE_SSAA
	width *= OPTION_SSAA_GRID_SIZE;
	height *= OPTION_SSAA_GRID_SIZE;
#endif

	cudaFree(dev_fragmentBuffer);
	cudaMalloc(&dev_fragmentBuffer, width * height * sizeof(Fragment));
	cudaMemset(dev_fragmentBuffer, 0, width * height * sizeof(Fragment));
	cudaFree(dev_framebuffer);
	cudaMalloc(&dev_framebuffer,   width * height * sizeof(glm::vec3));
	cudaMemset(dev_framebuffer, 0, width * height * sizeof(glm::vec3));

	cudaFree(dev_depth);
	cudaMalloc(&dev_depth, width * height * sizeof(int));

	// Additional vars
	// Why free before malloc?
	cudaFree(dev_mutex);
	const int devMutexSize = sizeof(int) * width * height;
	cudaMalloc(&dev_mutex, devMutexSize);
	// Initialize empty, since cuda does not have calloc
	cudaMemset(dev_mutex, 0, devMutexSize);

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

					// TODO: add new attributes for your PrimitiveDevBufPointers when you add new attributes
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

						// TODO: write your code for other materails
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
	int vid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (vid < numVertices) {

		// TODO: Apply vertex transformation here
		// Multiply the MVP matrix for each vertex position, this will transform everything into clipping space
		// Then divide the pos by its w element to transform into NDC space
		// Finally transform x and y to viewport space
		glm::vec4 vPosition = glm::vec4(primitive.dev_position[vid], 1.0f);
		glm::vec3 vNormal = primitive.dev_normal[vid];
		// Order of multiplication is important here!
		glm::vec4 clipPosition = MVP * vPosition;

		clipPosition = clipPosition / clipPosition.w;

		clipPosition.x = ( width * (clipPosition.x / clipPosition.w + 1.0f) / 2);
		clipPosition.y = ( height * (1 - (clipPosition.y / clipPosition.w)) / 2);

		glm::vec3 eyePos = glm::vec3(vPosition * MV);
		glm::vec3 eyeNor = glm::normalize(vNormal * MV_normal);

		// TODO: Apply vertex assembly here
		// Assemble all attribute arraies into the primitive array
		primitive.dev_verticesOut[vid].eyePos = eyePos;
		primitive.dev_verticesOut[vid].eyeNor = eyeNor;
		primitive.dev_verticesOut[vid].pos = clipPosition;

	}
}



static int curPrimitiveBeginId = 0;

__global__ 
void _primitiveAssembly(int numIndices, int curPrimitiveBeginId, Primitive* dev_primitives, PrimitiveDevBufPointers primitive) {

	// index id
	int iid = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (iid < numIndices) {

		// This is primitive assembly for triangles
		int pid;	// id for cur primitives vector
		if (primitive.primitiveMode == TINYGLTF_MODE_TRIANGLES) {
			pid = iid / (int)primitive.primitiveType;
			dev_primitives[pid + curPrimitiveBeginId].v[iid % (int)primitive.primitiveType]
			                                            = primitive.dev_verticesOut[primitive.dev_indices[iid]];
		}
	}
}

__device__
void xyz(glm::vec4 * v4, glm::vec3 * v3) {
	v3->x = v4->x;
	v3->y = v4->y;
	v3->z = v4->z;
}

__global__
void kernRasterizePrimitive (
		int N,
		Primitive * dev_primitives, Fragment * dev_fragmentBuffer,
		int * dev_depth, int * dev_mutex, int width, int height) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= N) { return; }

#if OPTION_ENABLE_BACK_FACE_CULLING
	//https://en.wikipedia.org/wiki/Back-face_culling
	//Convert from vec4 for vec3
	glm::vec3 p0(0);
	glm::vec3 p1(0);
	glm::vec3 p2(0);
	xyz(&dev_primitives[idx].v[0].pos, &p0);
	xyz(&dev_primitives[idx].v[1].pos, &p1);
	xyz(&dev_primitives[idx].v[2].pos, &p2);

	//Back culling calculation
	glm::vec3 _N = glm::cross(p1 - p0, p2 - p0);
	float backface_check = glm::dot(p0 - dev_primitives[idx].v[0].eyePos, _N);
	if (backface_check > 0) {
		return;
	}
#endif

	//Triangle, defined by three points, used to get AABB
	glm::vec3 points[3];
	points[0] = glm::vec3(dev_primitives[idx].v[0].pos);
	points[1] = glm::vec3(dev_primitives[idx].v[1].pos);
	points[2] = glm::vec3(dev_primitives[idx].v[2].pos);
	AABB aabb = getAABBForTriangle(points);

	//Get bounds, max of screen, img
	// x = width; y = height;
	int widthStart = max(0, (int) aabb.min.x);
	int widthEnd = min(width, (int) aabb.max.x);
	int heightStart = max(0, (int) aabb.min.y);
	int heightEnd = min(height, (int) aabb.max.y);

#if OPTION_SELECT_MODE == MODE_TRIANGLE
	// Process each visible pixel
	for (int h = heightStart; h <= heightEnd; h++) {
		for (int w = widthStart; w <= widthEnd; w++) {
			int fragmentIdx = width * h + w;
			glm::vec3 barycentricCoordinate =
					calculateBarycentricCoordinate(points, glm::vec2(w, h));
			if (isBarycentricCoordInBounds(barycentricCoordinate)) {
				// Wait for mutex lock
				int isSet;
				do {
					isSet = (atomicCAS(&dev_mutex[idx], 0, 1));
					if(!isSet) { continue; }

					float depth = getZAtCoordinate(barycentricCoordinate, points) * INT_MAX * -1;

					if (depth < dev_depth[fragmentIdx]) {
						//Update the fragment
						dev_depth[fragmentIdx] = depth;
						dev_fragmentBuffer[fragmentIdx].eyePos =
								dev_primitives[idx].v[0].eyePos * barycentricCoordinate[0] +
								dev_primitives[idx].v[1].eyePos * barycentricCoordinate[1] +
								dev_primitives[idx].v[2].eyePos * barycentricCoordinate[2];
						dev_fragmentBuffer[fragmentIdx].eyeNor =
								dev_primitives[idx].v[0].eyeNor * barycentricCoordinate[0] +
								dev_primitives[idx].v[1].eyeNor * barycentricCoordinate[1] +
								dev_primitives[idx].v[2].eyeNor * barycentricCoordinate[2];
						dev_fragmentBuffer[fragmentIdx].color = dev_fragmentBuffer[fragmentIdx].eyeNor;
					}
				} while(!isSet);
				dev_mutex[idx] = 0;
			}
		}
	}
#elif OPTION_SELECT_MODE == MODE_POINT
	//Iterate thru triangle, generating point at each vertex
	for (int i = 0; i < 3; i++) {
		int x = dev_primitives[idx].v[i].pos.x;
		int y = dev_primitives[idx].v[i].pos.y;
		int pointIdx = y * width + x;

		//Set to static (white) color
		dev_fragmentBuffer[pointIdx].color = glm::vec3(1, 1, 1);
	}
#elif OPTION_SELECT_MODE == MODE_LINE
	for (int i = 0; i < 3; i++) {
		int j = (i + 1) % 2;
		//Use vecs not ints so we can calculate diffY
		glm::vec4 origin = dev_primitives[idx].v[i].pos;
		glm::vec4 dest = dev_primitives[idx].v[j].pos;
		//Flip to ensure origin < dest
		if (dest.x < origin.x) {
			glm::vec4 tmp = origin;
			origin = dest;
			dest = tmp;
		}

		//Calculate travel distance
		int diffX = dest.x - origin.x;
		//Prevent divide by 0
		if (diffX == 0) {
			diffX = 1;
		}
		int diffY = dest.y - origin.y;

		int last = origin.y;
		for (int x = origin.x; x <= dest.x; x++) {
			int y = diffY * (x - origin.x) / diffX + origin.y;

			//Same flipping as before, this time sorted by y
			int originY = y;
			int destY = last;
			if (destY < originY) {
				int tmpy = originY;
				originY = destY;
				destY = tmpy;
			}

			for (int y2 = originY; y2 <= destY; y2++) {
				// Prevent memory access exception, OOB
				if (x > widthEnd || x < widthStart || y2 > heightEnd || y2 < heightStart) {
					continue;
				}

				int pointIdx = y2 * width + x;
				//Set to static (white) color
				dev_fragmentBuffer[pointIdx].color = glm::vec3(1, 1, 1);
			}
			last = y;
		}
	}
#else
	printf("ERROR: Invalid mode selected\n");
#endif
}


/**
 * Perform rasterization.
 */
void rasterize(uchar4 *pbo, const glm::mat4 & MVP, const glm::mat4 & MV, const glm::mat3 MV_normal) {
	int sideLength2d = 8;
	dim3 blockSize2d(sideLength2d, sideLength2d);
	dim3 blockCount2d((width  - 1) / blockSize2d.x + 1,
			(height - 1) / blockSize2d.y + 1);
	dim3 trueBlockCount2d((trueWidth  - 1) / blockSize2d.x + 1,
			(trueHeight - 1) / blockSize2d.y + 1);

	// Execute your rasterization pipeline here
	// (See README for rasterization pipeline outline.)

	// Vertex Process & primitive assembly
	{
		curPrimitiveBeginId = 0;
		dim3 numThreadsPerBlock(128);

		auto it = mesh2PrimitivesMap.begin();
		auto itEnd = mesh2PrimitivesMap.end();

		for (; it != itEnd; ++it) {
			auto p = (it->second).begin();	// each primitive
			auto pEnd = (it->second).end();
			for (; p != pEnd; ++p) {
				dim3 numBlocksForVertices((p->numVertices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
				dim3 numBlocksForIndices((p->numIndices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);

				_vertexTransformAndAssembly << < numBlocksForVertices, numThreadsPerBlock >> >(p->numVertices, *p, MVP, MV, MV_normal, width, height);
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

	// rasterize
	// Copied from code above, why 128?
	dim3 numThreadsPerBlock(128, 1, 1);
	int primitiveBlockCount = (numThreadsPerBlock.x + totalNumPrimitives - 1) / numThreadsPerBlock.x;
	// Launch primitive kernel
	kernRasterizePrimitive <<< primitiveBlockCount, numThreadsPerBlock >>>(
			totalNumPrimitives,
			dev_primitives, dev_fragmentBuffer, dev_depth, dev_mutex,
			width, height);

	// Copy depthbuffer colors into framebuffer
	render << <blockCount2d, blockSize2d >> >(width, height, dev_fragmentBuffer, dev_framebuffer);
	checkCUDAError("fragment shader");
	// Copy framebuffer into OpenGL buffer for OpenGL previewing
	//printf("TW, TH = %i, %i || %i, %i", trueWidth, trueHeight, width, height);
	sendImageToPBO<<<trueBlockCount2d, blockSize2d>>>(pbo, width, height, trueWidth, trueHeight, dev_framebuffer);
	checkCUDAError("copy render result to pbo");
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

	cudaFree(dev_mutex);
	dev_mutex = NULL;

	checkCUDAError("rasterize Free");
}
