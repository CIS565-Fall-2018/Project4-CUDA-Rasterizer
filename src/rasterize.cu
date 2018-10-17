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

#define DISPLAY_TRIANGLE true
#define DISPLAY_LINE false
#define DISPLAY_POINT false

#define CULL_FACE true
#define DISPLAY_NORMAL false
#define MUTEX_ON false

namespace {

	typedef unsigned short VertexIndex;
	typedef glm::vec3 VertexAttributePosition;
	typedef glm::vec3 VertexAttributeNormal;
	typedef glm::vec2 VertexAttributeTexcoord;
	typedef unsigned char TextureData;

	typedef unsigned char BufferByte;

	enum PrimitiveType {
		Point = 1,
		Line = 2,
		Triangle = 3
	};

	struct VertexOut {
		glm::vec4 pos;

		// TODO: add new attributes to your VertexOut
		// The attributes listed below might be useful, 
		// but always feel free to modify on your own

		glm::vec3 eyePos;
		glm::vec3 eyeNor;

		glm::vec2 texcoord0;
		TextureData* dev_diffuseTex = NULL;
		int texWidth;
		int texHeight;
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

		glm::vec3 eyePos;
		glm::vec3 eyeNor;
		VertexAttributeTexcoord texcoord0;
		TextureData* dev_diffuseTex;
		int texWidth;
		int texHeight;
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

static int totalNumPrimitives = 0;
static Primitive *dev_primitives = NULL;
static Fragment *dev_fragmentBuffer = NULL;
static glm::vec3 *dev_framebuffer = NULL;
static unsigned int *dev_mutex = NULL;
static float *dev_depth = NULL;

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
*/

static __host__ __device__ glm::vec3 lerp(const glm::vec3 &x, const glm::vec3 &y, const float &alpha)
{
	return (1.0f - alpha) * x + alpha * y;
}

__global__
void render(int w, int h, Fragment *fragmentBuffer, glm::vec3 *framebuffer) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * w);

	if (x < w && y < h) {
		framebuffer[index] = fragmentBuffer[index].color;

		// TODO: add your fragment shader code here
		int index = x + (y * w);
		glm::vec3 &buffer = framebuffer[index];
		const Fragment &frag = fragmentBuffer[index];
		const glm::vec3 lightDir(50.f, 50.f, 100.f);

		// clear frame buffer
		buffer = glm::vec3(0.f);

		const glm::vec3 L = glm::normalize(lightDir - frag.eyePos);
		const glm::vec3 &N = frag.eyeNor;
		const glm::vec3 V = glm::normalize(-frag.eyePos);
		const glm::vec3 H = glm::normalize(V + L);

		glm::vec3 texelFetched = glm::vec3(255.0f, 255.0f, 255.0f);

		if (frag.dev_diffuseTex != NULL)
		{
			int texW = frag.texWidth;
			int texH = frag.texHeight;
			
			float x = (texW - 1.0f) * fragmentBuffer[index].texcoord0[0];
			float y = (texH - 1.0f) * fragmentBuffer[index].texcoord0[1];

			int xLower = std::floor(x);
			int yLower = std::floor(y);
			int xUpper = (xLower + 1) % texW;
			int yUpper = (yLower + 1) % texH;
			int idxLowerLeft = xLower + yLower * texW;
			int idxLowerRight = xUpper + yLower * texW;
			int idxUpperLeft = xLower + yUpper * texW;
			int idxUpperRight = xUpper + yUpper * texW;

			glm::vec3 lowerLeftTexel = glm::vec3(frag.dev_diffuseTex[idxLowerLeft * 3],
				frag.dev_diffuseTex[idxLowerLeft * 3 + 1], frag.dev_diffuseTex[idxLowerLeft * 3 + 2]);
			glm::vec3 lowerRightTexel = glm::vec3(frag.dev_diffuseTex[idxLowerRight * 3],
				frag.dev_diffuseTex[idxLowerRight * 3 + 1], frag.dev_diffuseTex[idxLowerRight * 3 + 2]);
			glm::vec3 upperLeftTexel = glm::vec3(frag.dev_diffuseTex[idxUpperLeft * 3],
				frag.dev_diffuseTex[idxUpperLeft * 3 + 1], frag.dev_diffuseTex[idxUpperLeft * 3 + 2]);
			glm::vec3 upperRightTexel = glm::vec3(frag.dev_diffuseTex[idxUpperRight * 3],
				frag.dev_diffuseTex[idxUpperRight * 3 + 1], frag.dev_diffuseTex[idxUpperRight * 3 + 2]);
			//Coefs for bilinear filtering
			float alphaX = x - static_cast<float>(xLower);
			float alphaY = y - static_cast<float>(yLower);

			glm::vec3 lowerBelt = lerp(lowerLeftTexel, lowerRightTexel, alphaX);
			glm::vec3 upperBelt = lerp(upperLeftTexel, upperRightTexel, alphaX);
			
			texelFetched = lerp(lowerBelt, upperBelt, alphaY);
			
		}

		const float factor = 1.0f / 255.0f;
		glm::vec3 colorNormalized = factor * texelFetched;
		if (DISPLAY_TRIANGLE)
		{
			buffer = glm::max(0.f, glm::dot(L, N)) * colorNormalized;
		}
		if (DISPLAY_NORMAL)
		{
			glm::vec3 norm = frag.eyeNor;
			buffer = glm::normalize(norm);
		}
		if (DISPLAY_POINT || DISPLAY_LINE)
		{
			buffer = frag.color;
		}
		
		buffer += glm::vec3(pow(glm::max(0.f, glm::dot(N, H)), 128.0f));
		//buffer = glm::normalize(buffer);
	}
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
	cudaMalloc(&dev_framebuffer, width * height * sizeof(glm::vec3));
	cudaMemset(dev_framebuffer, 0, width * height * sizeof(glm::vec3));

	cudaFree(dev_depth);
	cudaMalloc(&dev_depth, width * height * sizeof(float));
	//====================================
	cudaMalloc(&dev_mutex, width * height * sizeof(unsigned int));
	cudaMemset(dev_mutex, 0, width * height * sizeof(unsigned int));
	checkCUDAError("rasterizeInit");
}

__global__
void initDepth(int w, int h, float * depth)
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
	}
	else {
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

void traverseNode(
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
	VertexOut &out = primitive.dev_verticesOut[vid];
	const glm::vec3 &pos = primitive.dev_position[vid];
	const glm::vec3 &nor = primitive.dev_normal[vid];
	if (vid < numVertices)
	{
		// TODO: Apply vertex transformation here
		// Multiply the MVP matrix for each vertex position, this will transform everything into clipping space
		// Then divide the pos by its w element to transform into NDC space
		// Finally transform x and y to viewport space
		out.pos = MVP * glm::vec4(pos, 1.f);
		out.pos /= out.pos.w;
		out.pos.x = (1.f - out.pos.x) * .5f * width;
		out.pos.y = (1.f - out.pos.y) * .5f * height;
		// TODO: Apply vertex assembly here
		// Assemble all attribute arraies into the primitive array
		out.eyePos = multiplyMV(MV, glm::vec4(pos, 1.f));
		out.eyeNor = glm::normalize(MV_normal * nor);
		if (primitive.dev_texcoord0 != NULL)
		{
			out.texcoord0 = primitive.dev_texcoord0[vid];
		}
		out.dev_diffuseTex = primitive.dev_diffuseTex;
		out.texWidth = primitive.diffuseTexWidth;
		out.texHeight = primitive.diffuseTexHeight;
	}
}



static int curPrimitiveBeginId = 0;

__global__
void _primitiveAssembly(int numIndices, int curPrimitiveBeginId, Primitive* dev_primitives, PrimitiveDevBufPointers primitive) {

	// index id
	int iid = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (iid < numIndices) {

		// TODO: uncomment the following code for a start
		// This is primitive assembly for triangles

		int pid;

		if (primitive.primitiveMode == TINYGLTF_MODE_TRIANGLES)
		{
			pid = iid / (int)primitive.primitiveType;
			dev_primitives[pid + curPrimitiveBeginId].primitiveType = Triangle;
			dev_primitives[pid + curPrimitiveBeginId].v[iid % (int)primitive.primitiveType]
				= primitive.dev_verticesOut[primitive.dev_indices[iid]];
		}
		else if (primitive.primitiveMode == TINYGLTF_MODE_LINE)
		{
			pid = iid / (int)primitive.primitiveType;
			dev_primitives[pid + curPrimitiveBeginId].primitiveType = Line;
			dev_primitives[pid + curPrimitiveBeginId].v[iid % (int)primitive.primitiveType]
				= primitive.dev_verticesOut[primitive.dev_indices[iid]];
		}
		else if (primitive.primitiveMode == TINYGLTF_MODE_POINTS)
		{
			pid = iid / (int)primitive.primitiveType;
			dev_primitives[pid + curPrimitiveBeginId].primitiveType = Point;
			dev_primitives[pid + curPrimitiveBeginId].v[iid % (int)primitive.primitiveType]
				= primitive.dev_verticesOut[primitive.dev_indices[iid]];
		}
	}
}

__host__ __device__ glm::vec2 getVec2AtCoord(const glm::vec3 &baryCoord, const glm::vec2 input[3])
{
	return baryCoord.x * input[0] + baryCoord.y * input[1] + baryCoord.z * input[2];
}
__host__ __device__ float getFloatAtCoord(const glm::vec3 barycentricCoord, const float input[3])
{
	return barycentricCoord.x * input[0] + barycentricCoord.y * input[1] + barycentricCoord.z * input[2];
}
__host__ __device__ glm::vec3 getVec3AtCoord(const glm::vec3 barycentricCoord, const glm::vec3 input[3])
{
	return barycentricCoord.x * input[0] + barycentricCoord.y * input[1] + barycentricCoord.z * input[2];
}
__host__ __device__ glm::vec2 getTexcoordAtCoord(const glm::vec3 &baryCoord,
	const glm::vec2 _texcoord[3], const float triDepth_1[3]) 
{
	const glm::vec2 texcoord[3] = 
	{
		_texcoord[0] * triDepth_1[0],
		_texcoord[1] * triDepth_1[1],
		_texcoord[2] * triDepth_1[2]
	};
	const glm::vec2 numerator = getVec2AtCoord(baryCoord, texcoord);
	const float denomenator = getFloatAtCoord(baryCoord, triDepth_1);

	return numerator / denomenator;
}

__global__
void rasterizePrimitive(int totalNumPrimitives, Primitive *dev_primitives,
	Fragment *dev_fragmentBuffer, float *dev_depth, int width, int height,
	unsigned int *dev_mutex) 
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx >= totalNumPrimitives)
	{
		return;
	}

	const Primitive &primitive = dev_primitives[idx];
	const glm::vec3 tri[3] =
	{
		glm::vec3(primitive.v[0].pos),
		glm::vec3(primitive.v[1].pos),
		glm::vec3(primitive.v[2].pos),
	};
	const glm::vec3 eyePos[3] =
	{
		primitive.v[0].eyePos,
		primitive.v[1].eyePos,
		primitive.v[2].eyePos
	};
	const glm::vec3 eyeNor[3] =
	{
		primitive.v[0].eyeNor,
		primitive.v[1].eyeNor,
		primitive.v[2].eyeNor
	};
	const glm::vec2 texcoord0[3] =
	{
		primitive.v[0].texcoord0,
		primitive.v[1].texcoord0,
		primitive.v[2].texcoord0
	};
	const float triDepth_1[3] =
	{
		1.f / tri[0].z,
		1.f / tri[1].z,
		1.f / tri[2].z
	};

	if (primitive.primitiveType == Triangle)
	{
		TextureData *pDiffuseTexData = primitive.v[0].dev_diffuseTex;
		int diffuseTexWidth = 0;
		int diffuseTexHeight = 0;

		if (CULL_FACE)
		{
			if (calculateSignedArea(tri) >= 0.f)
			{
				return;
			}
		}

		if (pDiffuseTexData != NULL) 
		{
			diffuseTexWidth = primitive.v[0].texWidth;
			diffuseTexHeight = primitive.v[0].texHeight;
		}

		const AABB aabb = getAABBForTriangle(tri);

		const int left = glm::max(0, (int)aabb.min.x - 1);
		const int right = glm::min(width, (int)aabb.max.x + 1);
		const int bottom = glm::max(0, (int)aabb.min.y - 1);
		const int top = glm::min(height, (int)aabb.max.y + 1);

		if (left >= right || bottom >= top)
		{
			return;
		}

		for (int i = left; i < right; ++i)
		{
			for (int j = bottom; j < top; ++j)
			{
				const int pixelId = i + j * width;
				const glm::vec2 p(i + .5f, j + .5f);
				const glm::vec3 baryCoord = calculateBarycentricCoordinate(tri, p);

				// outsides triangle
				if (!isBarycentricCoordInBounds(baryCoord)) continue;

				const float z = getZAtCoordinate(baryCoord, eyePos);
				// too far or too near
				//if (z < 0.f || z > 1.f) continue;

				// depth test, account for race condition when accessing depth buffer
				const float depth = z;
				bool isOccluded = true;

				if (MUTEX_ON)
				{
					bool isSet = false;
					while (!isSet)
					{
						isSet = atomicCAS(&dev_mutex[pixelId], 0, 1) == 0;
						if (isSet)
						{
							if (dev_depth[pixelId] > depth)
							{
								dev_depth[pixelId] = depth;
								isOccluded = false;
							}
							dev_mutex[pixelId] = 0;
						}
					}
				}
				else
				{
					if (dev_depth[pixelId] > depth)
					{
						dev_depth[pixelId] = depth;
						isOccluded = false;
					}
				}
				
				if (isOccluded)
				{
					continue;
				}

				Fragment &fragment = dev_fragmentBuffer[pixelId];
				fragment.eyePos = getVec3AtCoord(baryCoord, eyePos);
				fragment.eyeNor = glm::normalize(getVec3AtCoord(baryCoord, eyeNor));
				fragment.texcoord0 = getTexcoordAtCoord(baryCoord, texcoord0, triDepth_1);
				if (pDiffuseTexData != NULL)
				{
					fragment.dev_diffuseTex = pDiffuseTexData;
					fragment.texWidth = diffuseTexWidth;
					fragment.texHeight = diffuseTexHeight;
				}
			}
		}
	}
}


__host__ __device__
void rasterizeLineHelper(const glm::vec3 &a, const glm::vec3 &b,
	Fragment *dev_fragmentBuffer, int w, int h)
{
	const int left = glm::max(0, (int)glm::min(a.x, b.x));
	const int right = glm::min(w, (int)glm::max(a.x, b.x) + 1);
	const int bottom = glm::max(0, (int)glm::min(a.y, b.y));
	const int top = glm::min(h, (int)glm::max(a.y, b.y) + 1);

	// outsides window
	if (left >= right && bottom >= top)
	{
		return;
	}

	int begin = right - left >= top - bottom ? left : bottom;
	int end = right - left >= top - bottom ? right : top;

	for (int i = 0; i <= end - begin; ++i)
	{
		const glm::vec3 p = lerp(a, b, (i + 0.0f) / (end - begin));
		const int index = (int)p.x + (int)p.y * w;

		dev_fragmentBuffer[index].color = glm::vec3(1.0f, 1.0f, 1.0f);
	}
}

__global__
void rasterizeLine(int totalNumPrimitives, Primitive *dev_primitives,
	Fragment *dev_fragmentBuffer, int w, int h)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx >= totalNumPrimitives)
	{
		return;
	}

	const Primitive &primitive = dev_primitives[idx];
	const glm::vec3 &v0 = glm::vec3(primitive.v[0].pos);
	const glm::vec3 &v1 = glm::vec3(primitive.v[1].pos);
	const glm::vec3 &v2 = glm::vec3(primitive.v[2].pos);

	rasterizeLineHelper(v0, v1, dev_fragmentBuffer, w, h);
	rasterizeLineHelper(v1, v2, dev_fragmentBuffer, w, h);
	rasterizeLineHelper(v2, v0, dev_fragmentBuffer, w, h);
}

__global__
void rasterizePoint(int totalNumPrimitives, Primitive *dev_primitives,
	Fragment *dev_fragmentBuffer, int w, int h)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx >= totalNumPrimitives)
	{
		return;
	}

	const Primitive &primitive = dev_primitives[idx];
	const glm::vec3 tri[3] =
	{
		glm::vec3(primitive.v[0].pos),
		glm::vec3(primitive.v[1].pos),
		glm::vec3(primitive.v[2].pos)
	};

	for (int i = 0; i < 3; ++i)
	{
		const int x = (int)tri[i].x;
		const int y = (int)tri[i].y;
		const int index = x + y * w;

		if (x < 0 || x >= w || y < 0 || y >= h)
		{
			continue;
		}
		dev_fragmentBuffer[index].color = glm::vec3(1.0f, 1.0f, 1.0f);
	}
}
/**
* Perform rasterization.
*/
void rasterize(uchar4 *pbo, const glm::mat4 & MVP, const glm::mat4 & MV, const glm::mat3 MV_normal) {
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

	float time_vtfaa = 0;
	float time_primitive = 0;

	for (; it != itEnd; ++it)
	{
		auto p = (it->second).begin();	// each primitive
		auto pEnd = (it->second).end();
		for (; p != pEnd; ++p)
		{
			dim3 numBlocksForVertices((p->numVertices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
			dim3 numBlocksForIndices((p->numIndices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
			
			_vertexTransformAndAssembly<<<numBlocksForVertices, numThreadsPerBlock>>>(p->numVertices, *p, MVP, MV, MV_normal, width, height);
			checkCUDAError("Vertex Processing");
			cudaDeviceSynchronize();

			_primitiveAssembly<<<numBlocksForIndices, numThreadsPerBlock>>>
				(p->numIndices,
					curPrimitiveBeginId,
					dev_primitives,
					*p);
			checkCUDAError("Primitive Assembly");
			curPrimitiveBeginId += p->numPrimitives;
		}
	}
	
	checkCUDAError("Vertex Processing and Primitive Assembly");

	cudaMemset(dev_fragmentBuffer, 0, width * height * sizeof(Fragment));
	initDepth<<<blockCount2d, blockSize2d>>>(width, height, dev_depth);

	// TODO: rasterize

	dim3 primitiveBlocks((totalNumPrimitives + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);

	if (DISPLAY_TRIANGLE)
	{
		rasterizePrimitive<<<primitiveBlocks, numThreadsPerBlock>>>(
			totalNumPrimitives, dev_primitives, dev_fragmentBuffer, dev_depth,
			width, height, dev_mutex);
	}
	if (DISPLAY_LINE)
	{
		rasterizeLine<<<primitiveBlocks, numThreadsPerBlock>>>(
			totalNumPrimitives, dev_primitives, dev_fragmentBuffer,
			width, height);
	}
	if (DISPLAY_POINT)
	{
		rasterizePoint<<<primitiveBlocks, numThreadsPerBlock>>>(
			totalNumPrimitives, dev_primitives, dev_fragmentBuffer,
			width, height);
	}
	checkCUDAError("rasterize primitive");

	render<<<blockCount2d, blockSize2d>>>(width, height, dev_fragmentBuffer, dev_framebuffer);

	checkCUDAError("fragment shader");
	// Copy framebuffer into OpenGL buffer for OpenGL previewing
	sendImageToPBO<<<blockCount2d, blockSize2d>>>(pbo, width, height, dev_framebuffer);
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
	checkCUDAError("rasterize Free");
}