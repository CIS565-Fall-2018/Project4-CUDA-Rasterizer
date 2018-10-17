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

#include <device_functions.h>

#define PRIM_PER_BLOCK_MAX 256

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

		glm::vec3 eyePos;	// eye space position used for shading
		glm::vec3 eyeNor;	// eye space normal used for shading, cuz normal will go wrong after perspective transformation
	   // glm::vec3 col;
		glm::vec2 texcoord0;
		// ...
	};

	struct Primitive {
		PrimitiveType primitiveType = Triangle;	// C++ 11 init
		TextureData* dev_diffuseTex = NULL;
		int diffuseTexWidth;//me
		int diffuseTexHeight;//me
		int diffuseTexComponentCount;//me

		VertexOut v[3];

		__device__ __host__ Primitive()
		{
			primitiveType = Triangle;
			dev_diffuseTex = NULL;
			diffuseTexWidth = 0;
			diffuseTexHeight = 0;
			diffuseTexComponentCount = 0;
		}

		__device__ __host__ Primitive(const Primitive& p)
		{
			primitiveType = p.primitiveType;
			dev_diffuseTex = p.dev_diffuseTex;
			diffuseTexWidth = p.diffuseTexWidth;
			diffuseTexHeight = p.diffuseTexHeight;
			diffuseTexComponentCount = p.diffuseTexComponentCount;
			v[0] = p.v[0];
			v[1] = p.v[1];
			v[2] = p.v[2];
		}

		__device__ __host__ const Primitive& operator=(const Primitive& p)
		{
			primitiveType = p.primitiveType;
			dev_diffuseTex = p.dev_diffuseTex;
			diffuseTexWidth = p.diffuseTexWidth;
			diffuseTexHeight = p.diffuseTexHeight;
			diffuseTexComponentCount = p.diffuseTexComponentCount;
			v[0] = p.v[0];
			v[1] = p.v[1];
			v[2] = p.v[2];

			return *this;
		}
	};

	struct Fragment {
		glm::vec3 color;

		// TODO: add new attributes to your Fragment
		// The attributes listed below might be useful, 
		// but always feel free to modify on your own

		glm::vec3 eyePos;	// eye space position used for shading
		glm::vec3 eyeNor;
		VertexAttributeTexcoord texcoord0;
		TextureData* dev_diffuseTex;
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
		int diffuseTexComponentCount;//me
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

static int * dev_depth = NULL;	// you might need this buffer when doing depth test

static glm::vec3 lightDir_dev = glm::vec3(0, 0, 0);

/**
 * Kernel that writes the image to the OpenGL PBO directly.
 */
__global__
void sendImageToPBO(uchar4 *pbo, int w, int h, glm::vec3 *image) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	//me
	//since in OpenGL, texture is filled row by row starting from bottom left corner,
	//we need to re-arrange the coordinates so that it maps from screen space to texture space
	//so that we can display it using OpenGL by drawing a square whose uv origin is at the bottom left corner as well
	// 0,0-------w,0                 0,1------1,1
	//	|         |	                  |        |
	//  |         |    ---------->    |        |
	//	|         |					  |        |
	// 0,h-------w,h                 0,0------1,0

	int indexFrom = x + ((h - 1 - y) * w);

	int index = x + (y * w);

	if (x < w && y < h) {
		glm::vec3 color;
		color.x = glm::clamp(image[indexFrom].x, 0.0f, 1.0f) * 255.0;
		color.y = glm::clamp(image[indexFrom].y, 0.0f, 1.0f) * 255.0;
		color.z = glm::clamp(image[indexFrom].z, 0.0f, 1.0f) * 255.0;
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

//!!!IMPORTANT!!!
//When passing arguments from CPU to GPU using a __global__ kernal, the arguments are
//either GPU device pointers or values. They can not be CPU pointers(references).
//!!!IMPORTANT!!!
__global__
void render(int w, int h, Fragment *fragmentBuffer, glm::vec3 *framebuffer, glm::vec3 lightDir) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * w);

	if (x < w && y < h) {
		// TODO: add your fragment shader code here

		Fragment frag = fragmentBuffer[index];//copy to reduce global memory access

		//half Lambert
		glm::vec3 color = frag.color * (glm::dot(glm::normalize(-lightDir), glm::normalize(frag.eyeNor)) * 0.5f + 0.5f);
		
		framebuffer[index] = color;
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
	cudaMalloc(&dev_depth, width * height * sizeof(int));

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
					int diffuseTexComponentCount = 0;
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
									diffuseTexComponentCount = image.component;

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
						diffuseTexComponentCount,

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

__device__ bool isZero(float v)
{
	if (glm::abs(v) < EPSILON)
		return true;
	else
		return false;
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
		glm::vec4 tempP;
		glm::vec4 tempEyeP;
		glm::vec3 tempEyeN;
		glm::vec2 tempTexcoord0;

		if (primitive.dev_position != NULL)
		{
			tempP = MVP * glm::vec4(primitive.dev_position[vid], 1);
			tempP.x = tempP.x / tempP.w;
			tempP.y = tempP.y / tempP.w;
			tempP.z = tempP.z / tempP.w;
			//tempP.w remain the same for recovering from perspective correct interpolation
			tempP.x = (tempP.x * 0.5 + 0.5) * width;//[-1,1] to [0,1] to [0, width]
			tempP.y = (tempP.y * -0.5 + 0.5) * height;//[-1,1] to [1,0] to [height, 0], since in screen space, origin is at top left corner
			tempP.z = (tempP.z * 0.5 + 0.5);//[-1,1] to [0,1]
			tempEyeP = MV * glm::vec4(primitive.dev_position[vid], 1);
		}
		if (primitive.dev_normal != NULL)
		{
			tempEyeN = MV_normal * glm::vec3(primitive.dev_normal[vid]);

		}
		if (primitive.dev_texcoord0 != NULL)
		{
			tempTexcoord0 = primitive.dev_texcoord0[vid];
		}

		// TODO: Apply vertex assembly here
		// Assemble all attribute arraies into the primitive array

		primitive.dev_verticesOut[vid].pos = tempP;
		primitive.dev_verticesOut[vid].eyePos = glm::vec3(tempEyeP);//w will always be 1, no need to divide
		primitive.dev_verticesOut[vid].eyeNor = glm::normalize(tempEyeN);
		primitive.dev_verticesOut[vid].texcoord0 = tempTexcoord0;
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

		int pid;	// id for cur primitives vector
		
		if (primitive.primitiveMode == TINYGLTF_MODE_TRIANGLES) {
			pid = iid / (int)primitive.primitiveType;
			dev_primitives[pid].primitiveType = Triangle;//me
			dev_primitives[pid].dev_diffuseTex = primitive.dev_diffuseTex;//me
			dev_primitives[pid].diffuseTexComponentCount = primitive.diffuseTexComponentCount;//me
			dev_primitives[pid].diffuseTexHeight = primitive.diffuseTexHeight;//me
			dev_primitives[pid].diffuseTexWidth = primitive.diffuseTexWidth;//me
			dev_primitives[pid].v[iid % (int)primitive.primitiveType] = primitive.dev_verticesOut[primitive.dev_indices[iid]];
		}


		// TODO: other primitive types (point, line)
	}

}

__device__ bool intersectLineHorizontal(float horizontal, float x1, float y1, float x2, float y2, float* result)
{
	if (x1 == x2)
	{
		if ((horizontal >= y1 && horizontal <= y2) || (horizontal >= y2 && horizontal <= y1))
		{
			if (result != nullptr) *result = x1;//or x2
			return true;
		}
		else
		{
			return false;
		}
	}
	else
	{
		if (y1 == y2)
		{
			if (horizontal == y1)//or y2
			{
				if (result != nullptr) *result = x1 < x2 ? x1 : x2;
				return true;
			}
			else
			{
				return false;
			}
		}
		else
		{
			float temp = (x2 - x1)*(horizontal - y1) / (y2 - y1) + x1;
			if (temp <= glm::max(x1, x2) && temp >= glm::min(x1, x2))//if intersection is within the line segment
			{
				if(result != nullptr) *result = temp;
				return true;
			}
			else
			{
				return false;
			}
		}
	}
}

template<class T>
__device__ T barycentricInterpolate(const glm::vec3 &P,
	const glm::vec3 &p0, const glm::vec3 &p1, const glm::vec3 &p2,
	const T &v0, const T &v1, const T &v2)
{
	float S = 0.5f *glm::length(glm::cross(p0 - p1, p2 - p1));
	float s0 = 0.5f *glm::length(glm::cross(p1 - P, p2 - P)) / S;
	float s1 = 0.5f *glm::length(glm::cross(p2 - P, p0 - P)) / S;
	float s2 = 0.5f*glm::length(glm::cross(p0 - P, p1 - P)) / S;
	return v0 * s0 + v1 * s1 + v2 * s2;
}

__global__
void scanLineZ(int primitiveCount, const Primitive* primitives, int width, int height, int* depths)
{
	int pid = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (pid < primitiveCount)
	{
		Primitive primitive = primitives[pid];//copy so that no global memory access in this function anymore
		int yMin = INT_MAX;
		int yMax = INT_MIN;
		int vertCount = 0;

		if (primitive.primitiveType == Triangle)
		{
			vertCount = 3;

			//find the vertical boundary
			for (int i = 0; i < vertCount; i++)
			{
				if (primitive.v[i].pos.y < yMin) yMin = primitive.v[i].pos.y;
				if (primitive.v[i].pos.y > yMax) yMax = primitive.v[i].pos.y;
			}

			//clamp the vertical boundary
			yMin = yMin < 0 ? 0 : yMin > height - 1 ? height - 1 : yMin;
			yMax = yMax < 0 ? 0 : yMax > height - 1 ? height - 1 : yMax;

			//save the fragment row by row
			for (int i = yMin; i <= yMax; i++)
			{
				float xResult[3] = { -1, -1, -1 };
				int xMin = INT_MAX;
				int xMax = INT_MIN;
				bool hit = false;

				//find the horizontal boundary
				for (int j = 0; j < 3; j++)
				{
					int jPlusOne = (j + 1) % 3;
					if (intersectLineHorizontal(i, primitive.v[j].pos.x, primitive.v[j].pos.y, primitive.v[jPlusOne].pos.x, primitive.v[jPlusOne].pos.y, &xResult[j]))
					{
						if (xResult[j] > xMax) xMax = xResult[j];
						if (xResult[j] < xMin) xMin = xResult[j];
						hit = true;
					}
				}

				if (hit)
				{
					//clamp the horizontal boundary
					xMin = xMin < 0 ? 0 : xMin > width - 1 ? width - 1 : xMin;
					xMax = xMax < 0 ? 0 : xMax > width - 1 ? width - 1 : xMax;

					//loop from xMin to xMax
					for (int j = xMin; j <= xMax; j++)
					{
						glm::vec2 fragmentPos(j, i);

						Fragment fragment;

						glm::vec3 P(fragmentPos, 0);
						glm::vec3 p0(primitive.v[0].pos.x, primitive.v[0].pos.y, 0);
						glm::vec3 p1(primitive.v[1].pos.x, primitive.v[1].pos.y, 0);
						glm::vec3 p2(primitive.v[2].pos.x, primitive.v[2].pos.y, 0);

						//[0,1]
						float fragmentZ = barycentricInterpolate(P, p0, p1, p2,
							primitive.v[0].pos.z,
							primitive.v[1].pos.z,
							primitive.v[2].pos.z);

						if (fragmentZ > 0 && fragmentZ < 1)
						{

							int intZ = fragmentZ * INT_MAX;

							//copy the fragment to fragment buffer
							atomicMin(&depths[i * width + j], intZ);
						}
					}
				}
			}
		}
		else if (primitive.primitiveType == Line)
		{
			//do nothing
		}
		else if (primitive.primitiveType == Point)
		{
			//do nothing
		}
	}
}

//alpha is ignored
__device__ glm::vec3 bilinearSample(const TextureData* dev_texture, int comp, int width, int height, const glm::vec2& uv)
{
	float fracU = uv.x * width;
	int floorU = fracU;
	fracU -= floorU;
	float fracV = uv.y * height;
	int floorV = fracV;
	fracV -= floorV;
	int ceilU = glm::clamp(floorU + 1, 0, width - 1);
	int ceilV = glm::clamp(floorV + 1, 0, height - 1);
	int indexFUFV = comp * (floorU + floorV * width);
	int indexFUCV = comp * (floorU + ceilV * width);
	int indexCUFV = comp * (ceilU + floorV * width);
	int indexCUCV = comp * (ceilU + ceilV * width);
	glm::vec3 colFUFV(dev_texture[indexFUFV] / 255.f, dev_texture[indexFUFV + 1] / 255.f, dev_texture[indexFUFV + 2] / 255.f);
	glm::vec3 colFUCV(dev_texture[indexFUCV] / 255.f, dev_texture[indexFUCV + 1] / 255.f, dev_texture[indexFUCV + 2] / 255.f);
	glm::vec3 colCUFV(dev_texture[indexCUFV] / 255.f, dev_texture[indexCUFV + 1] / 255.f, dev_texture[indexCUFV + 2] / 255.f);
	glm::vec3 colCUCV(dev_texture[indexCUCV] / 255.f, dev_texture[indexCUCV + 1] / 255.f, dev_texture[indexCUCV + 2] / 255.f);
	glm::vec3 colFU = (1.f - fracV) * colFUFV + fracV * colFUCV;
	glm::vec3 colCU = (1.f - fracV) * colCUFV + fracV * colCUCV;
	return (1.f - fracU) * colFU + fracU * colCU;
}

__global__
void scanLine(int primitiveCount, const Primitive* primitives, int width, int height, int* depths, Fragment* fragments)
{
	int pid = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (pid < primitiveCount)
	{
		Primitive primitive = primitives[pid];//copy so that no global memory access in this function anymore
		int yMin = INT_MAX;
		int yMax = INT_MIN;
		int vertCount = 0;

		if (primitive.primitiveType == Triangle)
		{
			vertCount = 3;

			//find the vertical boundary
			for (int i = 0; i < vertCount; i++)
			{
				if (primitive.v[i].pos.y < yMin) yMin = primitive.v[i].pos.y;
				if (primitive.v[i].pos.y > yMax) yMax = primitive.v[i].pos.y;
			}

			//clamp the vertical boundary
			yMin = yMin < 0 ? 0 : yMin > height - 1 ? height - 1 : yMin;
			yMax = yMax < 0 ? 0 : yMax > height - 1 ? height - 1 : yMax;

			//save the fragment row by row
			for (int i = yMin; i <= yMax; i++)
			{
				float xResult[3] = { -1, -1, -1 };
				int xMin = INT_MAX;
				int xMax = INT_MIN;
				bool hit = false;

				//find the horizontal boundary
				for (int j = 0; j < 3; j++)
				{
					int jPlusOne = (j + 1) % 3;
					if (intersectLineHorizontal(i, primitive.v[j].pos.x, primitive.v[j].pos.y, primitive.v[jPlusOne].pos.x, primitive.v[jPlusOne].pos.y, &xResult[j]))
					{
						if (xResult[j] > xMax) xMax = xResult[j];
						if (xResult[j] < xMin) xMin = xResult[j];
						hit = true;
					}
				}

				if (hit)
				{
					//clamp the horizontal boundary
					xMin = xMin < 0 ? 0 : xMin > width - 1 ? width - 1 : xMin;
					xMax = xMax < 0 ? 0 : xMax > width - 1 ? width - 1 : xMax;

					//loop from xMin to xMax
					for (int j = xMin; j <= xMax; j++)
					{
						glm::vec2 fragmentPos(j, i);

						Fragment fragment;

						glm::vec3 P(fragmentPos, 0);
						glm::vec3 p0(primitive.v[0].pos.x, primitive.v[0].pos.y, 0);
						glm::vec3 p1(primitive.v[1].pos.x, primitive.v[1].pos.y, 0);
						glm::vec3 p2(primitive.v[2].pos.x, primitive.v[2].pos.y, 0);
						
						float fragmentZ = barycentricInterpolate(P, p0, p1, p2,
							primitive.v[0].pos.z,
							primitive.v[1].pos.z,
							primitive.v[2].pos.z);

						//depth clip, OpenGL and D3D do it in clip space, we do it in screen space due to the structure of our pipeline
						if (fragmentZ > 0 && fragmentZ < 1)
						{

							int intZ = fragmentZ * INT_MAX;

							//depth test
							if (intZ <= depths[i * width + j])
							{
								//for perspective correct interpolation
								float w = 1.f / barycentricInterpolate(P, p0, p1, p2,
									1.f / primitive.v[0].pos.w,
									1.f / primitive.v[1].pos.w,
									1.f / primitive.v[2].pos.w);

								//set the attributes of the fragment
								fragment.eyePos = w * barycentricInterpolate(P, p0, p1, p2,
									primitive.v[0].eyePos / primitive.v[0].pos.w,
									primitive.v[1].eyePos / primitive.v[1].pos.w,
									primitive.v[2].eyePos / primitive.v[2].pos.w);

								fragment.eyeNor = w * barycentricInterpolate(P, p0, p1, p2,
									primitive.v[0].eyeNor / primitive.v[0].pos.w,
									primitive.v[1].eyeNor / primitive.v[1].pos.w,
									primitive.v[2].eyeNor / primitive.v[2].pos.w);

								fragment.texcoord0 = w * barycentricInterpolate(P, p0, p1, p2,
									primitive.v[0].texcoord0 / primitive.v[0].pos.w,
									primitive.v[1].texcoord0 / primitive.v[1].pos.w,
									primitive.v[2].texcoord0 / primitive.v[2].pos.w);

								fragment.dev_diffuseTex = primitive.dev_diffuseTex;//this should be an uniform

								//temp glm::vec3(1, 0, 1);
								fragment.color = primitive.dev_diffuseTex == NULL ? glm::vec3(1, 1, 1) : 
									bilinearSample(primitive.dev_diffuseTex, 
										primitive.diffuseTexComponentCount, 
										primitive.diffuseTexWidth, 
										primitive.diffuseTexHeight, 
										fragment.texcoord0);

								//copy the fragment to fragment buffer
								fragments[i * width + j] = fragment;
							}
						}
					}
				}
			}
		}
		else if (primitive.primitiveType == Line)
		{
			//do nothing
		}
		else if (primitive.primitiveType == Point)
		{
			//do nothing
		}
	}
}

__device__
bool intersectLineSegments(const glm::vec2& a1, const glm::vec2& a2, const glm::vec2& b1, const glm::vec2& b2)
{
	float x = 0;
	float y = 0;
	if (a1.x == a2.x)//no ka
	{
		x = a1.x;
		if (b1.x == b2.x)//no kb, parallel
		{
			return false;//progressive
		}
		else//kb
		{
			float kb = (b1.y - b2.y) / (b1.x - b2.x);
			float bb = b1.y - kb * b1.x;
			y = kb * x + bb;
		}
	}
	else//ka
	{
		float ka = (a1.y - a2.y) / (a1.x - a2.x);
		float ba = a1.y - ka * a1.x;
		if (b1.x == b2.x)//no kb
		{
			x = b1.x;
			y = ka * x + ba;
		}
		else//kb
		{
			float kb = (b1.y - b2.y) / (b1.x - b2.x);
			float bb = b1.y - kb * b1.x;
			if (ka == kb)//parallel
			{
				return false;//progressive
			}
			else
			{
				x = (bb - ba) / (ka - kb);
				y = ka * x + ba;
			}
		}
	}

	glm::vec2 minA = glm::min(a1, a2);
	glm::vec2 maxA = glm::max(a1, a2);
	glm::vec2 minB = glm::min(b1, b2);
	glm::vec2 maxB = glm::max(b1, b2);

	if (x > minA.x && y > minA.y && 
		x < maxA.x && y < maxA.y &&
		x > minB.x && y > minB.y &&
		x < maxB.x && y < maxB.y)
		return true;
	else
		return false;

}

__device__
bool triangleInTile(const glm::vec3 v[3], 
	//xMin, xMax, yMin, yMax
	const float MinMaxXY[4])
{
	glm::vec3 min = glm::min(v[0], glm::min(v[1], v[2]));
	glm::vec3 max = glm::max(v[0], glm::max(v[1], v[2]));
	
	//early rejection
	if ((max.x < MinMaxXY[0]) || (min.x > MinMaxXY[1]) || (max.y < MinMaxXY[2]) || (min.y > MinMaxXY[3]))
	{
		return false;
	}

	return true;
}

template<int tileWidth, int tileHeight>
__global__
void scanLineTile(int primitiveCount, const Primitive* primitives, int width, int height, Fragment* fragments)
{
	//__shared__ Fragment shared_fragments[tileWidth * tileHeight];//what's the point of this? we are launching threads per pixel
	//__shared__ int shared_depths[tileWidth * tileHeight];//what's the point of this? we are launching threads per pixel
	__shared__ Primitive shared_primitives[PRIM_PER_BLOCK_MAX];
	__shared__ int shared_primitives_count;

	int pixelX = blockIdx.x * blockDim.x + threadIdx.x;
	int pixelY = blockIdx.y * blockDim.y + threadIdx.y;
	int pixelId = pixelX + pixelY * width;
	int threadId = threadIdx.x + threadIdx.y * blockDim.x;

	//shared_depths[threadId] = INT_MAX;
	int depth = INT_MAX;

	int threadCount = blockDim.x * blockDim.y;
	int primitivePerThread = (primitiveCount - 1 + threadCount) / threadCount;
	
	if (threadIdx.x == 0 && threadIdx.y == 0) shared_primitives_count = 0;

	__syncthreads();

	for (int i = 0; i < primitivePerThread; i++)
	{
		int temp = threadId * primitivePerThread + i;
		if (temp < primitiveCount)
		{
			Primitive tempPrimitive = primitives[temp];
			glm::vec3 tri[3] = { glm::vec3(tempPrimitive.v[0].pos), glm::vec3(tempPrimitive.v[1].pos) ,glm::vec3(tempPrimitive.v[2].pos) };
			float MinMaxXY[4] = { (float)blockIdx.x * blockDim.x, (blockIdx.x + 1.f) * blockDim.x, (float)blockIdx.y * blockDim.y, (blockIdx.y + 1.f) * blockDim.y };
			if (triangleInTile(tri, MinMaxXY))//inside
			{
				int oldPrimitiveCount = atomicAdd(&shared_primitives_count, 1);
				//printf("%d,%d:%d,%d:%d\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, oldPrimitiveCount);
				if (oldPrimitiveCount >= 0 && oldPrimitiveCount < PRIM_PER_BLOCK_MAX)
				{
					shared_primitives[oldPrimitiveCount] = tempPrimitive;
				}
				else
				{
					break;
				}
			}
		}
	}

	//if (threadIdx.x == 0 && threadIdx.y == 0 && shared_primitives_count>0) printf("%d,%d:%d\n", blockIdx.x, blockIdx.y, shared_primitives_count);

	__syncthreads();

	//printf("%d\n", shared_primitives_count);
	int actual_shared_primitives_count = shared_primitives_count < PRIM_PER_BLOCK_MAX ? shared_primitives_count : PRIM_PER_BLOCK_MAX;

	if (actual_shared_primitives_count > 0 && pixelX < width && pixelY < height)
	{
		glm::vec2 fragmentPos(pixelX, pixelY);
		Fragment fragment;
		glm::vec3 P(fragmentPos, 0);
		bool bingo = false;

		//depth pass
		for (int i = 0; i < actual_shared_primitives_count; i++)
		{
			Primitive primitive = shared_primitives[i];//copy so that no global memory access in this function anymore
			
			glm::vec3 p0(primitive.v[0].pos.x, primitive.v[0].pos.y, 0);
			glm::vec3 p1(primitive.v[1].pos.x, primitive.v[1].pos.y, 0);
			glm::vec3 p2(primitive.v[2].pos.x, primitive.v[2].pos.y, 0);

			//float bary = barycentricInterpolate<float>(P, p0, p1, p2, 1.f, 1.f, 1.f);
			//if (glm::abs(bary - 1.0) < EPSILON)//inside the triangle

			glm::vec3 tri[3];
			tri[0] = glm::vec3(primitive.v[0].pos);
			tri[1] = glm::vec3(primitive.v[1].pos);
			tri[2] = glm::vec3(primitive.v[2].pos);

			glm::vec3 baryCoords = calculateBarycentricCoordinate(tri, glm::vec2(pixelX, pixelY));
			bool isInsideTriangle = isBarycentricCoordInBounds(baryCoords);
			if (isInsideTriangle)
			{
				float fragmentZ = barycentricInterpolate(P, p0, p1, p2,
					primitive.v[0].pos.z,
					primitive.v[1].pos.z,
					primitive.v[2].pos.z);

				//depth clip, OpenGL and D3D do it in clip space, we do it in screen space due to the structure of our pipeline
				if (fragmentZ > 0 && fragmentZ < 1)
				{

					int intZ = fragmentZ * INT_MAX;
					if (intZ < depth)//shared_depths[threadId])
					{
						depth = intZ;
					}
				}
			}
		}
		
		//second pass
		for (int i = 0; i < actual_shared_primitives_count; i++)
		{
			Primitive primitive = shared_primitives[i];//copy so that no global memory access in this function anymore
			glm::vec3 p0(primitive.v[0].pos.x, primitive.v[0].pos.y, 0);
			glm::vec3 p1(primitive.v[1].pos.x, primitive.v[1].pos.y, 0);
			glm::vec3 p2(primitive.v[2].pos.x, primitive.v[2].pos.y, 0);

			//float bary = barycentricInterpolate<float>(P, p0, p1, p2, 1.f, 1.f, 1.f);
			//if (glm::abs(bary-1.0) < EPSILON)//inside the triangle

			glm::vec3 tri[3];
			tri[0] = glm::vec3(primitive.v[0].pos);
			tri[1] = glm::vec3(primitive.v[1].pos);
			tri[2] = glm::vec3(primitive.v[2].pos);

			glm::vec3 baryCoords = calculateBarycentricCoordinate(tri, glm::vec2(pixelX, pixelY));
			bool isInsideTriangle = isBarycentricCoordInBounds(baryCoords);
			if (isInsideTriangle)
			{
				float fragmentZ = barycentricInterpolate(P, p0, p1, p2,
					primitive.v[0].pos.z,
					primitive.v[1].pos.z,
					primitive.v[2].pos.z);

				//depth clip, OpenGL and D3D do it in clip space, we do it in screen space due to the structure of our pipeline
				if (fragmentZ > 0 && fragmentZ < 1)
				{

					int intZ = fragmentZ * INT_MAX;

					//depth test
					if (intZ <= depth)//shared_depths[threadId])
					{
						//for perspective correct interpolation
						float w = 1.f / barycentricInterpolate(P, p0, p1, p2,
							1.f / primitive.v[0].pos.w,
							1.f / primitive.v[1].pos.w,
							1.f / primitive.v[2].pos.w);

						//set the attributes of the fragment
						fragment.eyePos = w * barycentricInterpolate(P, p0, p1, p2,
							primitive.v[0].eyePos / primitive.v[0].pos.w,
							primitive.v[1].eyePos / primitive.v[1].pos.w,
							primitive.v[2].eyePos / primitive.v[2].pos.w);

						fragment.eyeNor = w * barycentricInterpolate(P, p0, p1, p2,
							primitive.v[0].eyeNor / primitive.v[0].pos.w,
							primitive.v[1].eyeNor / primitive.v[1].pos.w,
							primitive.v[2].eyeNor / primitive.v[2].pos.w);

						fragment.texcoord0 = w * barycentricInterpolate(P, p0, p1, p2,
							primitive.v[0].texcoord0 / primitive.v[0].pos.w,
							primitive.v[1].texcoord0 / primitive.v[1].pos.w,
							primitive.v[2].texcoord0 / primitive.v[2].pos.w);

						fragment.dev_diffuseTex = primitive.dev_diffuseTex;//this should be an uniform

						fragment.color = primitive.dev_diffuseTex == NULL ? glm::vec3(1, 1, 1) :
							bilinearSample(primitive.dev_diffuseTex,
								primitive.diffuseTexComponentCount,
								primitive.diffuseTexWidth,
								primitive.diffuseTexHeight,
								fragment.texcoord0);

						//copy the fragment to fragment buffer
						//shared_fragments[threadId] = fragment;
						bingo = true;
					}
				}
			}
		}

		//write back
		if(bingo) fragments[pixelId] = fragment;//shared_fragments[threadId];
	}
}


void setLightDirDev(const glm::vec3 &_lightDir)
{
	lightDir_dev = _lightDir;
}

/**
 * Perform rasterization.
 */
void rasterize(uchar4 *pbo, const glm::mat4 & MVP, const glm::mat4 & MV, const glm::mat3 MV_normal, const int mode) {
	int sideLength2d = 8;
	dim3 blockSize2d(sideLength2d, sideLength2d);
	dim3 blockCount2d((width - 1) / blockSize2d.x + 1,
		(height - 1) / blockSize2d.y + 1);

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

				_vertexTransformAndAssembly << < numBlocksForVertices, numThreadsPerBlock >> > (p->numVertices, *p, MVP, MV, MV_normal, width, height);
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
	initDepth << <blockCount2d, blockSize2d >> > (width, height, dev_depth);

	// TODO: rasterize
	cudaDeviceSynchronize();

	if (mode == 0)
	{
		dim3 numTreadsPerBlock(128);
		dim3 numBlocksForPremitives((curPrimitiveBeginId + numTreadsPerBlock.x - 1) / numTreadsPerBlock.x);
		scanLineZ << <numBlocksForPremitives, numTreadsPerBlock >> > (curPrimitiveBeginId, dev_primitives, width, height, dev_depth);
		checkCUDAError("scan line z");
		cudaDeviceSynchronize();
		scanLine << <numBlocksForPremitives, numTreadsPerBlock >> > (curPrimitiveBeginId, dev_primitives, width, height, dev_depth, dev_fragmentBuffer);
		checkCUDAError("scan line");

	}
	else if (mode == 1)
	{
		dim3 tileSize2d(25, 25);
		dim3 tileCount2d((width - 1 + tileSize2d.x) / tileSize2d.x, (height - 1 + tileSize2d.y) / tileSize2d.y);
		scanLineTile<25, 25> << <tileCount2d, tileSize2d >> > (curPrimitiveBeginId, dev_primitives, width, height, dev_fragmentBuffer);
		checkCUDAError("scan line tile");
	}
	cudaDeviceSynchronize();

	// Copy depthbuffer colors into framebuffer
	render << <blockCount2d, blockSize2d >> > (width, height, dev_fragmentBuffer, dev_framebuffer, lightDir_dev);
	checkCUDAError("fragment shader");
	// Copy framebuffer into OpenGL buffer for OpenGL previewing
	sendImageToPBO << <blockCount2d, blockSize2d >> > (pbo, width, height, dev_framebuffer);
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

	checkCUDAError("rasterize Free");
}
