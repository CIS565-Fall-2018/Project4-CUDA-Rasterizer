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

namespace {

	typedef unsigned short VertexIndex;
	typedef glm::vec3 VertexAttributePosition;
	typedef glm::vec3 VertexAttributeNormal;
	typedef glm::vec2 VertexAttributeTexcoord;
	typedef unsigned char TextureData;

	typedef unsigned char BufferByte;

#define NUM_INSTANCES 1

#define AA_SUPER_SAMPLE
//#define AA_MULTI_SAMPLE

#define SSAA_LEVEL 1
#define MSAA_LEVEL 2

	enum PrimitiveType
	{
		Point = 1,
		Line = 2,
		Triangle = 3
	};

	struct VertexOut 
	{
		glm::vec4 pos;
		glm::vec3 eyePos;	// eye space position used for shading
		glm::vec3 eyeNor;	// eye space normal used for shading, cuz normal will go wrong after perspective transformation

		glm::vec3 color;
		glm::vec2 texcoord0;
	};

	struct Primitive 
	{
		int instanceId;
		PrimitiveType primitiveType = Triangle;	// C++ 11 init
		VertexOut v[3];
		TextureData* dev_diffuseTex;
		int diffuseTexWidth;
		int diffuseTexHeight;
	};

	struct Fragment 
	{
		glm::vec3 color;
		glm::vec3 eyePos;
		glm::vec3 eyeNor;
		VertexAttributeTexcoord texcoord0;
		TextureData* dev_diffuseTex;
		int diffuseTexWidth;
		int diffuseTexHeight;
	};

	struct PrimitiveDevBufPointers 
	{
		int primitiveMode;	//from tinygltfloader macro
		
		PrimitiveType primitiveType;
		int numPrimitives;
		int numIndices;
		int numVertices;
		int numInstances;

		int vertexOutStartIndex;


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

static int ssaaWidth = 0;
static int ssaaHeight = 0;

static int msaaWidth = 0;
static int msaaHeight = 0;

static int width = 0;
static int height = 0;

static int originalWidth = 0;
static int originalHeight = 0;

static int totalNumPrimitives = 0;
static Primitive *dev_primitives = NULL;

static Fragment *dev_fragmentBuffer = NULL;

static glm::vec3 *dev_framebuffer = NULL;
static glm::vec3 *dev_AAFrameBuffer = NULL;

static float * dev_depth = NULL;	// you might need this buffer when doing depth test
static unsigned int* dev_mutex = NULL;

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
 * Called once at the beginning of the program to allocate memory.
 */
void rasterizeInit(int w, int h) 
{
	originalWidth = w;
	originalHeight = h;

	ssaaWidth = w * SSAA_LEVEL;
	ssaaHeight = h * SSAA_LEVEL;

	msaaWidth = w * MSAA_LEVEL;
	msaaHeight = h * MSAA_LEVEL;

#ifdef AA_SUPER_SAMPLE
	width = ssaaWidth;
	height = ssaaHeight;
#endif // AA_SUPER_SAMPLE

#ifdef AA_MULTI_SAMPLE
	width = msaaWidth;
	height = msaaHeight;
#endif // AA_MULTI_SAMPLE

	cudaFree(dev_fragmentBuffer);
	cudaMalloc(&dev_fragmentBuffer, width * height * sizeof(Fragment));
	cudaMemset(dev_fragmentBuffer, 0, width * height * sizeof(Fragment));

    cudaFree(dev_framebuffer);
    cudaMalloc(&dev_framebuffer,   width * height * sizeof(glm::vec3));
    cudaMemset(dev_framebuffer, 0, width * height * sizeof(glm::vec3));

	cudaFree(dev_AAFrameBuffer);
	cudaMalloc(&dev_AAFrameBuffer,   originalWidth * originalHeight * sizeof(glm::vec3));
	cudaMemset(dev_AAFrameBuffer, 0, originalWidth * originalHeight * sizeof(glm::vec3));
    
	cudaFree(dev_depth);
	cudaMalloc(&dev_depth, width * height * sizeof(float));

	cudaFree(dev_mutex);
	cudaMalloc(&dev_mutex,   width * height * sizeof(unsigned int));
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
		depth[index] = FLT_MAX;
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

					int numInstances = NUM_INSTANCES;
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
					cudaMalloc(&dev_vertexOut, numVertices * numInstances * sizeof(VertexOut));
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
						numInstances,

						0,

						dev_indices,
						dev_position,
						dev_normal,
						dev_texcoord0,

						dev_diffuseTex,
						diffuseTexWidth,
						diffuseTexHeight,

						dev_vertexOut	//VertexOut
					});

					totalNumPrimitives += (numPrimitives * numInstances);

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

//------------------------- SHADERS -----------------------------------------------------------------

static int curPrimitiveBeginId = 0;

__global__ 
void _vertexTransformAndAssembly(PrimitiveDevBufPointers primitive, glm::mat4 MVP, glm::mat4 MV, glm::mat3 MV_normal, glm::mat3 M_InverseTranspose, glm::vec3 eyePos, int width, int height) 
{
	// vertex id
	const int vid = (blockIdx.x * blockDim.x) + threadIdx.x;
	const int numVertices = primitive.numVertices;
	const int numInstances = primitive.numInstances;

	if (vid < numVertices) 
	{
		// Attributes
		const glm::vec3 inPos = primitive.dev_position[vid];
		const glm::vec3 inNormal = primitive.dev_normal[vid];
		
		// This is in NDC Space
		glm::vec4 outPos = MVP * glm::vec4(inPos, 1.f);
		outPos /= outPos.w;

		// Convert to screen Space
		const glm::vec4 screenPos = NDCToScreenSpace(&outPos, width, height);

		// TODO : Change this
		const glm::vec4 instanceOffset(100.f, 0.f, 0.f, 0.f);
		for (int instanceId = 0; instanceId < numInstances; ++instanceId)
		{
			// Output of vertex shader
			primitive.dev_verticesOut[vid * numInstances + instanceId].pos = screenPos + instanceOffset * float(instanceId);
			primitive.dev_verticesOut[vid * numInstances + instanceId].eyePos = glm::vec3(MV * glm::vec4(inPos, 1.f));
			primitive.dev_verticesOut[vid * numInstances + instanceId].eyeNor = MV_normal * inNormal;

			if (primitive.dev_diffuseTex != NULL)
			{
				primitive.dev_verticesOut[vid * numInstances + instanceId].texcoord0 = primitive.dev_texcoord0[vid];
			}
			
		}
	}
}

__global__ 
void _primitiveAssembly(int numIndices, int curPrimitiveBeginId, Primitive* dev_primitives, PrimitiveDevBufPointers primitive) {

	// index id
	const int iid = (blockIdx.x * blockDim.x) + threadIdx.x;
	const int numInstances = primitive.numInstances;

	if (iid < numIndices) {

		// This is primitive assembly for triangles
		
		for (int instanceId = 0; instanceId < numInstances; ++instanceId)
		{
			int pid = iid / (int)primitive.primitiveType;;	// id for cur primitives vector
			if (primitive.primitiveMode == TINYGLTF_MODE_TRIANGLES) 
			{
				const int devBufferIndexPos = primitive.dev_indices[iid] * numInstances + instanceId;
				const int primitiveIndexPos = (pid + curPrimitiveBeginId) * numInstances + instanceId;
				
				dev_primitives[primitiveIndexPos].v[iid % (int)primitive.primitiveType] = primitive.dev_verticesOut[devBufferIndexPos];
				dev_primitives[primitiveIndexPos].primitiveType = primitive.primitiveType;
				dev_primitives[primitiveIndexPos].instanceId = instanceId;

				if (primitive.dev_diffuseTex != NULL)
				{
					dev_primitives[primitiveIndexPos].dev_diffuseTex = primitive.dev_diffuseTex;
					dev_primitives[primitiveIndexPos].diffuseTexWidth = primitive.diffuseTexWidth;
					dev_primitives[primitiveIndexPos].diffuseTexHeight = primitive.diffuseTexHeight;
				}
				else
				{
					dev_primitives[primitiveIndexPos].dev_diffuseTex = NULL;
					dev_primitives[primitiveIndexPos].diffuseTexWidth = 0;
					dev_primitives[primitiveIndexPos].diffuseTexHeight = 0;
				}
			}

		}
	}
}

__global__
void _rasterizePrimitive(int width, int height, int totalNumPrimitives, Primitive* dev_primitives, Fragment* dev_fragmentBuffer, float* dev_depth, unsigned int* mutexLock) 
{
	const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	const int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	const int primitiveId = x + (y * width);

	if (primitiveId < totalNumPrimitives)
	{
		Primitive primitive = dev_primitives[primitiveId];

		if (primitive.primitiveType == Triangle) 
		{
			// Vertices in screen Space
			VertexOut v1 = primitive.v[0];
			VertexOut v2 = primitive.v[1];
			VertexOut v3 = primitive.v[2];

			glm::vec3 triangle[3];
			triangle[0] = glm::vec3(v1.pos);
			triangle[1] = glm::vec3(v2.pos);
			triangle[2] = glm::vec3(v3.pos);

			TextureData* dev_diffuseTex = primitive.dev_diffuseTex;
			int textureWidth = primitive.diffuseTexWidth;
			int textureHeight = primitive.diffuseTexHeight;

			const AABB bounds = getAABBForTriangle(triangle);

			// Clamp TO Screen
			float minX = glm::clamp(bounds.min[0], 0.f, width - 1.f);
			float maxX = glm::clamp(bounds.max[0], 0.f, width - 1.f);
			float minY = glm::clamp(bounds.min[1], 0.f, height - 1.f);
			float maxY = glm::clamp(bounds.max[1], 0.f, height - 1.f);

			for (int row = minY; row <= maxY; ++row)
			{
				for (int col = minX; col <= maxX; ++col)
				{
					const int pixelIndex = col + row * width;
					const glm::vec2 currPos(col, row);

					// Calculate BaryCentric coordinates
					const glm::vec3 baryCoord = calculateBarycentricCoordinate(triangle, currPos);

					// Check if point is inside triangle
					const bool isInside = isBarycentricCoordInBounds(baryCoord);

					if (isInside)
					{
						// Get the interop depth
						const float currDepth = -getZAtCoordinate(baryCoord, triangle);

						bool isSet;
						do {
							isSet = (atomicCAS(&mutexLock[pixelIndex], 0, 1) == 0);
							if (isSet) 
							{
								if (currDepth < dev_depth[pixelIndex])
								{
									dev_fragmentBuffer[pixelIndex].eyeNor = (baryCoord.x * v1.eyeNor) + (baryCoord.y * v2.eyeNor) + (baryCoord.z * v3.eyeNor);
									dev_fragmentBuffer[pixelIndex].eyePos = (baryCoord.x * v1.eyePos) + (baryCoord.y * v2.eyePos) + (baryCoord.z * v3.eyePos);
									
									if (dev_diffuseTex != NULL)
									{
										glm::vec2 textureCoord = (baryCoord.x * v1.texcoord0) + (baryCoord.y * v2.texcoord0) + (baryCoord.z * v3.texcoord0);
										textureCoord = glm::vec2(textureCoord.x * textureWidth, textureCoord.y * textureHeight);

										textureCoord = glm::clamp(textureCoord, glm::vec2(0.f), glm::vec2(textureWidth - 1, textureHeight - 1));

										// Apparently there are 3 bytes per pixel based on the texture array size and texture size.
										const int startPixelIndex = int(textureCoord.x + textureCoord.y * textureWidth) * 3;
										float r = dev_diffuseTex[startPixelIndex];
										float g = dev_diffuseTex[startPixelIndex + 1];
										float b = dev_diffuseTex[startPixelIndex + 2];
										dev_fragmentBuffer[pixelIndex].color = glm::vec3(r, g, b) / 255.f;
									}
									else
									{
										dev_fragmentBuffer[pixelIndex].color = glm::vec3(1.0f);// (baryCoord.x * v1.color) + (baryCoord.y * v2.color) + (baryCoord.z * v3.color);
									}
									
									dev_depth[pixelIndex] = currDepth;
								}
							}
							if (isSet) 
							{
								mutexLock[pixelIndex] = 0;
							}
						} while (!isSet);

					}
				}
			}
		}
	}
}

//#define MAT_PLAIN
#define MAT_LAMBERT
//#define MAT_BLINN_PHONG

#define BLINN_PHONEXP 64.f
#define AMBIENT_LIGHT 0.2f

/** 
* Writes fragment colors to the framebuffer
*/
__global__
void render(int w, int h, Fragment *fragmentBuffer, glm::vec3 *framebuffer) 
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * w);

	if (x < w && y < h) 
	{

		// No Shading
#ifdef MAT_PLAIN
		framebuffer[index] = fragmentBuffer[index].color;
#endif

		// Lamberts Material
#ifdef MAT_LAMBERT

		float lambertsTerm = glm::abs(glm::dot(fragmentBuffer[index].eyeNor, glm::vec3(0, 0, 1))) + AMBIENT_LIGHT;
		lambertsTerm = glm::clamp(lambertsTerm, 0.f, 1.f);
		framebuffer[index] = fragmentBuffer[index].color * lambertsTerm;
#endif

#ifdef MAT_BLINN_PHONG

		const glm::vec3 fsNorm = fragmentBuffer[index].eyeNor;
		const glm::vec3 fsCamera = glm::vec3(0, 0, 1.f);
		const glm::vec3 hVec = (fsNorm + fsCamera) / 2.f;

		const float specular = glm::max(glm::pow(glm::dot(glm::normalize(hVec), glm::normalize(fsNorm)), BLINN_PHONEXP), 0.f);
		const float lambertsTerm = glm::clamp(glm::abs(glm::dot(fsNorm, fsCamera)) + AMBIENT_LIGHT, 0.f, 1.f);

		framebuffer[index] = fragmentBuffer[index].color * (lambertsTerm + specular);
#endif

	}
}

#define SSAA_UNIFORM_GRID

/** 
* Performs Anti-Aliasing
*/
__global__
void _SSAA(int downWidth, int downHeight, int originalWidth, int originalHeight, glm::vec3 *inputFrameBuffer, glm::vec3 *aaFrameBuffer) 
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * downWidth);

	if (x < downWidth && y < downHeight) 
	{
#ifdef SSAA_UNIFORM_GRID

		const int upscaledIndexX = x * SSAA_LEVEL;
		const int upscaledIndexY = y * SSAA_LEVEL;

		glm::vec3 averageColor(0.f);
		int numColors = 0;

		for (int i = 0; i < SSAA_LEVEL; ++i)
		{
			for (int j = 0; j < SSAA_LEVEL; ++j)
			{
				const int newIndex = (upscaledIndexX + i) + ((upscaledIndexY + i) * originalWidth);
				averageColor += inputFrameBuffer[newIndex];
				numColors++;
			}
		}
		aaFrameBuffer[index] = (averageColor / float(numColors));

#endif // SSAA_UNIFORM_GRID

	}
}


/**
 * Perform rasterization.
 */
void rasterize(uchar4 *pbo, const glm::mat4 & MVP, const glm::mat4 & MV, const glm::mat3& MV_normal, const glm::mat3& M_inverseTranspose, const glm::vec3& eyePos) {
    
	int sideLength2d = 8;
    dim3 blockSize2d(sideLength2d, sideLength2d);

    dim3 blockCount2d((width  - 1) / blockSize2d.x + 1,
		(height - 1) / blockSize2d.y + 1);

	dim3 blockAACount2d((originalWidth  - 1) / blockSize2d.x + 1,
		(originalHeight - 1) / blockSize2d.y + 1);


	dim3 numThreadsPerBlock(128);
	dim3 blocksPrimitives((totalNumPrimitives + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);


	// Execute your rasterization pipeline here
	// (See README for rasterization pipeline outline.)

	// Vertex Process & primitive assembly
	{
		curPrimitiveBeginId = 0;
		

		auto it = mesh2PrimitivesMap.begin();
		auto itEnd = mesh2PrimitivesMap.end();

		for (; it != itEnd; ++it) 
		{
			auto p = (it->second).begin();	
			auto pEnd = (it->second).end();

			for (; p != pEnd; ++p) 
			{
				dim3 numBlocksForVertices((p->numVertices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
				dim3 numBlocksForIndices((p->numIndices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);

				// 1. Vertex Assembly and Shader
				_vertexTransformAndAssembly <<< numBlocksForVertices, numThreadsPerBlock >> >(*p, MVP, MV, MV_normal, M_inverseTranspose, eyePos, width, height);
				checkCUDAError("Vertex Processing");

				cudaDeviceSynchronize();

				// 2. Primitive Assembly
				_primitiveAssembly << < numBlocksForIndices, numThreadsPerBlock >> >
					(p->numIndices, 
					curPrimitiveBeginId, 
					dev_primitives, 
					*p);
				checkCUDAError("Primitive Assembly");

				curPrimitiveBeginId += (p->numPrimitives * p->numInstances);
			}
		}

		checkCUDAError("Vertex Processing and Primitive Assembly");
	}
	
	cudaMemset(dev_fragmentBuffer, 0, width * height * sizeof(Fragment));

	// 3. Depth Check
	initDepth << <blockCount2d, blockSize2d >> >(width, height, dev_depth);
	
	// 4. Rasterize - Call per primitive
#ifdef AA_MULTI_SAMPLE
	_rasterizePrimitive << <blockCount2d, blockSize2d >> > (width, height, totalNumPrimitives, dev_primitives, dev_fragmentBuffer, dev_depth, dev_mutex);
#else
	_rasterizePrimitive << <blocksPrimitives, numThreadsPerBlock >> > (width, height, totalNumPrimitives, dev_primitives, dev_fragmentBuffer, dev_depth, dev_mutex);
#endif
	checkCUDAError("Rasterizer");

    // Copy fragmentBuffer colors into framebuffer
	render << <blockCount2d, blockSize2d >> >(width, height, dev_fragmentBuffer, dev_framebuffer);
	checkCUDAError("fragment shader");

	// perform SSAA
	_SSAA <<< blockAACount2d, blockSize2d >>> (originalWidth, originalHeight, width, height, dev_framebuffer, dev_AAFrameBuffer);
	checkCUDAError("SSAA");

    // Copy framebuffer into OpenGL buffer for OpenGL previewing
    sendImageToPBO<<<blockAACount2d, blockSize2d>>>(pbo, originalWidth, originalHeight, dev_AAFrameBuffer);
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

	cudaFree(dev_AAFrameBuffer);
	dev_AAFrameBuffer = NULL;

	cudaFree(dev_depth);
	dev_depth = NULL;

	cudaFree(dev_mutex);
	dev_mutex = NULL;


    checkCUDAError("rasterize Free");
}
