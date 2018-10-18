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
		 int texWidth, texHeight;
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
		VertexAttributeTexcoord texcoord0;
		TextureData* dev_diffuseTex;
		int texWidth, texHeight;
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

static int totalNumPrimitives = 0;
static Primitive *dev_primitives = NULL;
static Fragment *dev_fragmentBuffer = NULL;
static glm::vec3 *dev_framebuffer = NULL;
static glm::vec3 *dev_framebuffer_2 = NULL;

static float * dev_depth = NULL;	// you might need this buffer when doing depth test
static int *dev_mutex = NULL;

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

__forceinline__
__device__
glm::vec3 fetchColor(glm::vec3 *textureData, int x, int y, int w, int h) {
	int pix = y * w + x;
	return textureData[pix];
}

__device__
glm::vec3 colorAt(TextureData* texture, int textureWidth, float u, float v) {
	int flatIndex = u + v * textureWidth;
	float r = (float) texture[flatIndex * 3] / 255.0f;
	float g = (float) texture[flatIndex * 3 + 1] / 255.0f;
	float b = (float) texture[flatIndex * 3 + 2] / 255.0f;
	return glm::vec3(r, g, b);
}

/** 
* Writes fragment colors to the framebuffer
*/
__global__
void render(int w, int h, Fragment *fragmentBuffer, glm::vec3 *framebuffer) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * w);

    if (x < w && y < h) {
		glm::vec3 lightPos = glm::vec3(1, 1, 1);
		Fragment frag = fragmentBuffer[index];
		int u = frag.texcoord0.x * frag.texWidth;
		int v = frag.texcoord0.y * frag.texHeight;
		glm::vec3 col;
		if (frag.dev_diffuseTex != NULL) {
			col = colorAt(frag.dev_diffuseTex, frag.texWidth, u, v);
		} else {
			col = frag.color;
		}
		glm::vec3 lightDir = (lightPos - frag.eyePos);
		framebuffer[index] = col * glm::max(0.f, glm::dot(frag.eyeNor, lightDir));

		// TODO: add your fragment shader code here

    }
}

/**
 * Called once at the beginning of the program to allocate memory.
 */
void rasterizeInit(int w, int h) {
	cudaError_t stat = cudaDeviceSetLimit(cudaLimitStackSize, 8192);
	checkCUDAError("set stack limit");
    width = w;
    height = h;
	cudaFree(dev_fragmentBuffer);
	cudaMalloc(&dev_fragmentBuffer, width * height * sizeof(Fragment));
	cudaMemset(dev_fragmentBuffer, 0, width * height * sizeof(Fragment));

    cudaFree(dev_framebuffer);
    cudaMalloc(&dev_framebuffer,   width * height * sizeof(glm::vec3));
    cudaMemset(dev_framebuffer, 0, width * height * sizeof(glm::vec3));

	cudaFree(dev_framebuffer_2);
	cudaMalloc(&dev_framebuffer_2, width * height * sizeof(glm::vec3));
	cudaMemset(dev_framebuffer_2, 0, width * height * sizeof(glm::vec3));
    
	cudaFree(dev_depth);
	cudaMalloc(&dev_depth, width * height * sizeof(float));

	cudaFree(dev_mutex);
	cudaMalloc(&dev_mutex, width * height * sizeof(int));

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
		glm::vec4 mPos(primitive.dev_position[vid], 1.f);
		// clip space
		glm::vec4 camPos = MVP * mPos;
		// NDC
		glm::vec4 ndcPos = camPos / camPos.w;
		// viewport
		float x = (ndcPos.x + 1.f) * ((float) width) * 0.5f;
		float y = (1.f - ndcPos.y) * ((float) height) * 0.5f;
		float z = -ndcPos.z;

		// TODO: Apply vertex assembly here
		// Assemble all attribute arraies into the primitive array
		primitive.dev_verticesOut[vid].pos = { x, y, z, 1.f };
		primitive.dev_verticesOut[vid].eyePos = glm::vec3(MV * mPos);
		primitive.dev_verticesOut[vid].eyeNor = glm::normalize(MV_normal * primitive.dev_normal[vid]);
		primitive.dev_verticesOut[vid].texcoord0 = primitive.dev_texcoord0[vid];
		primitive.dev_verticesOut[vid].dev_diffuseTex = primitive.dev_diffuseTex;
		primitive.dev_verticesOut[vid].texHeight = primitive.diffuseTexHeight;
		primitive.dev_verticesOut[vid].texWidth = primitive.diffuseTexWidth;
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
			dev_primitives[pid + curPrimitiveBeginId].v[iid % (int)primitive.primitiveType]
				= primitive.dev_verticesOut[primitive.dev_indices[iid]];
		}


		// TODO: other primitive types (point, line)
	}
	
}

__global__
void _rasterize(int numPrimitives, Primitive *dev_primitives, Fragment *dev_fragmentBuffer, int width, int height, float *dev_depthBuffer, int *mutexes) {
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (idx < numPrimitives) {
		Primitive prim = dev_primitives[idx];
		VertexOut vs[3];
		vs[0] = prim.v[0];
		vs[1] = prim.v[1];
		vs[2] = prim.v[2];
		glm::vec3 pos[3];
		pos[0] = glm::vec3(vs[0].pos);
		pos[1] = glm::vec3(vs[1].pos);
		pos[2] = glm::vec3(vs[2].pos);

		// Get bounds of this primitive
		glm::vec2 min, max;
		AABB bounds = getAABBForTriangle(pos);
		min.x = glm::clamp(bounds.min.x, 0.f, (float) (width - 1));
		min.y = glm::clamp(bounds.min.y, 0.f, (float) (height - 1));
		max.x = glm::clamp(bounds.max.x, 0.f, (float) (width - 1));
		max.y = glm::clamp(bounds.max.y, 0.f, (float) (height - 1));

		// Generate fragments for each pixel this primitive overlaps
		for (int x = min.x; x <= max.x; ++x) {
			for (int y = min.y; y <= max.y; ++y) {
				glm::vec3 bary = calculateBarycentricCoordinate(pos, { x, y });
				if (isBarycentricCoordInBounds(bary)) {
					int pixIdx = x + width * y;
					float depth = getZAtCoordinate(bary, pos);
					glm::vec3 nor = bary.x * vs[0].eyeNor +
									bary.y * vs[1].eyeNor +
									bary.z * vs[2].eyeNor;
					
					Fragment frag;
					frag.color = glm::vec3(.95f, .95, .15);
					frag.eyeNor = nor;


					glm::vec2 cord = bary.x * vs[0].texcoord0 / vs[0].eyePos.z +
									 bary.y * vs[1].texcoord0 / vs[1].eyePos.z +
									 bary.z * vs[2].texcoord0 / vs[2].eyePos.z;
					
					float z = bary.x * (1.f / vs[0].eyePos.z) +
							  bary.y * (1.f / vs[1].eyePos.z) +
							  bary.z * (1.f / vs[2].eyePos.z);

					frag.texcoord0 = cord / z;

					frag.texHeight = vs[0].texHeight;
					frag.texWidth = vs[0].texWidth;
					frag.dev_diffuseTex = vs[0].dev_diffuseTex;

					int *mutex = &mutexes[pixIdx];
					bool isSet;
					do {
						isSet = (atomicCAS(mutex, 0, 1) == 0);
						if (isSet) {
							if (depth < dev_depthBuffer[pixIdx]) {
								dev_depthBuffer[pixIdx] = depth;
								dev_fragmentBuffer[pixIdx] = frag;
							}
						}
						if (isSet) {
							mutexes[pixIdx] = 0;
						}
					} while (!isSet);

				}
			}
		}
	}
}

__forceinline__
__device__
float rgb2luma(glm::vec3 rgb) {
	return glm::dot(rgb, glm::vec3(0.299, 0.587, 0.114));
}

__forceinline__
__device__
int flatIdx(int w, int h, glm::vec2 pos) {
	pos.x = w - pos.x;
	pos = glm::clamp(pos, glm::vec2(0, 0), glm::vec2(w - 1, h - 1));
	return pos.x + (pos.y * w);
}

// bilinear filtering
__forceinline__
__device__
float getAlpha(float y, float py, float qy) {
	return (y - py) / (qy - py);
}

__forceinline__
__device__
glm::vec3 slerp(float alpha, glm::vec3 az, glm::vec3 bz) {
	return glm::vec3((1 - alpha) * az.r + alpha * bz.r,
		(1 - alpha) * az.g + alpha * bz.g,
					 (1 - alpha) * az.b + alpha * bz.b);
}

__forceinline__
__device__
float fract(float t) {
	return t - glm::floor(t);
}

__forceinline__
__device__
glm::vec3 textureFetch(glm::vec3 *t, glm::vec2 pix, int w, int h) {
	pix.x = w - pix.x;
	pix = glm::clamp(pix, glm::vec2(0.f, 0.f), glm::vec2(w - 1, h - 1));
	glm::vec3 f = slerp(getAlpha(pix.x, glm::ceil(pix.x), glm::floor(pix.y)),
						fetchColor(t, glm::ceil(pix.x), glm::ceil(pix.y), w, h),
						fetchColor(t, glm::floor(pix.x), glm::ceil(pix.y), w, h));
	glm::vec3 s = slerp(getAlpha(pix.x, glm::ceil(pix.x), glm::floor(pix.y)),
						fetchColor(t, glm::ceil(pix.x), glm::floor(pix.y), w, h),
						fetchColor(t, glm::floor(pix.x), glm::floor(pix.y), w, h));
	return slerp(getAlpha(pix.y, glm::ceil(pix.y), glm::floor(pix.y)), f, s);
}

__forceinline__
__device__ int pow2(int e) {
	int r = 1;
	for (int i = 0; i < e; ++i) {
		r *= 2;
	}
	return r;
}

__forceinline__
__device__
float fxaaQualityStep(int i) {
	return i < 5 ? 2.f : pow2(i - 3);
}

#define EDGE_THRESHOLD_MIN 0.0312
#define EDGE_THRESHOLD_MAX 0.125
#define FXAA_ITERATIONS    12
#define SUBPIXEL_QUALITY   0.75
#define FXAA_REDUCE_MIN   1.0 / 128.0
#define FXAA_REDUCE_MUL   1.0 / 8.0
#define FXAA_SPAN_MAX 8.0
__global__
void _fxaa_post(int w, int h, glm::vec3 *i_framebuffer, glm::vec3 *o_framebuffer) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int idx = (w - x) + (y * w);
	if (x < w && y < h) {
		glm::vec3 rgbN = i_framebuffer[flatIdx(w, h, { x, y - 1 })];
		glm::vec3 rgbW = i_framebuffer[flatIdx(w, h, { x - 1, y })];
		glm::vec3 rgbE = i_framebuffer[flatIdx(w, h, { x + 1, y })];
		glm::vec3 rgbS = i_framebuffer[flatIdx(w, h, { x, y + 1 })];

		glm::vec3 rgbM = i_framebuffer[flatIdx(w, h, { x, y })];

		float lumaN = rgb2luma(rgbN);
		float lumaW = rgb2luma(rgbW);
		float lumaE = rgb2luma(rgbE);
		float lumaS = rgb2luma(rgbS);
		float lumaM = rgb2luma(rgbM);

		float rangeMin = glm::min(lumaM, glm::min(glm::min(lumaN, lumaW), glm::min(lumaS, lumaE)));
		float rangeMax = glm::max(lumaM, glm::max(glm::max(lumaN, lumaW), glm::max(lumaS, lumaE)));

		// Check local contrast to avoid processing non edges
		float range = rangeMax - rangeMin;
		if (range < glm::max(FXAA_EDGE_THRESHOLD_MIN, rangeMax * FXAA_EDGE_THRESHOLD)) {
			o_framebuffer[idx] = i_framebuffer[idx];
			return;
		}

		#if FXAA_DEBUG_PASSTHROUGH
		// Set edges to red
		o_framebuffer[idx] = COLOR_RED;
		return;
		#endif

		float lumaL = (lumaN + lumaW + lumaE + lumaS) * 0.25f;
		float rangeL = glm::abs(lumaL - lumaM);
		float blendL = glm::max(0.f, (rangeL / range) - FXAA_SUBPIX_TRIM) * FXAA_SUBPIX_TRIM_SCALE;
		blendL = glm::min(FXAA_SUBPIX_CAP, blendL);
		glm::vec3 rgbL = rgbN + rgbW + rgbM + rgbE + rgbS;
		glm::vec3 rgbNW = i_framebuffer[flatIdx(w, h, { x - 1, y - 1 })];
		glm::vec3 rgbNE = i_framebuffer[flatIdx(w, h, { x + 1, y - 1 })];
		glm::vec3 rgbSW = i_framebuffer[flatIdx(w, h, { x - 1, y + 1 })];
		glm::vec3 rgbSE = i_framebuffer[flatIdx(w, h, { x + 1, y + 1 })];
		float lumaNW = rgb2luma(rgbNW);
		float lumaNE = rgb2luma(rgbNE);
		float lumaSW = rgb2luma(rgbSW);
		float lumaSE = rgb2luma(rgbSE);
		rgbL += (rgbNW + rgbNE + rgbSW + rgbSE);
		rgbL *= (1.f / 9.f);

		float edgeVert =
			glm::abs((0.25f * lumaNW) + (-0.5f * lumaN) + (0.25f * lumaNE)) +
			glm::abs((0.50f * lumaW ) + (-1.0f * lumaM) + (0.50f * lumaE )) +
			glm::abs((0.25f * lumaSW) + (-0.5f * lumaS) + (0.25f * lumaSE));

		float edgeHorz =
			glm::abs((0.25f * lumaNW) + (-0.5f * lumaW) + (0.25f * lumaSW)) +
			glm::abs((0.50f * lumaN ) + (-1.0f * lumaM) + (0.50f * lumaS )) +
			glm::abs((0.25f * lumaNE) + (-0.5f * lumaE) + (0.25f * lumaSE));

		bool isHor = edgeHorz >= edgeVert;

		#if FXAA_DEBUG_HORZVERT
		// Set horizontal edges to yellow, vertical edges to blue
		o_framebuffer[idx] = isHor ? COLOR_YELLOW : COLOR_BLUE;
		return;
		#endif

		// Select highest contrast pixel pair orthogonal to the edge

		// If horizontal edge, check pair of M with S and N
		// If vertical edge, check pair of M with W and E
		float luma1 = isHor ? lumaS : lumaE;
		float luma2 = isHor ? lumaN : lumaW;

		float grad1 = luma1 - lumaM;
		float grad2 = luma2 - lumaM;

		bool is1Steepest = glm::abs(grad1) >= glm::abs(grad2);
		float gradScaled = 0.25f * glm::max(glm::abs(grad1), glm::abs(grad2));

		float stepLen = 1.f;
		float lumaLocalAvg = 0.f;
		
		if (is1Steepest) {
			lumaLocalAvg = 0.5f * (luma1 + lumaM);
		} else {
			stepLen = -stepLen;
			lumaLocalAvg = 0.5f * (luma2 + lumaM);
		}

		glm::vec2 currUV = { x, y };
		if (isHor) {
			currUV.y += stepLen * 0.5f;
		} else {
			currUV.x += stepLen * 0.5f;
		}

		#if FXAA_DEBUG_PAIR
		// Set pixel up or left to BLUE
		// Set pixel down or right to GREEN
		glm::vec2 secondCoord = { x + (isHor ? stepLen : 0), y + (isHor ? 0 : stepLen) };
		int secondIdx = flatIdx(w, h, secondCoord);
		if (secondCoord.x < x || secondCoord.y < y) {
			o_framebuffer[idx] = COLOR_GREEN;
		} else {
			o_framebuffer[idx] = COLOR_BLUE;
		}
		return;
		#endif

		// Search for end of edge in both - and + directions
		glm::vec2 offset = isHor ? glm::vec2(1.f, 0.f) : glm::vec2(0.f, 1.f);
		glm::vec2 uv1 = currUV;
		glm::vec2 uv2 = currUV;
		float lumaEnd1, lumaEnd2;

		bool reached1 = false;
		bool reached2 = false;
		bool reachedBoth = reached1 && reached2;

		for (int i = 0; i < FXAA_SEARCH_STEPS; ++i) {
			if (!reached1) {
				uv1 -= offset * fxaaQualityStep(i);
				lumaEnd1 = rgb2luma(textureFetch(i_framebuffer, uv1, w, h));
				//lumaEnd1 -= lumaLocalAvg;
			}

			if (!reached2) {
				uv2 += offset * fxaaQualityStep(i);
				lumaEnd2 = rgb2luma(textureFetch(i_framebuffer, uv2, w, h));
				//lumaEnd2 -= lumaLocalAvg;
			}

			reached1 = (glm::abs(lumaEnd1 - lumaN) >= gradScaled);
			reached2 = (glm::abs(lumaEnd2 - lumaN) >= gradScaled);
			reachedBoth = (reached1 && reached2);
			
			if (reachedBoth) { break; }
		}

		// Compute subpixel offset based on distance to end of edge
		float dist1 = glm::abs(isHor ? (x - uv1.x) : (y - uv1.y));
		float dist2 = glm::abs(isHor ? (uv2.x - x) : (uv2.y - y));
		bool isDir1 = dist1 < dist2;
		float distFinal = glm::min(dist1, dist2);
		float edgeLength = dist1 + dist2;

		#if FXAA_DEBUG_EDGEPOS
		float alpha = distFinal / 12.f;
		o_framebuffer[idx] = alpha * COLOR_YELLOW + (1 - alpha) * COLOR_GREEN;
		return;
		#endif

		float pixelOffset = -distFinal / edgeLength + 0.5;
		//printf("pixelOffset: %f\n", pixelOffset);

		bool isLumaCenterSmaller = lumaM < lumaLocalAvg;

		bool correctVariation = ((isDir1 ? lumaEnd1 : lumaEnd2) < 0.0) != isLumaCenterSmaller;

		pixelOffset = correctVariation ? pixelOffset : 0.f;
		glm::vec2 finalUV = isHor ? glm::vec2(x, y + pixelOffset) : glm::vec2(x + pixelOffset, y);

		o_framebuffer[idx] = textureFetch(i_framebuffer, finalUV, w, h);

		/*
		float lumaC = rgb2luma(i_framebuffer[idx]);

		float lumaD = rgb2luma(i_framebuffer[flatIdx(w, h, { x, y + 1 })]);
		float lumaU = rgb2luma(i_framebuffer[flatIdx(w, h, { x, y - 1 })]);
		float lumaL = rgb2luma(i_framebuffer[flatIdx(w, h, { x - 1, y })]);
		float lumaR = rgb2luma(i_framebuffer[flatIdx(w, h, { x + 1, y })]);

		float lumaMin = glm::min(lumaC, glm::min(glm::min(lumaD, lumaU), glm::min(lumaL, lumaR)));
		float lumaMax = glm::max(lumaC, glm::max(glm::max(lumaD, lumaU), glm::max(lumaL, lumaR)));

		float lumaDelta = lumaMax - lumaMin;

		if (glm::isnan(lumaDelta)) {
			lumaDelta = 0.f;
		}

		if (lumaDelta < glm::max(EDGE_THRESHOLD_MIN, lumaMax * EDGE_THRESHOLD_MAX)) {
			o_framebuffer[idx] = i_framebuffer[idx];
			return;
		}

		float lumaDL = rgb2luma(i_framebuffer[flatIdx(w, h, { x - 1, y + 1 })]);
		float lumaUL = rgb2luma(i_framebuffer[flatIdx(w, h, { x - 1, y - 1 })]);
		float lumaDR = rgb2luma(i_framebuffer[flatIdx(w, h, { x + 1, y + 1 })]);
		float lumaUR = rgb2luma(i_framebuffer[flatIdx(w, h, { x + 1, y - 1 })]);

		float lumaDU = lumaD + lumaU;
		float lumaLR = lumaL + lumaR;
		float lumaLCorn = lumaDL + lumaUL;
		float lumaDCorn = lumaDL + lumaDR;
		float lumaRCorn = lumaDR + lumaUR;
		float lumaUCorn = lumaUL + lumaUR;

		float edgeHor = glm::abs(-2.f * lumaL + lumaLCorn) +
						glm::abs(-2.f * lumaC + lumaDU) * 2.f +
						glm::abs(-2.f * lumaR + lumaRCorn);

		float edgeVer = glm::abs(-2.f * lumaU + lumaUCorn) +
						glm::abs(-2.f * lumaC + lumaLR) * 2.f +
						glm::abs(-2.f * lumaD + lumaDCorn);

		bool isHor = (edgeHor >= edgeVer);

		float luma1 = isHor ? lumaD : lumaL;
		float luma2 = isHor ? lumaU : lumaR;

		float grad1 = luma1 - lumaC;
		float grad2 = luma2 - lumaC;

		bool is1Steepest = glm::abs(grad1) >= glm::abs(grad2);

		float gradScale = 0.25f * glm::max(glm::abs(grad1), glm::abs(grad2));

		float stepLen = 1.f;
		float lumaLocalAvg = 0.f;
		if (is1Steepest) {
			stepLen = -stepLen;
			lumaLocalAvg = 0.5f * (luma1 + lumaC);
		} else {
			lumaLocalAvg = 0.5f * (luma2 + lumaC);
		}

		glm::vec2 currPos(x, y);
		if (isHor) {
			currPos.y += stepLen * 0.5f;
		} else {
			currPos.x += stepLen * 0.5f;
		}

		glm::vec2 offset = isHor ? glm::vec2(1.f, 0.f) : glm::vec2(0.f, 1.f);
		glm::vec2 p1 = currPos - offset;
		glm::vec2 p2 = currPos + offset;

		float lumaEnd1 = rgb2luma(textureFetch(i_framebuffer, p1, w, h));
		float lumaEnd2 = rgb2luma(textureFetch(i_framebuffer, p2, w, h));
		lumaEnd1 -= lumaLocalAvg;
		lumaEnd2 -= lumaLocalAvg;

		bool reached1 = glm::abs(lumaEnd1) >= gradScale;
		bool reached2 = glm::abs(lumaEnd2) >= gradScale;
		bool reachedBoth = reached1 && reached2;

		if (!reached1) {
			p1 -= offset;
		}
		if (!reached2) {
			p2 += offset;
		}

		if (!reachedBoth) {
			for (int i = 2; i < FXAA_ITERATIONS; ++i) {
				if (!reached1) {
					lumaEnd1 = rgb2luma(textureFetch(i_framebuffer, p1, w, h));
					lumaEnd1 -= lumaLocalAvg;
				}
				if (!reached2) {
					lumaEnd2 = rgb2luma(textureFetch(i_framebuffer, p2, w, h));
					lumaEnd2 -= lumaLocalAvg;
				}
				reached1 = glm::abs(lumaEnd1) >= gradScale;
				reached2 = glm::abs(lumaEnd2) >= gradScale;
				reachedBoth = reached1 && reached2;
				if (!reached1) {
					p1 -= offset * fxaaQualityStep(i);
				}
				if (!reached2) {
					p2 += offset * fxaaQualityStep(i);
				}
				if (reachedBoth) { break; }
			}
		}

		float dist1 = isHor ? ((float) x - p1.x) : ((float) y - p1.y);
		float dist2 = isHor ? (p2.x - (float) x) : (p2.y - (float) y);

		bool isDir1 = dist1 < dist2;
		float distFinal = glm::min(dist1, dist2);

		float edgeThickness = (dist1 + dist2);
		float pixOffset = -distFinal / edgeThickness + 0.5f;

		bool isLumaCSmaller = lumaC < lumaLocalAvg;
		bool correctVar = ((isDir1 ? lumaEnd1 : lumaEnd2) < 0.f) != isLumaCSmaller;
		float finalOffset = correctVar ? pixOffset : 0.f;

		float lumaAvg = (1.f / 12.f) * (2.f * (lumaDU + lumaLR) + lumaLCorn + lumaRCorn);
		float subPixOffset1 = glm::clamp(glm::abs(lumaAvg - lumaC) / lumaDelta, 0.f, 1.f);
		float subPixOffset2 = (-2.f * subPixOffset1 + 3.f) * subPixOffset1 * subPixOffset1;

		float subPixOffsetFinal = subPixOffset2 * subPixOffset2 * SUBPIXEL_QUALITY;

		finalOffset = glm::max(finalOffset, subPixOffsetFinal);
		glm::vec2 finalPixPos = glm::vec2(x, y);
		if (isHor) {
			finalPixPos.y += finalOffset * stepLen;
		} else {
			finalPixPos.x += finalOffset * stepLen;
		}

		o_framebuffer[idx] = textureFetch(i_framebuffer, finalPixPos, w, h);
		o_framebuffer[idx] = isHor ? glm::vec3(1, 0, 0) : glm::vec3(0, 1, 0);
		*/
		
	}
}

/**
 * Perform rasterization.
 */
void rasterize(uchar4 *pbo, const glm::mat4 & MVP, const glm::mat4 & MV, const glm::mat3 MV_normal) {
    int sideLength2d = 8;
    dim3 blockSize2d(sideLength2d, sideLength2d);
    dim3 blockCount2d((width  - 1) / blockSize2d.x + 1,
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
	
	// TODO: rasterize
	cudaMemset(dev_mutex, 0, sizeof(int));
	dim3 numThreadsPerBlock(128);
	dim3 numBlocksPerPrimitive = (totalNumPrimitives + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x;
	_rasterize << <numBlocksPerPrimitive, numThreadsPerBlock >> > (totalNumPrimitives, dev_primitives, dev_fragmentBuffer, width, height, dev_depth, dev_mutex);

    // Copy depthbuffer colors into framebuffer
	render << <blockCount2d, blockSize2d >> >(width, height, dev_fragmentBuffer, dev_framebuffer);
	checkCUDAError("fragment shader");

	// Do post process effects here:
	// FXAA, SSAO
#if FXAA
	{
		_fxaa_post << < blockCount2d , blockSize2d >> > (width, height, dev_framebuffer, dev_framebuffer_2);
		checkCUDAError("FXAA postprocess");

		std::swap(dev_framebuffer, dev_framebuffer_2);
	}
#endif

#if SSAO
	_ssao_post << <blockCount2d, blockSize2d >> > (width, height, dev_framebuffer, dev_framebuffer_2);
#endif
	

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

	cudaFree(dev_framebuffer_2);
	dev_framebuffer_2 = NULL;

	cudaFree(dev_depth);
	dev_depth = NULL;

	cudaFree(dev_mutex);
	dev_mutex = NULL;

    checkCUDAError("rasterize Free");
}
