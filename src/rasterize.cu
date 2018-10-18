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
#include <chrono>
#include <cuda_runtime.h>
#include <thrust/random.h>
#include <util/checkCUDAError.h>
#include <util/tiny_gltf_loader.h>
#include "rasterizeTools.h"
#include "rasterize.h"
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>

#define MAX_BLOCK_SIZE 128
#define LAMBERT 1
#define BLINN_PHONG 1
#define RENDER_LINE 0
#define RENDER_POINT 0
#define PERSP_CORRECT_UV 1
#define BILINEAR_TEX_FILTERING 1

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
		 glm::vec3 col;
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
        float depth;
		glm::vec3 eyePos;	// eye space position used for shading
		glm::vec3 eyeNor;
		VertexAttributeTexcoord texcoord0;
		TextureData* dev_diffuseTex;
        int texHeight, texWidth;
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

static int * dev_depth = NULL;	// you might need this buffer when doing depth test
static int * dev_fragMutex = NULL;	// you might need this buffer when doing depth test

__host__ __device__ static
glm::vec2 getBCInterpolatePersp(const glm::vec3& bcCoord, const Primitive& primitive)
{

    float f0 = primitive.v[0].eyePos.z;
    float f1 = primitive.v[1].eyePos.z;
    float f2 = primitive.v[2].eyePos.z;

    glm::vec2 p0 = primitive.v[0].texcoord0;
    glm::vec2 p1 = primitive.v[1].texcoord0;
    glm::vec2 p2 = primitive.v[2].texcoord0;

    glm::vec2 t0 = bcCoord.x * p0 / f0
        + bcCoord.y * p1 / f1
        + bcCoord.z * p2 / f2;
    float t1 = bcCoord.x / f0
        + bcCoord.y / f1
        + bcCoord.z / f2;
    return t0 / t1;
}

__host__ __device__
void passPoint(Fragment* fragmentBuffer, glm::vec3 p, glm::vec3 color, int width, int height)
{
    int x = glm::clamp(p.x, 0.f, (float)(width - 1));
    int y = glm::clamp(p.y, 0.f, (float)(height - 1));
    fragmentBuffer[x + y * width].color = color;
}

// Thanks to https://github.com/ssloy/tinyrenderer/
__host__ __device__
void passLine(Fragment* fragmentBuffer, glm::vec3 p0, glm::vec3 p1, glm::vec3 color, int width, int height)
{
    glm::vec2 pos0(glm::clamp(p0.x, 0.f, float(width - 1)),
        glm::clamp(p0.y, 0.f, float(height - 1)));

    glm::vec2 pos1(glm::clamp(p1.x, 0.f, float(width - 1)),
        glm::clamp(p1.y, 0.f, float(height - 1)));

    float length = glm::length(pos0 - pos1);
    for (float t = 0.f; t < 1.f; t += 1.f / length)
    {
        int x = pos0.x * (1.f - t) + pos1.x * t;
        int y = pos0.y * (1.f - t) + pos1.y * t;
        fragmentBuffer[x + y * width].color = color;
    }
}

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
__global__
void render(int w, int h, Fragment* fragmentBuffer, glm::vec3* framebuffer) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * w);

    if (x >= w || y >= h) {
        return;
    }


#if RENDER_LINE || RENDER_POINT
    framebuffer[index] = fragmentBuffer[index].color;
    return;
#endif

    Fragment fragment = fragmentBuffer[index];
    glm::vec3 diffuse;
    glm::vec3 color_out(0.0f, 0.0f, 0.0f);

    

    if (fragment.dev_diffuseTex != NULL) {
        // texture
        TextureData* textureData = fragment.dev_diffuseTex;
        float texW = fragment.texWidth;
        float texH = fragment.texHeight;

#if BILINEAR_TEX_FILTERING
        // https://www.scratchapixel.com/lessons/mathematics-physics-for-computer-graphics/interpolation/bilinear-filtering
        
        float uFlt = fragment.texcoord0.x * texW;
        float vFlt = fragment.texcoord0.y * texH;
        int uInt = (int)uFlt;
        int vInt = (int)vFlt;
        float uOffset = uFlt - (float)uInt;
        float vOffset = vFlt - (float)vInt;
        int uIntOne = (uInt + 1 < texW - 1) ? uInt + 1 : texW - 1;
        int vIntOne = (vInt + 1 < texH - 1) ? vInt + 1 : texH - 1;
        int uvIdx; 

        // Get 4 samples
        // First fix to vInt, then fix to vIntOne
        uvIdx = (uInt + vInt * texW);
        glm::vec3 col0 = glm::vec3(
            textureData[3 * uvIdx] / 255.f,
            textureData[3 * uvIdx + 1] / 255.f,
            textureData[3 * uvIdx + 2] / 255.f);
        
        uvIdx = (uIntOne + vInt * texW);
        glm::vec3 col1 = glm::vec3(
            textureData[3 * uvIdx] / 255.f,
            textureData[3 * uvIdx + 1] / 255.f,
            textureData[3 * uvIdx + 2] / 255.f);
        
        uvIdx = (uInt + vIntOne * texW);
        glm::vec3 col2 = glm::vec3(
            textureData[3 * uvIdx] / 255.f,
            textureData[3 * uvIdx + 1] / 255.f,
            textureData[3 * uvIdx + 2] / 255.f);
        
        uvIdx = (uIntOne + vIntOne * texW);
        glm::vec3 col3 = glm::vec3(
            textureData[3 * uvIdx] / 255.f,
            textureData[3 * uvIdx + 1] / 255.f,
            textureData[3 * uvIdx + 2] / 255.f);
        
        // mix u then mix v
        glm::vec3 col01 = glm::mix(col0, col1, uOffset);
        glm::vec3 col23 = glm::mix(col2, col3, uOffset);
        diffuse = glm::mix(col01, col23, vOffset);
#else
        int u = fragment.texcoord0.x * texW;
        int v = fragment.texcoord0.y * texH;
        int uvIdx = (u + v * texW);
        diffuse = glm::vec3(textureData[3 * uvIdx] / 255.f,
            textureData[3 * uvIdx + 1] / 255.f,
            textureData[3 * uvIdx + 2] / 255.f);
#endif
        color_out += diffuse;
    }
    else {
        diffuse += fragment.color ;
    }

    color_out = diffuse;

    glm::vec3 lightPos = glm::vec3(0.f, 10.f, 10.f);
    glm::vec3 lightDir = glm::normalize(lightPos - fragment.eyePos);

#if LAMBERT
    float lambertTerm = glm::dot(fragment.eyeNor, lightDir);
    color_out *= glm::clamp(lambertTerm, 0.2f, 1.0f);
#endif

#if BLINN_PHONG
    
    glm::vec3 halfwayDir = glm::normalize(lightDir - fragment.eyePos);
    float specTerm = pow(glm::max(glm::dot(fragment.eyeNor, halfwayDir), 0.f), 10.f);
    glm::vec3 specCol = glm::vec3(0.6f, 0.6f, 0.6f);
    color_out += specTerm * specCol;
#endif
    
    framebuffer[index] = color_out;
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

    cudaFree(dev_fragMutex);
    cudaMalloc(&dev_fragMutex, width * height * sizeof(int));

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

				auto res = mesh2PrimitivesMap.insert(std::pair<std::string,
                    std::vector<PrimitiveDevBufPointers>>(mesh.name, std::vector<PrimitiveDevBufPointers>()));
				std::vector<PrimitiveDevBufPointers>& primitiveVector = (res.first)->second;

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
	int vid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (vid >= numVertices) return;
	// TODO: Apply vertex transformation here
	// Multiply the MVP matrix for each vertex position, this will transform everything into clipping space
	// Then divide the pos by its w element to transform into NDC space
	// Finally transform x and y to viewport space

    // data prefatch
    glm::vec3 origin_pos_v3 = primitive.dev_position[vid];
    glm::vec3 origin_normal_3 = primitive.dev_normal[vid];
    VertexOut vert_out;
    glm::vec4 origin_pos_v4 = glm::vec4(origin_pos_v3, 1.f);

    glm::vec4 pos = MVP * origin_pos_v4;
    pos /= pos.w;
    pos.x = width * (pos.x + 1.f) / 2.f;
    pos.y = height * (1.f - pos.y) / 2.f;

    vert_out.eyePos = glm::vec3(MV * origin_pos_v4); // pos in eye
    vert_out.eyeNor = glm::normalize(MV_normal * origin_normal_3);
    vert_out.pos = pos;

    if (primitive.dev_diffuseTex != NULL)
    {
        vert_out.texcoord0 = primitive.dev_texcoord0[vid];
        vert_out.texWidth = primitive.diffuseTexWidth;
        vert_out.texHeight = primitive.diffuseTexHeight;
        vert_out.dev_diffuseTex = primitive.dev_diffuseTex;
    }

    primitive.dev_verticesOut[vid] = vert_out;
}

__global__
void _vertexTransformAndAssembly2(
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
        glm::vec4 position = MVP * glm::vec4(primitive.dev_position[vid], 1.0f);
        position /= position.w;
        position.x = width * (position.x + 1.0f) / 2.0f;
        position.y = height * (1.0f - position.y) / 2.0f;
        glm::vec4 eyeposition = MV * glm::vec4(primitive.dev_position[vid], 1.0f);
        //eyeposition /= eyeposition.w;
        // TODO: Apply vertex assembly here
        // Assemble all attribute arraies into the primitive array
        primitive.dev_verticesOut[vid].eyeNor = glm::normalize(MV_normal * primitive.dev_normal[vid]);
        primitive.dev_verticesOut[vid].eyePos = glm::vec3(eyeposition);
        primitive.dev_verticesOut[vid].pos = position;
        if (primitive.dev_diffuseTex != NULL)
        {
            primitive.dev_verticesOut[vid].texcoord0 = primitive.dev_texcoord0[vid];
            primitive.dev_verticesOut[vid].dev_diffuseTex = primitive.dev_diffuseTex;
            primitive.dev_verticesOut[vid].texWidth = primitive.diffuseTexWidth;
            primitive.dev_verticesOut[vid].texHeight = primitive.diffuseTexHeight;
        }

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
void rasterizeKern(int totalNumPrimitives,
    Primitive* primitives, Fragment* fragmentBuffer,
    int* depths, int* fragMutex, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalNumPrimitives) return;
    
    // data prefatch
    Primitive primitive = primitives[idx];
    glm::vec3 pos_proj[3] = {   glm::vec3(primitive.v[0].pos),
                                glm::vec3(primitive.v[1].pos),
                                glm::vec3(primitive.v[2].pos) };

    glm::vec3 pos_in_eye[3] = { glm::vec3(primitive.v[0].eyePos),
                                glm::vec3(primitive.v[1].eyePos),
                                glm::vec3(primitive.v[2].eyePos) };

#if RENDER_LINE
    glm::vec3 color = glm::vec3(1.0f, 0.3f, 0.3f);
    passLine(fragmentBuffer, pos_proj[0], pos_proj[1], color, width, height);
    passLine(fragmentBuffer, pos_proj[1], pos_proj[2], color, width, height);
    passLine(fragmentBuffer, pos_proj[2], pos_proj[0], color, width, height);
    return;
#elif RENDER_POINT
    glm::vec3 color = glm::vec3(0.3f, 1.0f, 0.3f);
    passPoint(fragmentBuffer, pos_proj[0], color, width, height);
    passPoint(fragmentBuffer, pos_proj[1], color, width, height);
    passPoint(fragmentBuffer, pos_proj[2], color, width, height);
    return;
#endif

    AABB boundBox = getAABBForTriangle(pos_proj);

    Fragment frag_out;

    // loop through x and y, not z 
    for (int y = (int)boundBox.min.y;  y <= (int)boundBox.max.y; ++y) {
        // boundary clamp
        if (y < 0 || y > height) continue;
        for (int x = (int)boundBox.min.x; x <= (int)boundBox.max.x; ++x) {
            // boundary clamp
            if (x < 0 || x > width) continue;
            glm::vec3 bCCoord = calculateBarycentricCoordinate(pos_proj, glm::vec2(x, y));
            if (!isBarycentricCoordInBounds(bCCoord)) continue;
            int pixelId = x + y * width;
            float zP = getZAtCoordinate(bCCoord, pos_proj);
            int depth = (int)(zP * INT_MIN);

            atomicMin(&depths[pixelId], depth);

            if (depths[pixelId] != depth) continue;
            frag_out.depth = fabs(zP);
            frag_out.eyeNor = getBCInterpolate(bCCoord, primitive.v[0].eyeNor, primitive.v[1].eyeNor, primitive.v[2].eyeNor);
            frag_out.eyePos = getBCInterpolate(bCCoord, primitive.v[0].eyePos, primitive.v[1].eyePos, primitive.v[2].eyePos);

#if PERSP_CORRECT_UV
            frag_out.texcoord0 = getBCInterpolatePersp(bCCoord, primitive);
#else
            frag_out.texcoord0 = getBCInterpolate(bCCoord, primitive.v[0].texcoord0, primitive.v[1].texcoord0, primitive.v[2].texcoord0);
#endif
            
            frag_out.dev_diffuseTex = primitive.v[0].dev_diffuseTex;


            frag_out.texHeight = primitive.v[0].texHeight;
            frag_out.texWidth = primitive.v[0].texWidth;

            frag_out.color = glm::vec3(1.f, 1.f, 1.f);
            fragMutex[pixelId] = 0;
            fragmentBuffer[pixelId] = frag_out;

            //bool isSet;
            //depth = (int)(zP * INT_MAX);
            //do {
            //    isSet = (atomicCAS(&fragMutex[pixelId], 0, 1) == 0);
            //    if (!isSet) continue;
            //    if (depths[pixelId] <= depth) continue;
            //    frag_out.depth = fabs(zP);
            //    frag_out.eyeNor = getBCInterpolate(bCCoord, primitive.v[0].eyeNor, primitive.v[1].eyeNor, primitive.v[2].eyeNor);
            //    frag_out.eyePos = getBCInterpolate(bCCoord, primitive.v[0].eyePos, primitive.v[1].eyePos, primitive.v[2].eyePos);
            //    
            //    frag_out.texcoord0 = getBCInterpolate(bCCoord, primitive.v[0].texcoord0, primitive.v[1].texcoord0, primitive.v[2].texcoord0);
            //    frag_out.dev_diffuseTex = primitive.v[0].dev_diffuseTex;


            //    frag_out.texHeight = primitive.v[0].texHeight;
            //    frag_out.texWidth = primitive.v[0].texWidth;

            //    frag_out.color = glm::vec3(1.f, 1.f, 1.f);
            //    fragMutex[pixelId] = 0;
            //    fragmentBuffer[pixelId] = frag_out;
            //} while (!isSet);
            
        }   
    }
}


/**
 * Perform rasterization.
 */
void rasterize(uchar4* pbo, const glm::mat4& MVP, const glm::mat4& MV, const glm::mat3 MV_normal) {
	// Execute your rasterization pipeline here
	// (See README for rasterization pipeline outline.)

	// Vertex Process & primitive assembly
	{
		curPrimitiveBeginId = 0;
		dim3 numThreadsPerBlock(128);

		auto it = mesh2PrimitivesMap.begin();
		auto itEnd = mesh2PrimitivesMap.end();

		for (; it != itEnd; ++it) {
            // p is one PrimitiveDevBufPointers
			auto p = (it->second).begin();	// each primitive
			auto pEnd = (it->second).end();
			for (; p != pEnd; ++p) {
				dim3 numBlocksForVertices((p->numVertices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
				dim3 numBlocksForIndices((p->numIndices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);

				_vertexTransformAndAssembly << < numBlocksForVertices, numThreadsPerBlock >> >(p->numVertices, *p, MVP, MV, MV_normal, width, height);
				checkCUDAError("Vertex Processing");
                // actually no need to write this
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

    int sideLength2d = 8;
    // threads are flattened in a 2D pattern, one thread one pixel
    dim3 blockSize2d(sideLength2d, sideLength2d);
    dim3 blockCount2d((width - 1) / blockSize2d.x + 1,
        (height - 1) / blockSize2d.y + 1);
	
	cudaMemset(dev_fragmentBuffer, 0, width * height * sizeof(Fragment));
	initDepth << <blockCount2d, blockSize2d >> >(width, height, dev_depth);

    // init mutex array
    cudaMemset(dev_fragMutex, 0, width * height * sizeof(int));
    checkCUDAError("mutex init fail");

    dim3 rasterizeBlockSize(totalNumPrimitives < MAX_BLOCK_SIZE ? totalNumPrimitives : MAX_BLOCK_SIZE);
    dim3 rasterizeGridSize((totalNumPrimitives + rasterizeBlockSize.x - 1) / rasterizeBlockSize.x);
    rasterizeKern << <rasterizeGridSize, rasterizeBlockSize >> > (
        totalNumPrimitives,
        dev_primitives,
        dev_fragmentBuffer,
        dev_depth,
        dev_fragMutex,
        width, height);
    checkCUDAError("rasterize fail");

    // Copy depthbuffer colors into framebuffer
	render << <blockCount2d, blockSize2d >> >(width, height, dev_fragmentBuffer, dev_framebuffer);
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

    cudaFree(dev_fragMutex);
    dev_fragMutex = NULL;

    checkCUDAError("rasterize Free");
}
