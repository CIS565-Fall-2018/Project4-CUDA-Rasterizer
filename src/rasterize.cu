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
#include <random>
#include <cuda_runtime.h>
#include <thrust/random.h>
#include <util/checkCUDAError.h>
#include <util/tiny_gltf_loader.h>
#include "rasterizeTools.h"
#include "rasterize.h"
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include<curand.h>
#include<curand_kernel.h>
#include "common.h"

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

	enum DrawMode {
		pmode = 1,
		lmode = 2,
		tmode = 3
	};

	struct VertexOut {
		glm::vec4 pos;
		glm::vec3 actualnorm;
		// TODO: add new attributes to your VertexOut
		// The attributes listed below might be useful, 
		// but always feel free to modify on your own
		glm::vec4 actualpos;
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
		 glm::vec3 fragmentPos;
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

int N = 40;

static int width = 0;
static int height = 0;

static int totalNumPrimitives = 0;
static Primitive *dev_primitives = NULL;
static Fragment *dev_fragmentBuffer = NULL;
static glm::vec3 *dev_framebuffer = NULL;

static glm::vec3 *dev_randpts = NULL;

curandState* devStates;


DrawMode mode = tmode;
//#define SSAO
#define shader 0
#define color_interp  0
#define normaldebug  0;
#define depthdebug  0;
#define timer 1
#define perspectcorrect 0

static int * dev_depth = NULL;	// you might need this buffer when doing depth test

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

__device__ __host__ glm::vec3 getBilinearFilteredPixelColor(TextureData* tex,glm::vec2 UV, int texwidth, int texheight) {
	float u = UV.x * texwidth - 0.5;
	float v = UV.y * texheight - 0.5;
	int x = floor(u);
	int y = floor(v);
	double u_ratio = u - x;
	double v_ratio = v - y;
	double u_opposite = 1 - u_ratio;
	double v_opposite = 1 - v_ratio;
	int uvidx1 = 3 * (x + y*texwidth);
	int uvidx2 = 3 * (x + 1 + y*texwidth);
	int uvidx3 = 3 * (x + (y + 1)*texwidth);
	int uvidx4 = 3 * (x + 1 + (y + 1)*texwidth);
	double r = (tex[uvidx1] * u_opposite + tex[uvidx2] * u_ratio) * v_opposite +
		(tex[uvidx3] * u_opposite + tex[uvidx4] * u_ratio) * v_ratio;
	double g = (tex[uvidx1+1] * u_opposite + tex[uvidx2+1] * u_ratio) * v_opposite +
		(tex[uvidx3+1] * u_opposite + tex[uvidx4+1] * u_ratio) * v_ratio;
	double b = (tex[uvidx1+2] * u_opposite + tex[uvidx2+2] * u_ratio) * v_opposite +
		(tex[uvidx3+2] * u_opposite + tex[uvidx4+2] * u_ratio) * v_ratio;
	glm::vec3 colval(r, g, b);
	colval /= 255.f;
	return colval;
}

__device__ __host__ glm::vec3 generateRand()
{
	
	thrust::uniform_real_distribution<float> randomFLTs(0.0, 1.0);
	thrust::default_random_engine generator;
	glm::vec3 sample(randomFLTs(generator)*2.0 - 1.0,
		randomFLTs(generator)*2.0 - 1.0,
		randomFLTs(generator));
	sample = glm::normalize(sample);
	sample *= randomFLTs(generator);
	return sample;
}

 __host__ glm::vec3 generateRand1()
{
	std::random_device rd;
	std::uniform_real_distribution<float> randomFLTs(0.0, 1.0);
	std::default_random_engine generator(rd());
	glm::vec3 sample(randomFLTs(generator)*2.0 - 1.0,
		randomFLTs(generator)*2.0 - 1.0,
		randomFLTs(generator));
	sample = glm::normalize(sample);
	sample *= randomFLTs(generator);
	return sample;
}

/** 
* Writes fragment colors to the framebuffer
*/

__global__
void render(int w, int h, Fragment *fragmentBuffer, glm::vec3 *framebuffer,DrawMode mode,curandState* globalstate,glm::vec3 *randpts, glm::mat4 MVP, glm::mat4 MV, glm::mat3 MV_normal ) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * w);

	if (x < w && y < h) {
		glm::vec3 outcol;
		if (mode == lmode||mode == pmode)
		{

			outcol = fragmentBuffer[index].color;
		}
		else if(mode == tmode)
		{
#if perspectcorrect == 1
			glm::vec3 lightdir(0.5, 0.5, 1);
#else
			glm::vec3 lightdir(-0.5, -0.5, -1);
#endif
			float intensity = glm::dot(lightdir, fragmentBuffer[index].eyeNor);
			intensity = glm::clamp(intensity, 0.f, 0.8f);
#if shader==0
			outcol = glm::vec3(0);
			if (fragmentBuffer[index].color != glm::vec3(0)) outcol += glm::vec3(0.2f);
			outcol += intensity*fragmentBuffer[index].color;
#elif shader == 1
			float spec = 0.f;
			if (intensity > 0.1f)
			{
				glm::vec3 viewdir = -fragmentBuffer[index].eyePos;
				glm::vec3 half = glm::normalize(lightdir + viewdir);
				float theta = glm::clamp(glm::dot(half, fragmentBuffer[index].eyeNor), 0.f, 1.f);
				spec = glm::pow(theta, 20);
			}
			outcol = intensity*fragmentBuffer[index].color + spec*glm::vec3(1.f);
#elif shader == 2
			outcol = fragmentBuffer[index].color;
#endif
		}
		


		// TODO: add your fragment shader code here
		framebuffer[index] = outcol;
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
    cudaMalloc(&dev_framebuffer,   width * height * sizeof(glm::vec3));
    cudaMemset(dev_framebuffer, 0, width * height * sizeof(glm::vec3));
    
	cudaMalloc(&dev_randpts, 40 * sizeof(glm::vec3));
	cudaMemset(dev_randpts, 0, 40 * sizeof(glm::vec3));

	glm::vec3 arr[40];
	for (int i = 0; i < 40; ++i)
	{
		arr[i] = generateRand1();
		std::cout << arr[i].x<<","<< arr[i]. y<<","<< arr[i]. z<<std::endl;
	}
	cudaMemcpy(dev_randpts, arr, 40 * sizeof(glm::vec3), cudaMemcpyHostToDevice);
	cudaFree(dev_depth);
	cudaMalloc(&dev_depth, width * height * sizeof(int));

	cudaFree(dev_mutex);
	cudaMalloc(&dev_mutex, width*height * sizeof(int));

	cudaMalloc(&devStates, N * sizeof(curandState));
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

__global__
void initMutex(int w, int h, int * mutex)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < w && y < h)
	{
		int index = x + (y * w);
		mutex[index] = 0;
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

		glm::vec4 MVPpos = MVP*glm::vec4(primitive.dev_position[vid], 1.0f);
		primitive.dev_verticesOut[vid].actualpos = glm::vec4(primitive.dev_position[vid], 1.0f);
		primitive.dev_verticesOut[vid].actualnorm = primitive.dev_normal[vid];
		MVPpos /= MVPpos.w;
		MVPpos.x = 0.5f*(float)width*(MVPpos.x + 1.0f);
		MVPpos.y = 0.5f*(float)height*(-MVPpos.y + 1.0f);
		primitive.dev_verticesOut[vid].pos = MVPpos;
		// TODO: Apply vertex assembly here
		// Assemble all attribute arraies into the primitive array
		primitive.dev_verticesOut[vid].eyeNor = glm::normalize(MV_normal*primitive.dev_normal[vid]);
		glm::vec4 tmpvec = MV*glm::vec4(primitive.dev_position[vid],1.0f);
		//tmpvec /= tmpvec.w;
		primitive.dev_verticesOut[vid].eyePos = glm::vec3(tmpvec);
		if (primitive.dev_diffuseTex)
		{
			primitive.dev_verticesOut[vid].dev_diffuseTex = primitive.dev_diffuseTex;
			primitive.dev_verticesOut[vid].texcoord0 = primitive.dev_texcoord0[vid];
			primitive.dev_verticesOut[vid].texHeight = primitive.diffuseTexHeight;
			primitive.dev_verticesOut[vid].texWidth = primitive.diffuseTexWidth;
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


//__device__ float fatomicMin(float *addr, float value)

//{
//
//	float old = *addr, assumed;
//
//	if (old <= value) return old;
//
//	do
//
//	{
//
//		assumed = old;
//
//		old = atomicCAS((unsigned int*)addr, __float_as_int(assumed), __float_as_int(value));
//
//	} while (old != assumed)
//
//		return old;
//
//}

//perspective correction reference : https://www.scratchapixel.com/lessons/3d-basic-rendering/rasterization-practical-implementation/perspective-correct-interpolation-vertex-attributes
__host__ __device__ float getZatperspCorrected(glm::vec3 bary, glm::vec3 tri[3])
{
	return 1.f / (bary.x / tri[0].z + bary.y / tri[1].z + bary.z / tri[2].z);
}


__host__ __device__ glm::vec3 getperspCorrectedInterp(glm::vec3 bary, glm::vec3 a[3], glm::vec3 tri[3], float zcorrected)
{
	return zcorrected*(a[0] * bary.x / tri[0].z + a[1] * bary.y / tri[1].z + a[2] * bary.z / tri[2].z);
}

__host__ __device__ glm::vec3 getnonperspCorrectedInterp(glm::vec3 bary, glm::vec3 a[3], glm::vec3 tri[3])
{
	return (a[0] * bary.x  + a[1] * bary.y  + a[2] * bary.z );
}


__global__ void _TraingleRasterizer(int w, int h, Fragment* fragmentbuffer, Primitive* primitives, int *depth, int numPrimitives, int *mutex, glm::mat4 MVP, glm::mat4 MV, glm::mat3 MV_normal, glm::vec3 *randpts )
{
	int pid = (blockIdx.x*blockDim.x) + threadIdx.x;
	if (pid < numPrimitives)
	{
		Primitive& curP = primitives[pid];
		//caculate triangle bounding box:
		glm::vec3 tri[3];
		glm::vec3 triNor[3];
		glm::vec3 trieyePos[3];
		glm::vec3 triUV[3];
		glm::vec3 col[3]{glm::vec3(1,0,0),glm::vec3(0,1,0),glm::vec3(0,0,1)};
		triNor[0] = curP.v[0].eyeNor;
		triNor[1] = curP.v[1].eyeNor;
		triNor[2] = curP.v[2].eyeNor;

		trieyePos[0] = curP.v[0].eyePos;
		trieyePos[1] = curP.v[1].eyePos;
		trieyePos[2] = curP.v[2].eyePos;

		triUV[0] = glm::vec3(curP.v[0].texcoord0, 0);
		triUV[1] = glm::vec3(curP.v[1].texcoord0, 0);
		triUV[2] = glm::vec3(curP.v[2].texcoord0, 0);

		tri[0] = glm::vec3(curP.v[0].pos);
		tri[1] = glm::vec3(curP.v[1].pos);
		tri[2] = glm::vec3(curP.v[2].pos);
		

		AABB curBB = getAABBForTriangle(tri);
		for (int i = curBB.min.x; i < curBB.max.x; ++i)
		{
			for (int j = curBB.min.y; j < curBB.max.y; ++j)
			{
				if (i<0 || i>w || j<0 || j>h) continue;
				glm::vec3 baryCoord = calculateBarycentricCoordinate(tri, glm::vec2(i, j));
				if (isBarycentricCoordInBounds(baryCoord))
				{	
					int ScreenIdx = i + j*w;
#if perspectcorrect == 1
					int depthval = (int)(getZatperspCorrected(baryCoord, tri)*INT_MAX);
					float correctedZ = getZatperspCorrected(baryCoord, tri);
					glm::vec3 curnormal = getperspCorrectedInterp(baryCoord, triNor, tri, correctedZ);
					curnormal = glm::normalize(curnormal);
					glm::vec3 cureyePos = getperspCorrectedInterp(baryCoord, trieyePos, tri, correctedZ);
					glm::vec3 fragpos = getperspCorrectedInterp(baryCoord, tri, tri, correctedZ);
					glm::vec3 interpcol = getperspCorrectedInterp(baryCoord, col, tri, correctedZ);
#ifdef SSAO
					__shared__ glm::vec3 sharedfrags[16];
					int occlusion = 0;
					int samplecount = 40;
					float sampleradius = 5;
					glm::vec3 centernor;
					centernor = curnormal;

					glm::vec3 tangent;
					glm::vec3 c1 = glm::cross(centernor, glm::vec3(0, 0, 1.0));
					glm::vec3 c2 = glm::cross(centernor, glm::vec3(0, 1.0, 0));
					if (glm::length(c1) > glm::length(c2))
					{
						tangent = c1;
					}
					else
						tangent = c2;
					tangent = glm::normalize(tangent);
					glm::vec3 bitangent = glm::cross(tangent, centernor);
					glm::mat3 TBN(tangent, bitangent, centernor);
					for (int i = 0; i < samplecount; ++i)
					{
						glm::vec3 samp = randpts[i];
						samp = TBN*samp;
						samp = cureyePos + samp*sampleradius;
						glm::vec4 sample = glm::vec4(samp, 1);
						glm::mat4 p = glm::inverse(MV)*MVP;
						sample *= p;
						sample /= sample.w;
						sample.x = 0.5f*(float)w*(sample.x + 1.0f);
						sample.y = 0.5f*(float)h*(-sample.y + 1.0f);
						int curidx = sample.x + sample.y*w;
						glm::vec3 baryCoord1 = calculateBarycentricCoordinate(tri, glm::vec2(sample.x, sample.y));

						occlusion += (samp.z > cureyePos.z ? 1.0 : 0);
					}
					float resocclusion = 1 - (occlusion / samplecount);
					 
#endif // SSAO
#else
					int depthval = (int)(getZAtCoordinate(baryCoord, tri)*INT_MAX);
					glm::vec3 curnormal = glm::normalize(getnonperspCorrectedInterp(baryCoord, triNor, tri));
					glm::vec3 cureyePos = getnonperspCorrectedInterp(baryCoord, trieyePos, tri);
					glm::vec3 fragpos = getnonperspCorrectedInterp(baryCoord, tri, tri);
					glm::vec3 interpcol = getnonperspCorrectedInterp(baryCoord, col, tri);
#endif
					//glm::vec3 interpcol = getnonperspCorrectedInterp(baryCoord, col, tri);
					
					bool isSet;
					do {
						isSet = (atomicCAS(&mutex[ScreenIdx], 0, 1) == 0);
						if (isSet)
						{
							if (depth[ScreenIdx] >= depthval)
							{
								depth[ScreenIdx] = depthval;
								fragmentbuffer[ScreenIdx].eyeNor = curnormal;
								fragmentbuffer[ScreenIdx].eyePos = cureyePos;
								fragmentbuffer[ScreenIdx].fragmentPos = fragpos;
#if normaldebug == 1

								fragmentbuffer[ScreenIdx].color = glm::normalize(fragmentbuffer[ScreenIdx].eyeNor);
#elif depthdebug == 1
								fragmentbuffer[ScreenIdx].color = (1/correctedZ)*glm::vec3(1.f) / 38.f;
#elif color_interp == 1

								fragmentbuffer[ScreenIdx].color = interpcol;
#else
								if (curP.v[0].dev_diffuseTex != NULL)
								{
#if perspectcorrect == 1
									glm::vec2 UV = glm::vec2(getperspCorrectedInterp(baryCoord, triUV, tri, correctedZ));
#else
									glm::vec2 UV = glm::vec2(getnonperspCorrectedInterp(baryCoord, triUV, tri));
#endif
									fragmentbuffer[ScreenIdx].texcoord0 = UV;
									fragmentbuffer[ScreenIdx].dev_diffuseTex = curP.v[0].dev_diffuseTex;
									fragmentbuffer[ScreenIdx].texHeight = curP.v[0].texHeight;
									fragmentbuffer[ScreenIdx].texWidth = curP.v[0].texWidth;
									fragmentbuffer[ScreenIdx].color = getBilinearFilteredPixelColor(fragmentbuffer[ScreenIdx].dev_diffuseTex, fragmentbuffer[ScreenIdx].texcoord0
										, fragmentbuffer[ScreenIdx].texWidth, fragmentbuffer[ScreenIdx].texHeight);
									
								}
								else
								{
									fragmentbuffer[ScreenIdx].color = glm::vec3(1.0f);
								}
								//fragmentbuffer[ScreenIdx].color *= resocclusion;
#endif // color_interp
							}
							mutex[ScreenIdx] = 0;
						}
					} while (!isSet);
				}
			}
		}
	}
	
}

__host__ __device__ void kern_swap(float &a, float &b)
{
	float tmp = a;
	a = b;
	b = tmp;
}

__global__ void _PointRasterizer(int w, int h, Fragment* fragmentbuffer, Primitive* primitives, int *depth, int numPrimitives, int *mutex)
{
	int pid = (blockIdx.x*blockDim.x) + threadIdx.x;
	if (pid < numPrimitives)
	{
		Primitive& curP = primitives[pid];
		glm::vec3 tri[3];
		tri[0] = glm::vec3(curP.v[0].pos);
		tri[1] = glm::vec3(curP.v[1].pos);
		tri[2] = glm::vec3(curP.v[2].pos);
		for (int i = 0; i < 3; ++i)
		{
			int curpx = tri[i].x;
			int curpy = tri[i].y;
			int screenidx = curpx + curpy*w;
			fragmentbuffer[screenidx].color = glm::vec3(1.0f);
		}
		
	}
}


__global__ void _LineRasterizer(int w, int h, Fragment* fragmentbuffer, Primitive* primitives, int *depth, int numPrimitives, int *mutex)
{
	int pid = (blockIdx.x*blockDim.x) + threadIdx.x;
	if (pid < numPrimitives)
	{
		Primitive& curP = primitives[pid];
		glm::vec3 trin[3];
		trin[0] = glm::vec3(curP.v[0].pos);
		trin[1] = glm::vec3(curP.v[1].pos);
		trin[2] = glm::vec3(curP.v[2].pos);
		glm::vec3 tri[3][2];
		tri[0][0] = glm::vec3(curP.v[0].pos);
		tri[0][1] = glm::vec3(curP.v[1].pos);
		tri[1][0] = glm::vec3(curP.v[1].pos);
		tri[1][1] = glm::vec3(curP.v[2].pos);
		tri[2][0] = glm::vec3(curP.v[2].pos);
		tri[2][1] = glm::vec3(curP.v[0].pos);
		for (int i = 0; i < 3; ++i)
		{
			glm::vec3 startp = tri[i][0];
			glm::vec3 endp = tri[i][1];
			float x1 = startp.x; float x2 = endp.x;
			float y1 = startp.y; float y2 = endp.y;
			if (x1<0 || x1>w || x2<0 || x2>w || y1<0 || y1>h || y2<0 || y2>h) continue;
			const bool steep = (fabs(y2 - y1) > fabs(x2 - x1));
			if (steep)
			{
				kern_swap(x1, y1);
				kern_swap(x2, y2);
			}

			if (x1 > x2)
			{
				kern_swap(x1, x2);
				kern_swap(y1, y2);
			}

			const float dx = x2 - x1;
			const float dy = fabs(y2 - y1);

			float error = dx / 2.0f;
			const int ystep = (y1 < y2) ? 1 : -1;
			int y = (int)y1;

			const int maxX = (int)x2;

			for (int x = (int)x1; x<maxX; x++)
			{
				int screenidx;
				glm::vec2 rcoord;
				if (steep)
				{
					rcoord = glm::vec2(y, x);
					screenidx = y + x*w;
				}
				else
				{
					rcoord = glm::vec2(x, y);
					screenidx = x + y*w;
				}
				glm::vec3 baryCoord = calculateBarycentricCoordinate(trin, rcoord);
				//if(!isBarycentricCoordInBounds(baryCoord)) continue;
				float testdepth = (getZatperspCorrected(baryCoord, trin));
				int depthval = (int)(testdepth*INT_MAX);
				bool isSet;
				do {
					isSet = (atomicCAS(&mutex[screenidx], 0, 1) == 0);
					if (isSet)
					{
						if (depth[screenidx] > depthval)
						{
							depth[screenidx] = depthval;
							fragmentbuffer[screenidx].color =/*(1-glm::clamp(testdepth,0.f,1.f))* */glm::vec3(1.f);
						}
						mutex[screenidx] = 0;
					}
				} while (!isSet);
				error -= dy;
				if (error < 0)
				{
					y += ystep;
					error += dx;
				}
			}

		}
	}
}

/**
 * Perform rasterization.
 */
void rasterize(uchar4 *pbo, const glm::mat4 & MVP, const glm::mat4 & MV, const glm::mat3 MV_normal) {
	StreamCompaction::Common::PerformanceTimer vertexTranstimer;
	StreamCompaction::Common::PerformanceTimer primitiveassemblytimer;
	StreamCompaction::Common::PerformanceTimer rasterizertimer;
	StreamCompaction::Common::PerformanceTimer rendertimer;

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

		float vertextranstimer = 0.f;
		float primitiveassembletimer = 0.f;
		for (; it != itEnd; ++it) {
			auto p = (it->second).begin();	// each primitive
			auto pEnd = (it->second).end();
			for (; p != pEnd; ++p) {
				dim3 numBlocksForVertices((p->numVertices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
				dim3 numBlocksForIndices((p->numIndices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
				vertexTranstimer.startGpuTimer();
				_vertexTransformAndAssembly << < numBlocksForVertices, numThreadsPerBlock >> >(p->numVertices, *p, MVP, MV, MV_normal, width, height);
				vertexTranstimer.endGpuTimer();
				vertextranstimer += vertexTranstimer.getGpuElapsedTimeForPreviousOperation();
				checkCUDAError("Vertex Processing");
				
				cudaDeviceSynchronize();
				primitiveassemblytimer.startGpuTimer();
				_primitiveAssembly << < numBlocksForIndices, numThreadsPerBlock >> >
					(p->numIndices, 
					curPrimitiveBeginId, 
					dev_primitives, 
					*p);
				primitiveassemblytimer.endGpuTimer();
				primitiveassembletimer += primitiveassemblytimer.getGpuElapsedTimeForPreviousOperation();
				checkCUDAError("Primitive Assembly");
				

				curPrimitiveBeginId += p->numPrimitives;
			}
		}
#if timer == 1
		std::cout << "vertex transform time:" << vertextranstimer << std::endl;
		std::cout << "primitive assembly time:" << primitiveassembletimer << std::endl;
#endif

		checkCUDAError("Vertex Processing and Primitive Assembly");
	}
	
	cudaMemset(dev_fragmentBuffer, 0, width * height * sizeof(Fragment));
	initDepth << <blockCount2d, blockSize2d >> >(width, height, dev_depth);

	initMutex << <blockCount2d, blockSize2d >> > (width, height, dev_mutex);
	
	int ScanNumPrimitives = totalNumPrimitives;
	dim3 numTreadsPerBlock(128);
	dim3 numBlocksScan((ScanNumPrimitives + numTreadsPerBlock.x - 1)/numTreadsPerBlock.x);

	// TODO: rasterize
	rasterizertimer.startGpuTimer();
	if(mode==tmode)
	_TraingleRasterizer << <numBlocksScan, numTreadsPerBlock >> > (width, height, dev_fragmentBuffer, dev_primitives, dev_depth, totalNumPrimitives,dev_mutex, MVP, MV, MV_normal,dev_randpts);
	if(mode==lmode)
	_LineRasterizer << <numBlocksScan, numTreadsPerBlock >> > (width, height, dev_fragmentBuffer, dev_primitives, dev_depth, totalNumPrimitives, dev_mutex);
	if(mode==pmode)
	_PointRasterizer << <numBlocksScan, numTreadsPerBlock >> > (width, height, dev_fragmentBuffer, dev_primitives, dev_depth, totalNumPrimitives, dev_mutex);
	rasterizertimer.endGpuTimer();
#if timer ==1
	std::cout << "rasterization time:" << rasterizertimer.getGpuElapsedTimeForPreviousOperation() << std::endl;
#endif
    // Copy depthbuffer colors into framebuffer
	rendertimer.startGpuTimer();
	render << <blockCount2d, blockSize2d >> >(width, height, dev_fragmentBuffer, dev_framebuffer,mode,devStates,dev_randpts, MVP, MV, MV_normal);
	rendertimer.endGpuTimer();
#if timer == 1
	std::cout << "render time" << rendertimer.getGpuElapsedTimeForPreviousOperation() << std::endl;
#endif
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
	dev_mutex = NULL;

	cudaFree(devStates);

	cudaFree(dev_randpts);
    checkCUDAError("rasterize Free");
}
