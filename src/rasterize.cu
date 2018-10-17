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


#define BLINN 0
#define LAMBERT 1
#define BILINEAR_FILTERING 0
#define PERSPECTIVE_CORRECT 0
#define RASTERIZE_POINT 0
#define RASTERIZE_LINE 0
#define COLOR_INTERPOLATION 0
#define TIMER 1

static std::map<std::string, std::vector<PrimitiveDevBufPointers>> mesh2PrimitivesMap;


static int width = 0;
static int height = 0;

static int totalNumPrimitives = 0;
static Primitive *dev_primitives = NULL;
static Fragment *dev_fragmentBuffer = NULL;
static glm::vec3 *dev_framebuffer = NULL;

static int * dev_depth = NULL;	// you might need this buffer when doing depth test
#if TIMER
static double time_assembly = 0.0;
static double time_rasterize = 0.0;
static double time_render = 0.0;
static double time_sendToPBO = 0.0;
static int iter = 0;
#endif

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
void render(int w, int h, Fragment *fragmentBuffer, glm::vec3 *framebuffer) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * w);

    if (x < w && y < h) {

#if RASTERIZE_POINT || RASTERIZE_LINE
		framebuffer[index] = fragmentBuffer[index].color;
#else

		if (fragmentBuffer[index].dev_diffuseTex == NULL) {
			framebuffer[index] = fragmentBuffer[index].color;
		}
		else {
			// TODO: add your fragment shader code here
			Fragment frag = fragmentBuffer[index];
			glm::vec3 lightPos(5.f, 5.f, 5.f);
			glm::vec3 lightVec = glm::normalize((lightPos - frag.eyePos));
			glm::vec3 specColor(0.f, 0.f, 0.f);
			float lambertian = glm::max(glm::dot(lightVec, frag.eyeNor), 0.0f);
			glm::vec3 V = glm::normalize(frag.eyePos);
			glm::vec3 L = lightVec;
			glm::vec3 H = (V + L) / 2.f;
			float exp = 20.f;
			float specularTerm = glm::max(pow(glm::dot(glm::normalize(H), glm::normalize(frag.eyeNor)), exp), 0.f);;
			float ambientTerm = 0.2;
			float diffuseTerm = glm::dot(glm::normalize(frag.eyeNor), glm::normalize(lightVec));
			diffuseTerm = glm::clamp(diffuseTerm, 0.f, 1.f);
			glm::vec3 res = glm::vec3();
#if BLINN
			res = ambientTerm * glm::vec3(0.1, 0.1, 0.1) + diffuseTerm * frag.color + specularTerm * glm::vec3(1.f, 1.f, 1.f);
#elif LAMBERT
			res = ambientTerm * glm::vec3(0.1, 0.1, 0.1) + diffuseTerm * frag.color;
#else 
			res = frag.color;
#endif
			framebuffer[index] = res;
		}
#endif
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
		glm::vec4 pos = glm::vec4(primitive.dev_position[vid], 1.f);
		pos = MVP * pos;
		pos = pos / pos.w;
		pos.x = (pos.x + 1.f) * 0.5f * width;
		pos.y = (1.f - pos.y) * 0.5f * height;
		primitive.dev_verticesOut[vid].pos = pos;
		glm::vec4 eyePos = glm::vec4(primitive.dev_position[vid], 1.f);
		eyePos = MV * eyePos;
		eyePos = eyePos / eyePos.w;
		primitive.dev_verticesOut[vid].eyePos = glm::vec3(eyePos);
		glm::vec3 eyeNor = primitive.dev_normal[vid];
		primitive.dev_verticesOut[vid].eyeNor = glm::normalize(MV_normal * eyeNor);
		primitive.dev_verticesOut[vid].col = glm::vec3(0.5, 0.5, 0.5);
		// TODO: Apply vertex assembly here
		// Assemble all attribute arraies into the primitive array
		if (primitive.dev_texcoord0 != NULL) {
			primitive.dev_verticesOut[vid].texcoord0 = primitive.dev_texcoord0[vid];
		}
		else {
			primitive.dev_verticesOut[vid].texcoord0 = glm::vec2(0.f, 0.f);
		}
		if (primitive.dev_diffuseTex != NULL) {
			primitive.dev_verticesOut[vid].dev_diffuseTex = primitive.dev_diffuseTex;

		}
		primitive.dev_verticesOut[vid].texWidth = primitive.diffuseTexWidth;
		primitive.dev_verticesOut[vid].texHeight = primitive.diffuseTexHeight;

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

__device__ glm::vec3 getBilinearFilteredPixelColor(TextureData* texture, float u, float v, int width, int height) {
	u = u * width - 0.5;
	v = v * height - 0.5;
	int x = floor(u);
	int y = floor(v);
	float du = u - x;
	float dv = v - y;
	float u_opposite = 1 - du;
	float v_opposite = 1 - dv;
	int xy00 = 3 * (x + y * width);
	int xy10 = 3 * (x + 1 + y * width);
	int xy01 = 3 * (x + (y + 1) * width);
	int xy11 = 3 * (x + 1 + (y + 1) * width);
	float r = (texture[xy00] * u_opposite + texture[xy10] * du) * v_opposite +
		(texture[xy10] * u_opposite + texture[xy11] * du) * dv;
	float g = (texture[xy00 + 1] * u_opposite + texture[xy10 + 1] * du) * v_opposite +
		(texture[xy01 + 1] * u_opposite + texture[xy11 + 1] * du) * dv;
	float b = (texture[xy00 + 2] * u_opposite + texture[xy10 + 2] * du) * v_opposite +
		(texture[xy01 + 2] * u_opposite + texture[xy11 + 2] * du) * dv;
	return glm::vec3(r, g, b);
}

__global__ void rasterize_triangle(const int width, const int height, int* depth, int numPrimitives, Primitive* primitives, Fragment* fragmentBuffer) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index >= numPrimitives) {
		return;
	}
	Primitive frag = primitives[index];
	glm::vec3 tri[3];
	tri[0] = glm::vec3(frag.v[0].pos);
	tri[1] = glm::vec3(frag.v[1].pos);
	tri[2] = glm::vec3(frag.v[2].pos);
#if COLOR_INTERPOLATION
		frag.v[0].col = glm::vec3(1.f, 0.f, 0.f);
		if (index % 2 == 0) {
			frag.v[1].col = glm::vec3(0.f, 1.f, 0.f);
			frag.v[2].col = glm::vec3(0.f, 0.f, 1.f);
		}
		else {
			frag.v[2].col = glm::vec3(0.f, 1.f, 0.f);
			frag.v[1].col = glm::vec3(0.f, 0.f, 1.f);
		}
#endif
#if RASTERIZE_POINT
	glm::vec3 pointCol = glm::vec3(1.f, 0.f, 0.f);
	for (int i = 0; i < 3; i++) {
		tri[i].x = glm::clamp(tri[i].x, 0.f, (float)(width - 1));
		tri[i].y = glm::clamp(tri[i].y, 0.f, (float)(height - 1));
		int pixel = int(tri[i].x) + int(tri[i].y) * width;
		fragmentBuffer[pixel].color = pointCol;
	}
#elif RASTERIZE_LINE
	glm::vec3 lineCol = glm::vec3(0.f, 0.f, 1.f);
	glm::vec2 start = glm::vec2(tri[0].x, tri[0].y);
	glm::vec2 end = glm::vec2(tri[1].x, tri[1].y);
	float length = glm::length(glm::vec2(end.x - start.x, end.y - start.y));
	for (float di = 0.f; di < 1.f; di += 1.f / length) {
		int x = start.x * (1 - di) + end.x * di;
		int y = start.y * (1 - di) + end.y * di;
		int pixel = x + y * width;
		fragmentBuffer[pixel].color = lineCol;
	}
	start = glm::vec2(tri[1].x, tri[1].y);
	end = glm::vec2(tri[2].x, tri[2].y);
	length = glm::length(glm::vec2(end.x - start.x, end.y - start.y));
	for (float di = 0.f; di < 1.f; di += 1.f / length) {
		int x = start.x * (1 - di) + end.x * di;
		int y = start.y * (1 - di) + end.y * di;
		int pixel = x + y * width;
		fragmentBuffer[pixel].color = lineCol;
	}
	start = glm::vec2(tri[2].x, tri[2].y);
	end = glm::vec2(tri[0].x, tri[0].y);
	length = glm::length(glm::vec2(end.x - start.x, end.y - start.y));
	for (float di = 0.f; di < 1.f; di += 1.f / length) {
		int x = start.x * (1 - di) + end.x * di;
		int y = start.y * (1 - di) + end.y * di;
		int pixel = x + y * width;
		fragmentBuffer[pixel].color = lineCol;
	}
#else
	AABB bb = getAABBForTriangle(tri);
	bb.min[0] = glm::clamp(bb.min[0], 0.f, float(width) - 1);
	bb.min[1] = glm::clamp(bb.min[1], 0.f, float(height) - 1);
	bb.max[0] = glm::clamp(bb.max[0], 0.f, float(width) - 1);
	bb.max[1] = glm::clamp(bb.max[1], 0.f, float(height) - 1);

	for (int i = bb.min[0]; i <= bb.max[0]; i++) {
		for (int j = bb.min[1]; j <= bb.max[1]; j++) {
			glm::vec3 bary = calculateBarycentricCoordinate(tri, glm::vec2(i, j));
			if (isBarycentricCoordInBounds(bary)) {
#if PERSPECTIVE_CORRECT
				int curDepth = 1.0f / (bary[0] / tri[0].z + bary[1] / tri[1].z + bary[2] / tri[2].z);
#else
				int curDepth = getZAtCoordinate(bary, tri) * INT_MIN;
#endif
				int pixel = i + j * width;
				atomicMin(&depth[pixel], curDepth);
				if (depth[pixel] == curDepth) {
					fragmentBuffer[pixel].eyePos = bary[0] * frag.v[0].eyePos + bary[1] * frag.v[1].eyePos + bary[2] * frag.v[2].eyePos;
#if PERSPECTIVE_CORRECT
					tri[0].z += FLT_EPSILON;
					tri[1].z += FLT_EPSILON;
					tri[2].z += FLT_EPSILON;
					float perspectiveZ = 1.0f / (bary[0] / tri[0].z + bary[1] / tri[1].z + bary[2] / tri[2].z);
					fragmentBuffer[pixel].eyeNor = glm::normalize(perspectiveZ * (bary[0] * frag.v[0].eyeNor / tri[0].z + bary[1] * frag.v[1].eyeNor / tri[1].z + bary[2] * frag.v[2].eyeNor / tri[2].z));
					fragmentBuffer[pixel].texcoord0 = perspectiveZ * (bary[0] * frag.v[0].texcoord0 / tri[0].z + bary[1] * frag.v[1].texcoord0 / tri[1].z + bary[2] * frag.v[2].texcoord0 / tri[2].z);
#else
					fragmentBuffer[pixel].eyeNor = glm::normalize(bary[0] * frag.v[0].eyeNor + bary[1] * frag.v[1].eyeNor + bary[2] * frag.v[2].eyeNor);
					fragmentBuffer[pixel].texcoord0 = bary[0] * frag.v[0].texcoord0 + bary[1] * frag.v[1].texcoord0 + bary[2] * frag.v[2].texcoord0;
#endif
					fragmentBuffer[pixel].texWidth = frag.v[0].texWidth;
					fragmentBuffer[pixel].texHeight = frag.v[0].texHeight;
					fragmentBuffer[pixel].dev_diffuseTex = frag.v[0].dev_diffuseTex;
					if (fragmentBuffer[pixel].dev_diffuseTex != NULL) {
						TextureData* texture = fragmentBuffer[pixel].dev_diffuseTex;
						float uf = fragmentBuffer[pixel].texcoord0.x * fragmentBuffer[pixel].texWidth;
						float vf = fragmentBuffer[pixel].texcoord0.y * fragmentBuffer[pixel].texHeight;
						int u = int(uf);
						int v = int(vf);
						u = glm::min(glm::max(0, u), fragmentBuffer[pixel].texWidth - 1);
						v = glm::min(glm::max(0, v), fragmentBuffer[pixel].texHeight - 1);		
#if BILINEAR_FILTERING
						fragmentBuffer[pixel].color = getBilinearFilteredPixelColor(texture, fragmentBuffer[pixel].texcoord0.x, fragmentBuffer[pixel].texcoord0.y, fragmentBuffer[pixel].texWidth, fragmentBuffer[pixel].texHeight) / 255.f;
#else
						int colorInd = 3 * (v * fragmentBuffer[pixel].texWidth + u);
						fragmentBuffer[pixel].color = glm::vec3(texture[colorInd], texture[colorInd + 1], texture[colorInd + 2]) / 255.f;
#endif
					}
					else {
						fragmentBuffer[pixel].color = bary[0] * frag.v[0].col + bary[1] * frag.v[1].col + bary[2] * frag.v[2].col;
					}
				}
			}
		}
	}
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

	// Execute your rasterization pipeline here
	// (See README for rasterization pipeline outline.)
#if TIMER
	auto start = std::chrono::high_resolution_clock::now();
#endif
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
#if TIMER
	cudaDeviceSynchronize();
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	time_assembly += double(duration.count());
	std::cout << ++iter << " iteration" << std::endl;
	std::cout << "Vertex transform and assembly cost " << time_assembly << " microsecond" << std::endl;
#endif	
	cudaMemset(dev_fragmentBuffer, 0, width * height * sizeof(Fragment));
	initDepth << <blockCount2d, blockSize2d >> >(width, height, dev_depth);
	
	// TODO: rasterize

	dim3 numThreadsPerBlock(128);
	dim3 numBlocksPerGrid((totalNumPrimitives + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
#if TIMER
	start = std::chrono::high_resolution_clock::now();
#endif
	rasterize_triangle << <numBlocksPerGrid, numThreadsPerBlock >> > (width, height, dev_depth, totalNumPrimitives, dev_primitives, dev_fragmentBuffer);
#if TIMER
	cudaDeviceSynchronize();
	end = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	time_rasterize += double(duration.count());
	std::cout << "Rasterization cost " << time_rasterize << " microsecond" << std::endl;
	start = std::chrono::high_resolution_clock::now();
#endif
    // Copy depthbuffer colors into framebuffer
	render << <blockCount2d, blockSize2d >> >(width, height, dev_fragmentBuffer, dev_framebuffer);
	checkCUDAError("fragment shader");
#if TIMER
	cudaDeviceSynchronize();
	end = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	time_render += double(duration.count());
	std::cout << "rendering cost " << time_render << " microsecond" << std::endl;
	start = std::chrono::high_resolution_clock::now();
#endif
    // Copy framebuffer into OpenGL buffer for OpenGL previewing
    sendImageToPBO<<<blockCount2d, blockSize2d>>>(pbo, width, height, dev_framebuffer);
    checkCUDAError("copy render result to pbo");
#if TIMER
	cudaDeviceSynchronize();
	end = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	time_sendToPBO += double(duration.count());
	std::cout << "sendImageToPBO cost " << time_sendToPBO << " microsecond" << std::endl;
	std::cout << "total time is  " << time_sendToPBO + time_render + time_rasterize + time_assembly << " microsecond" << std::endl;
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


