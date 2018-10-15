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
#define LAMBERT 0
#define BLINN 1

#define POINT 1
#define TRI 0
#define LINE 0

#define TEXTURE 0
#define BILINEAR 1

#define PERSP_CORRECT 1

#define BACKCULL 1
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
		 glm::vec3 camPos;
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
		 glm::vec3 camPos;
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

static float * dev_depth = NULL;	// you might need this buffer when doing depth test
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
*/
__global__
void render(int w, int h, Fragment *fragmentBuffer, glm::vec3 *framebuffer) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * w);

    if (x < w && y < h) {
        

		// TODO: add your fragment shader code here
		Fragment &fragment = fragmentBuffer[index];
#if POINT || LINE 
		framebuffer[index] = fragment.color;
#elif LAMBERT
		glm::vec3 v = fragment.eyePos;
		glm::vec3 n = fragment.eyeNor;
		glm::vec3 fragColor(1, 0, 0);
		glm::vec3 lightPos(1, 1, 1);
		glm::vec3 L = glm::normalize(lightPos - v);
		float lambert = glm::max(0.f, glm::dot(L, n));
		framebuffer[index] = lambert * fragment.color;
#elif BLINN
		glm::vec3 v = fragment.eyePos;
		glm::vec3 n = fragment.eyeNor;
		glm::vec3 camPos = fragment.camPos;
		glm::vec3 lightPos(1, 1, 1);
		glm::vec3 L = glm::normalize(lightPos - v);
		glm::vec3 Ev = glm::normalize(-v);
		glm::vec3 R = glm::normalize(-glm::reflect(v - camPos, n));
		float specular = glm::pow(glm::max(glm::dot(R, Ev), 0.f), 16.f);
		float lambert = glm::max(0.f, glm::dot(L, n));
		glm::vec3 diffuse = lambert * fragment.color;
		framebuffer[index] = glm::clamp(diffuse*glm::vec3(0.7, 0.7, 0.7)
			+ specular * glm::vec3(0.3, 0.3, 0.3), glm::vec3(0), glm::vec3(1));
#else
		framebuffer[index] = fragment.color;
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
	cudaMalloc(&dev_depth, width * height * sizeof(float));

	cudaFree(dev_mutex);
	cudaMalloc(&dev_mutex, width * height * sizeof(int));
	cudaMemset(dev_mutex, 0, width * height * sizeof(int));
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


__forceinline__ __device__ glm::vec3 baryPos(VertexOut *v, glm::vec3  barycentric) {
	VertexOut vert0, vert1, vert2;
	vert0 = v[0];
	vert1 = v[1];
	vert2 = v[2];
	glm::vec3 p0 = vert0.eyePos * barycentric[0];
	glm::vec3 p1 = vert1.eyePos * barycentric[1];
	glm::vec3 p2 = vert2.eyePos * barycentric[2];
	return p0 + p1 + p2;
}

__forceinline__ __device__ glm::vec3 baryNorm(VertexOut *v, glm::vec3  barycentric) {
	VertexOut vert0, vert1, vert2;
	vert0 = v[0];
	vert1 = v[1];
	vert2 = v[2];
	glm::vec3 p0 = vert0.eyeNor * barycentric[0];
	glm::vec3 p1 = vert1.eyeNor * barycentric[1];
	glm::vec3 p2 = vert2.eyeNor * barycentric[2];
	return p0 + p1 + p2;
}


__forceinline__ __device__ glm::vec2 baryUVs(VertexOut *v, glm::vec3  barycentric) {
	VertexOut vert0, vert1, vert2;
	vert0 = v[0];
	vert1 = v[1];
	vert2 = v[2];
	glm::vec2 p0 = vert0.texcoord0 * barycentric[0];
	glm::vec2 p1 = vert1.texcoord0 * barycentric[1];
	glm::vec2 p2 = vert2.texcoord0 * barycentric[2];
	return p0 + p1 + p2;
}


__forceinline__ __device__ glm::vec2 baryUVsPerspective(VertexOut *v, glm::vec3  barycentric) {
	VertexOut vert0, vert1, vert2;
	vert0 = v[0];
	vert1 = v[1];
	vert2 = v[2];

	const float pc_z = 1.f /
		(barycentric.x / vert0.eyePos.z
			+ barycentric.y / vert1.eyePos.z
			+ barycentric.z / vert2.eyePos.z);

	glm::vec2 p0 = vert0.texcoord0 * barycentric[0] / vert0.eyePos.z;
	glm::vec2 p1 = vert1.texcoord0 * barycentric[1] / vert1.eyePos.z;
	glm::vec2 p2 = vert2.texcoord0 * barycentric[2] / vert2.eyePos.z;
	return (p0 + p1 + p2) * pc_z;
}

__device__
glm::vec3 getColor(TextureData* tex, int width, float x, float y) {
	int i = x + y * width;
	return glm::vec3(tex[i * 3], tex[i * 3 + 1], tex[i * 3 + 2]) / 255.f;
}

__global__ void kernelRasterize(int totalNumPrimitives, Primitive *dev_primitives, Fragment *dev_fragmentBuffer, float *dev_depth, int *dev_mutex, int width, int height) {
	int pid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (pid > totalNumPrimitives) return;
	// compute triangle from primitive
	Primitive primitive = dev_primitives[pid];
	glm::vec3 v0, v1, v2;
	v0 = glm::vec3(primitive.v[0].pos);
	v1 = glm::vec3(primitive.v[1].pos);
	v2 = glm::vec3(primitive.v[2].pos);
	glm::vec3 triangle[3] = { v0, v1, v2 };
#if BACKCULL
	if (glm::dot(primitive.v->eyeNor, primitive.v->camPos - primitive.v->eyePos) < 0.0f) {
		return;
	}
#endif

#if POINT
int x, y;
for (int i = 0; i < 3; i++) {
	x = triangle[i].x;
	y = triangle[i].y;
	int fragmentId = x + y * width;
	if ((x >= 0 && x <= width - 1) && (y >= 0 && y <= height - 1)) {
		dev_fragmentBuffer[fragmentId].color = primitive.v->eyeNor;
	}
}
#elif LINE
for (int i = 0; i < 3; i++) {
	int x1 = triangle[i].x;
	int x2 = triangle[i + 1].x;
	int y1 = triangle[i].y;
	int y2 = triangle[i + 1].y;
	int dx = x2 - x1;
	int dy = y2 - y1;
	for (int x = x1; x <= x2; x++) {
		int y = y1 + dy * (x - x1) / dx;
		int fragmentId = x + y * width;
		if (x < 0 || x >= width) continue;
		if (y < 0 || y >= height) continue;
		dev_fragmentBuffer[fragmentId].color = primitive.v->eyeNor;
	}
}
#elif TRI
	// compute bounding box and clip to screen
	AABB boundingBox = getAABBForTriangle(triangle);
	const int minX = glm::min(width - 1, glm::max(0, (int)boundingBox.min.x));
	const int minY = glm::min(height - 1, glm::max(0, (int)boundingBox.min.y));
	const int maxX = glm::min(width - 1, glm::max(0, (int)boundingBox.max.x));
	const int maxY = glm::min(height - 1, glm::max(0, (int)boundingBox.max.y));

	// iterate over bounding box and test which pixels are inside
	for (int x = minX; x <= maxX; ++x) {
		for (int y = minY; y <= maxY; ++y) {
			glm::vec3 barycentric = calculateBarycentricCoordinate(triangle, glm::vec2(x, y));
			bool inTriangle = isBarycentricCoordInBounds(barycentric);

			if (inTriangle) {
				const int fragmentId = x + (y * width);

				bool isSet;
				do {
					// it was unlocked so we lock it
					isSet = atomicCAS(&dev_mutex[fragmentId], 0, 1) == 0;
					if (isSet) {
						float depth = -getZAtCoordinate(barycentric, triangle) * INT_MAX;

						// if this fragment is closer, we set the new depth and fragment
						if (depth < dev_depth[fragmentId]) {
							dev_depth[fragmentId] = depth;
							Fragment &fragment = dev_fragmentBuffer[fragmentId];
							fragment.eyeNor = baryNorm(primitive.v, barycentric);
							fragment.eyePos = baryPos(primitive.v, barycentric);
							
							fragment.camPos = primitive.v[0].camPos;
#if PERSP_CORRECT
							glm::vec2 uvs = baryUVsPerspective(primitive.v, barycentric);
							fragment.texcoord0 = uvs;
#else
							glm::vec2 uvs = baryUVs(primitive.v, barycentric);
							fragment.texcoord0 = uvs;
#endif 

#if BILINEAR
							if (primitive.v->dev_diffuseTex != NULL) {
								float x = uvs[0] * primitive.v->texWidth;
								float y = uvs[1] * primitive.v->texHeight;
								int xx = glm::floor(x);
								int yy = glm::floor(y);
								float xfract = x - xx;
								float yfract = y - yy;
								float xinv = 1.f - xfract;
								float yinv = 1.f - yfract;
							
								TextureData *text = primitive.v->dev_diffuseTex;
								int width = primitive.v->texWidth;
								glm::vec3 tex00 = getColor(text, width, xx, yy);
								glm::vec3 tex10 = getColor(text, width, xx + 1, yy);
								glm::vec3 tex01 = getColor(text, width, xx, yy + 1);
								glm::vec3 tex11 = getColor(text, width, xx + 1, yy + 1);

								fragment.color = (tex00 * xinv + tex10 * xfract) * yinv + (tex01 * xinv + tex11 * xfract) * yfract;
							}
#elif TEXTURE
							if (primitive.v->dev_diffuseTex != NULL) {
								float x = uvs[0] * primitive.v->texWidth;
								float y = uvs[1] * primitive.v->texHeight;
								TextureData *text = primitive.v->dev_diffuseTex;
								int width = primitive.v->texWidth;
								fragment.color = getColor(text, width, glm::floor(x), glm::floor(y));
							}
#endif
							fragment.color = baryNorm(primitive.v, barycentric);
						}
						dev_mutex[fragmentId] = 0;
					}
				} while (!isSet);
			}
		}
	}
#endif
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
		glm::vec4 objPos(primitive.dev_position[vid], 1.f);
		glm::vec4 eyePos = MVP * objPos;
		glm::vec4 mvpPos = eyePos;
		mvpPos /= mvpPos.w;
		mvpPos.x = 0.5f * float(width) * (mvpPos.x + 1.f);
		mvpPos.y = 0.5f * float(height) * (1.f - mvpPos.y);

		// TODO: Apply vertex assembly here
		// Assemble all attribute arraies into the primitive array
		VertexOut &vo = primitive.dev_verticesOut[vid];
		vo.pos = mvpPos;
		vo.eyePos = glm::vec3(eyePos[0], eyePos[1], eyePos[2]);
		vo.eyeNor = glm::normalize(MV_normal * primitive.dev_normal[vid]);
		vo.camPos = glm::vec3(MVP * glm::vec4(0, 0, 0, 1));
		vo.texcoord0 = primitive.dev_texcoord0[vid];
		vo.dev_diffuseTex = primitive.dev_diffuseTex;
		vo.texHeight = primitive.diffuseTexHeight;
		vo.texWidth = primitive.diffuseTexWidth;
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
	dim3 numThreadsPerBlock(128);
	dim3 numBlocksForPrimitives((totalNumPrimitives + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
	kernelRasterize << <numBlocksForPrimitives, numThreadsPerBlock >> > (totalNumPrimitives, dev_primitives, dev_fragmentBuffer, dev_depth, dev_mutex, width, height);


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

    checkCUDAError("rasterize Free");
}
