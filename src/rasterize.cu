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

#define depthtest 0
#define normaltest 0
#define textureenabled 1

#define lambert 0
#define blinnphong 0
#define perspectivecorrection 0
#define bilinearfiltering 0
#define primitiveline 0
#define primitivepoints 0
#define backfaceculling 0
#define timeanalysis 0

PerformanceTimer& timer()
{
	static PerformanceTimer timer;
	return timer;
}

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

		 glm::vec2 texcoord0;
		 TextureData* dev_diffuseTex = NULL;
		 int texWidth, texHeight;

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

		glm::vec3 pos;

		// used for shading
		glm::vec3 eyePos;
		glm::vec3 eyeNor;
		int texHeight, texWidth;
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
		//glm::vec3 dev_diffuseColor;
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
__device__
static int depthscale = 100;

static int totalNumPrimitives = 0;
static Primitive *dev_primitives = NULL;
static Fragment *dev_fragmentBuffer = NULL;
static glm::vec3 *dev_framebuffer = NULL;

static int * dev_depth = NULL;	// you might need this buffer when doing depth test

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

__device__
bool isBackface(glm::vec3 tri[3]) {
	glm::vec3 d0 = tri[1] - tri[0];
	glm::vec3 d1 = tri[2] - tri[0];
	glm::vec3 nor = glm::normalize(glm::cross(d0, d1));
	glm::vec3 dir = glm::vec3(0.0f, 0.0f, 1.0f);
	return (glm::dot(dir, nor) > 0.0f);
}

//=================================================================stages====================================================================
#pragma region stages

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
void _clearFragment(int w, int h, Fragment* dev_fragmentBuffer, glm::vec3 color) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	if (x < w && y < h)
	{
		int id = x + (y * w);
		dev_fragmentBuffer[id].color = color;
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

		// TODO: Apply vertex assembly here
		// Assemble all attribute arraies into the primitive array

		glm::vec4 position = glm::vec4(primitive.dev_position[vid], 1.0f);
		position = MVP * position;
		position /= position.w;
		position.x = (position.x + 1.0f) * 0.5f * width;
		position.y = (1.0f - position.y) * 0.5f * height;
		glm::vec4 eyePos = MV * glm::vec4(primitive.dev_position[vid], 1.0f);

		glm::vec3 normal = MV_normal * primitive.dev_normal[vid];

		VertexOut& thisVOut = primitive.dev_verticesOut[vid];
		thisVOut.pos = position;
		thisVOut.eyePos = glm::vec3(eyePos);
		thisVOut.eyeNor = glm::normalize(normal);

		if (primitive.dev_texcoord0 != NULL) {
			thisVOut.texcoord0 = primitive.dev_texcoord0[vid];
		}
		if (primitive.dev_diffuseTex != NULL) {
			thisVOut.dev_diffuseTex = primitive.dev_diffuseTex;
			thisVOut.texHeight = primitive.diffuseTexHeight;
			thisVOut.texWidth = primitive.diffuseTexWidth;
		}

	}
}

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
			dev_primitives[pid + curPrimitiveBeginId].primitiveType = PrimitiveType::Triangle;
		}
		else if (primitive.primitiveMode == TINYGLTF_MODE_TRIANGLE_STRIP) {
			pid = (iid - 2) < 0 ? 0 : iid - 2;
			int pidf = (iid < numIndices - 2) ? iid : numIndices - 3;
			for (; pid <= pidf; pid++) {
				dev_primitives[pid + curPrimitiveBeginId].v[iid - pid] =
					primitive.dev_verticesOut[primitive.dev_indices[iid]];
				dev_primitives[pid + curPrimitiveBeginId].primitiveType = PrimitiveType::Triangle;
			}
		}
		else if (primitive.primitiveMode == TINYGLTF_MODE_TRIANGLE_FAN) {
			if (iid > 0) {
				int pid0 = iid - 2;
				int pid1 = iid - 1;
				if (pid1 >= numIndices - 2) {
					dev_primitives[pid0 + curPrimitiveBeginId].primitiveType = PrimitiveType::Triangle;
					dev_primitives[pid0 + curPrimitiveBeginId].v[iid - pid0] = 
						primitive.dev_verticesOut[primitive.dev_indices[iid]];
					dev_primitives[pid0 + curPrimitiveBeginId].v[0] =
						primitive.dev_verticesOut[primitive.dev_indices[0]];
				}
				else if (pid0 < 0) {
					dev_primitives[pid1 + curPrimitiveBeginId].v[iid - pid1] =
						primitive.dev_verticesOut[primitive.dev_indices[iid]];
				}
				else {
					dev_primitives[pid0 + curPrimitiveBeginId].primitiveType = PrimitiveType::Triangle;
					dev_primitives[pid0 + curPrimitiveBeginId].v[iid - pid0] =
						primitive.dev_verticesOut[primitive.dev_indices[iid]];
					dev_primitives[pid0 + curPrimitiveBeginId].v[0] =
						primitive.dev_verticesOut[primitive.dev_indices[0]];
					dev_primitives[pid1 + curPrimitiveBeginId].v[iid - pid1] =
						primitive.dev_verticesOut[primitive.dev_indices[iid]];
				}

			}
		}

	}

}


__global__
void _rasterize(int numPrimitives, int w, int h, Primitive* primitives, Fragment* fragments, int* depths)
{
	int id = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (id >= numPrimitives) {
		return;
	}

	Primitive p = primitives[id];
	glm::vec3 tri[3] = {
		glm::vec3(p.v[0].pos),
		glm::vec3(p.v[1].pos),
		glm::vec3(p.v[2].pos)
	};

#if backfaceculling
	if (isBackface(tri)) {
		return;
	}
#endif

	glm::vec3 col[3] = {
		glm::vec3(1, 0, 0),
		glm::vec3(0, 1, 0),
		glm::vec3(0, 0, 1)
	};
	glm::vec3 eyePos[3] = {
		p.v[0].eyePos,
		p.v[1].eyePos,
		p.v[2].eyePos
	};

#if primitivepoints
	
	int x0 = glm::clamp((int)tri[0].x, 0, w - 1), y0 = glm::clamp((int)tri[0].y, 0, h - 1);
	int idx0 = x0 + w * y0;
	fragments[idx0].color = glm::vec3(1);
	int x1 = glm::clamp((int)tri[1].x, 0, w - 1), y1 = glm::clamp((int)tri[1].y, 0, h - 1);
	int idx1 = x1 + w * y1;
	fragments[idx1].color = glm::vec3(1);
	int x2 = glm::clamp((int)tri[2].x, 0, w - 1), y2 = glm::clamp((int)tri[2].y, 0, h - 1);
	int idx2 = x2 + w * y2;
	fragments[idx2].color = glm::vec3(1);

#elif primitiveline



#else

	AABB aabb = getAABBForTriangle(tri);

	for (int x = aabb.min.x; x <= aabb.max.x; x++) {
		for (int y = aabb.min.y; y <= aabb.max.y; y++) {
			if (x >= w || x < 0 || y >= h || y < 0) continue;
			glm::vec3 baryCoord = calculateBarycentricCoordinate(tri, glm::vec2(x, y));
			if (isBarycentricCoordInBounds(baryCoord, 3, 0.05f))
			{
				int index = x + y * w;
#if perspectivecorrection
				float z = getPerspectiveZAtCoordinate(baryCoord, eyePos);
#else
				float z = getZAtCoordinate(baryCoord, tri);
#endif
				int depth = depthscale * (-z);
				atomicMin(&depths[index], depth);
				if (depth == depths[index]) {
					fragments[index].eyePos = barycentricInterpolation(baryCoord, p.v[0].eyePos, p.v[1].eyePos, p.v[2].eyePos);
					fragments[index].eyeNor = barycentricInterpolation(baryCoord, p.v[0].eyeNor, p.v[1].eyeNor, p.v[2].eyeNor);
					fragments[index].dev_diffuseTex = p.v[0].dev_diffuseTex;
					fragments[index].texWidth = p.v[0].texWidth;
					fragments[index].texHeight = p.v[0].texHeight;
#if perspectivecorrection
					fragments[index].pos = perspectiveCorrectBCIterpolation(baryCoord, eyePos, tri, -z);
					fragments[index].pos.z = 1.0f - fragments[index].pos.z;
					fragments[index].color = perspectiveCorrectBCIterpolation(baryCoord, eyePos, col, -z);
					fragments[index].texcoord0 = perspectiveCorrectBCIterpolation(baryCoord, eyePos, p.v[0].texcoord0, p.v[1].texcoord0, p.v[2].texcoord0, -z);
#else
					fragments[index].pos = barycentricInterpolation(baryCoord, tri);
					fragments[index].pos.z = 1.0f - fragments[index].pos.z;
					fragments[index].color = barycentricInterpolation(baryCoord, col);
					fragments[index].texcoord0 = barycentricInterpolation(baryCoord, p.v[0].texcoord0, p.v[1].texcoord0, p.v[2].texcoord0);
#endif
				}
			}
		}
	}

#endif
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
		framebuffer[index] = glm::vec3(0, 0, 0);
		Fragment fragment = fragmentBuffer[index];

		TextureData* diffuse = fragment.dev_diffuseTex;

#if textureenabled

		glm::vec3 lightPos(0.0f, 10.0f, 10.0f);
		glm::vec3 lightDir(glm::normalize(lightPos - fragment.eyePos));

		// diffuse
		if (diffuse != NULL) {

#if bilinearfiltering
			float u = fragment.texcoord0.x * fragment.texWidth;
			float v = fragment.texcoord0.y * fragment.texHeight;
			int u0 = (int)u;
			int v0 = (int)v;
			int u1 = glm::clamp((u0 + 1), 0, fragment.texWidth - 1);
			int v1 = glm::clamp((v0 + 1), 0, fragment.texHeight - 1);
			float udelta = u - u0, vdelta = v - v0;

			int id = u0 + v0 * fragment.texWidth;
			glm::vec3 color00 = glm::vec3(diffuse[3 * id] / 255.f, diffuse[3 * id + 1] / 255.f, diffuse[3 * id + 2] / 255.f);
			id = u0 + v1 * fragment.texWidth;
			glm::vec3 color01 = glm::vec3(diffuse[3 * id] / 255.f, diffuse[3 * id + 1] / 255.f, diffuse[3 * id + 2] / 255.f);
			id = u1 + v0 * fragment.texWidth;
			glm::vec3 color10 = glm::vec3(diffuse[3 * id] / 255.f, diffuse[3 * id + 1] / 255.f, diffuse[3 * id + 2] / 255.f);
			id = u1 + v1 * fragment.texWidth;
			glm::vec3 color11 = glm::vec3(diffuse[3 * id] / 255.f, diffuse[3 * id + 1] / 255.f, diffuse[3 * id + 2] / 255.f);

			glm::vec3 color0 = (1 - udelta) * color00 + udelta * color10;
			glm::vec3 color1 = (1 - udelta) * color01 + udelta * color11;

			framebuffer[index] = (1 - vdelta) * color0 + vdelta * color1;

#else
			int u = fragment.texcoord0.x * fragment.texWidth;
			int v = fragment.texcoord0.y * fragment.texHeight;
			int uvid = u + v * fragment.texWidth;
			framebuffer[index] = glm::vec3(diffuse[3 * uvid] / 255.f, diffuse[3 * uvid + 1] / 255.f, diffuse[3 * uvid + 2] / 255.f);

#endif
		}
		else {
			framebuffer[index] = fragment.color;
		}

#if lambert
		float lambertCoef = glm::dot(fragment.eyeNor, lightDir);
		framebuffer[index] *= glm::clamp(lambertCoef, 0.2f, 1.0f);
#endif

#if blinnphong
		// higher shineness, more obvious and smaller specular area
		glm::vec3 h = lightDir;
		float specular = pow(glm::max(glm::dot(fragment.eyeNor, h), 0.f), 80.f);
		framebuffer[index] += specular * glm::vec3(1.f, 1.f, 1.f);
#endif


#else
		framebuffer[index] = fragment.color;

#endif

#if depthtest
		framebuffer[index] = glm::vec3(fragment.pos.z);
#endif

		if (diffuse != NULL) {

#if normaltest
			framebuffer[index] = (fragment.eyeNor + glm::vec3(1.0f)) / 2.0f;
#endif
		}

	}
}

#pragma endregion stages
//===========================================================================================================================================


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
							//std::vector<double> diffuseColor = mat.values.at("diffuse").number_array;
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


static int curPrimitiveBeginId = 0;
static int blocksize = 128;
/**
 * Perform rasterization.
 */
void rasterize(uchar4 *pbo, const glm::mat4 & MVP, const glm::mat4 & MV, const glm::mat3 MV_normal, std::vector<glm::vec3> &finalimage) {
    int sideLength2d = 8;
    dim3 blockSize2d(sideLength2d, sideLength2d);
    dim3 blockCount2d((width  - 1) / blockSize2d.x + 1,
		(height - 1) / blockSize2d.y + 1);

	// Vertex Process & primitive assembly
	{
		curPrimitiveBeginId = 0;
		dim3 numThreadsPerBlock(blocksize);

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
	checkCUDAError("clear fragment buffer");
	initDepth << <blockCount2d, blockSize2d >> >(width, height, dev_depth);
	checkCUDAError("init depth");

#if timeranalysis
	timer().startGpuTimer();
#endif
	//std::cout << totalNumPrimitives << std::endl;
	// TODO: rasterize
	dim3 numThreadsPerBlock(blocksize < totalNumPrimitives ? blocksize : totalNumPrimitives);
	dim3 numBlocksForPrimitives((totalNumPrimitives + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
	_rasterize << <numBlocksForPrimitives, numThreadsPerBlock >> > (totalNumPrimitives, width, height, dev_primitives, dev_fragmentBuffer, dev_depth);
	checkCUDAError("rasterize");

#if timeranalysis
	timer().endGpuTimer();
#endif

    // Copy depthbuffer colors into framebuffer
	render << <blockCount2d, blockSize2d >> >(width, height, dev_fragmentBuffer, dev_framebuffer);
	checkCUDAError("fragment shader");
    // Copy framebuffer into OpenGL buffer for OpenGL previewing
    sendImageToPBO<<<blockCount2d, blockSize2d>>>(pbo, width, height, dev_framebuffer);
    checkCUDAError("copy render result to pbo");

	cudaMemcpy(&finalimage[0], dev_framebuffer, width * height * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
	checkCUDAError("copy final image");
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
