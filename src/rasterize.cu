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

#define PAINT_NORMALS 1
#define NO_LIGHTING 1
#define WIREFRAME_MODE 0
#define POINT_CLOUD_MODE 0

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

		//glm::vec3 eyePos;	// eye space position used for shading
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


__device__
glm::vec3 genTextureColor(Fragment f) {
	float u = f.texcoord0.x * (f.texWidth - 1);
	float v = f.texcoord0.y * (f.texHeight - 1);

	int tex_x = u;
	int tex_y = v;

	float u_ratio = u - tex_x;
	float v_ratio = v - tex_y;

	glm::vec3 tex_col;
	glm::vec3 temp;

	int tex_idx = 3 * (tex_x + f.texWidth * tex_y);

	// if out of bounds give error color RED
	if (tex_y > (f.texHeight - 1) || tex_y < 0 || tex_x >(f.texWidth - 1) || tex_x < 0) {
		tex_col = glm::vec3(255.0f, 0.0f, 0.0f);
	}
	else {
		bool wrap_x = false; bool wrap_y = false;

		if (tex_y + 1 == f.texHeight) wrap_y = true;
		if (tex_x + 1 == f.texWidth) wrap_x = true;

		// (x, y)
		tex_col.x = f.dev_diffuseTex[tex_idx];
		tex_col.y = f.dev_diffuseTex[tex_idx + 1];
		tex_col.z = f.dev_diffuseTex[tex_idx + 2];

		tex_col *= (1 - u_ratio) * (1 - v_ratio);

		// (x + 1, y)
		if (wrap_x) {
			tex_idx = 3 * tex_y * f.texWidth;
		}
		else {
			tex_idx = tex_idx + 3;
		}
		temp.x = f.dev_diffuseTex[tex_idx];
		temp.y = f.dev_diffuseTex[tex_idx + 1];
		temp.z = f.dev_diffuseTex[tex_idx + 2];

		tex_col += (temp * u_ratio * (1 - v_ratio));

		// (x, y + 1)
		if (wrap_y) {
			tex_idx = 3 * tex_x;
		}
		else {
			tex_idx = 3 * (tex_x + f.texWidth * (tex_y + 1));
		}

		temp.x = f.dev_diffuseTex[tex_idx];
		temp.y = f.dev_diffuseTex[tex_idx + 1];
		temp.z = f.dev_diffuseTex[tex_idx + 2];

		tex_col += (temp * v_ratio * (1 - u_ratio));

		// (x + 1, y + 1)
		if (wrap_x && wrap_y) {
			tex_idx = 0;
		}
		else if (wrap_x) {
			tex_idx = 3 * (f.texWidth * (tex_y + 1));
		}
		else if (wrap_y) {
			tex_idx = tex_idx = 3 * (tex_x + 1);
		}
		else {
			tex_idx = 3 * ((tex_x + 1) + f.texWidth * (tex_y + 1));
		}

		temp.x = f.dev_diffuseTex[tex_idx];
		temp.y = f.dev_diffuseTex[tex_idx + 1];
		temp.z = f.dev_diffuseTex[tex_idx + 2];

		tex_col += (temp * v_ratio * u_ratio);

	}
	tex_col = tex_col / 255.0f;
	return tex_col;
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
		Fragment f = fragmentBuffer[index];

		// lambertian lighting
		float diffuseLight = glm::dot(glm::normalize(f.eyeNor), glm::vec3(0.0f, 1.0f, 0.0f)) + 0.2f; // 0.2f is ambient
		if (diffuseLight > 1) diffuseLight = 1.0f;
		if (diffuseLight < 0) diffuseLight = 0.0f;

		if (NO_LIGHTING) diffuseLight = 1.0f;

		if (PAINT_NORMALS) {
			framebuffer[index] = fragmentBuffer[index].eyeNor * diffuseLight;
		}
		// apply texture color
		else if (f.dev_diffuseTex != NULL) {
			
			framebuffer[index] = genTextureColor(f) * diffuseLight;
		}
		
		else {
			framebuffer[index] = fragmentBuffer[index].color * diffuseLight;
		}

		// TODO: add your fragment shader code here

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

	cudaFree(dev_mutex);
	cudaMalloc(&dev_mutex, width * height * sizeof(int));
	cudaMemset(dev_mutex, 0, width * height * sizeof(int));

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

		VertexOut & vOut = primitive.dev_verticesOut[vid];

		// screen-space pos
		vOut.pos = MVP * glm::vec4(primitive.dev_position[vid], 1.0f);
		vOut.pos /= vOut.pos.w;
		vOut.pos.x = 0.5f * (float)width * (vOut.pos.x + 1.0f);
		vOut.pos.y = 0.5f * (float)height * (1.0f - vOut.pos.y);

		// eye-space pos
		vOut.eyePos = glm::vec3(MV * glm::vec4(primitive.dev_position[vid], 1.0f));

		// eye-space normal
		vOut.eyeNor = MV_normal * primitive.dev_normal[vid];
		vOut.eyeNor = glm::normalize(vOut.eyeNor);

		vOut.dev_diffuseTex = primitive.dev_diffuseTex;
		if (vOut.dev_diffuseTex != NULL) {
			vOut.texcoord0 = primitive.dev_texcoord0[vid];


			vOut.texHeight = primitive.diffuseTexHeight;
			vOut.texWidth = primitive.diffuseTexWidth;
		}

		// TODO: Apply vertex assembly here
		// Assemble all attribute arrays into the primitive array

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

			dev_primitives[pid + curPrimitiveBeginId].primitiveType = Triangle;
		}


		// TODO: other primitive types (point, line)
		else if (primitive.primitiveMode == TINYGLTF_MODE_LINE) {
			pid = iid / (int)primitive.primitiveType;
			dev_primitives[pid + curPrimitiveBeginId].v[iid % (int)primitive.primitiveType]
				= primitive.dev_verticesOut[primitive.dev_indices[iid]];

			dev_primitives[pid + curPrimitiveBeginId].primitiveType = Line;
		}

		else if (primitive.primitiveMode == TINYGLTF_MODE_POINTS) {
			pid = iid;
			dev_primitives[pid + curPrimitiveBeginId].v[0] = primitive.dev_verticesOut[primitive.dev_indices[iid]];

			dev_primitives[pid + curPrimitiveBeginId].primitiveType = Point;
		}
	}

}

__device__ 
void rasterizeTriangle(Primitive p, Fragment* fragBuf, int* depth, int* mutex, int width, int height) {

	Fragment f_true;
	f_true.color = glm::vec3(1.0f); // default white
	f_true.dev_diffuseTex = p.v[0].dev_diffuseTex;

	if (f_true.dev_diffuseTex != NULL) {
		f_true.texWidth = p.v[0].texWidth;
		f_true.texHeight = p.v[0].texHeight;
	}

	// triangle vertex positions
	glm::vec3 tri[3] = { glm::vec3(p.v[0].pos), glm::vec3(p.v[1].pos), glm::vec3(p.v[2].pos) };

	// backface culling
	glm::vec3 normal = glm::cross(tri[1] - tri[0], tri[2] - tri[0]);
	if (glm::dot(normal, glm::vec3(0.0f, 0.0f, 1.0f)) > 0) return;

	AABB aabb = getAABBForTriangle(tri);

	// if completely off screen return
	if (aabb.max.y >= height && aabb.min.y >= height) return;
	if (aabb.max.y < 0 && aabb.min.y < 0) return;

	if (aabb.max.x >= width && aabb.min.x >= width) return;
	if (aabb.max.x < 0 && aabb.min.x < 0) return;

	// clamp bounds
	if (aabb.max.y >= height) aabb.max.y = height - 1;
	if (aabb.min.y < 0) aabb.min.y = 0;
	if (aabb.max.x >= width) aabb.max.x = width - 1;
	if (aabb.min.x < 0) aabb.min.x = 0;

	int f_idx;

	glm::vec3 bary;
	bool inside = false;

	int z;
	float z_float;
	float w;

	bool isSet = false;

	for (int y = aabb.min.y; y <= aabb.max.y; y++) {
		for (int x = aabb.min.x; x <= aabb.max.x; x++) {
			f_idx = y * width + x;
			// identify if in bounds of triangle
			bary = calculateBarycentricCoordinate(tri, glm::vec2(x, y));
			inside = isBarycentricCoordInBounds(bary);

			// depth buffer
			z_float = getZAtCoordinate(bary, tri);
			if (z_float < 0 || z_float > 1) continue;
			z = z_float * INT_MAX;

			w = 1.0f / ((bary.x / tri[0].z) + (bary.y / tri[1].z) + (bary.z / tri[2].z));

			if (inside) {
				isSet = false;
				do {
					isSet = (atomicCAS(mutex + f_idx, 0, 1) == 0);
					if (isSet) {
						// Critical section
						if (z < depth[f_idx]) {

							depth[f_idx] = z;
							fragBuf[f_idx] = f_true;

							// generate interpolated attributes
							if (p.v[0].dev_diffuseTex != NULL) {
								fragBuf[f_idx].texcoord0 = w * ((p.v[0].texcoord0 * bary.x / tri[0].z) + (p.v[1].texcoord0 * bary.y / tri[1].z) + (p.v[2].texcoord0 * bary.z / tri[2].z));
							}
							fragBuf[f_idx].eyeNor = w * ((p.v[0].eyeNor * bary.x / tri[0].z) + (p.v[1].eyeNor * bary.y / tri[1].z) + (p.v[2].eyeNor * bary.z / tri[2].z));
						}
					}
					if (isSet) {
						mutex[f_idx] = 0;
					}
				} while (!isSet);
			}
		}
	}

}

__device__
void rasterizePoint(VertexOut v, Fragment* fragBuf, int* depth, int* mutex, int width) {
	Fragment f;
	f.color = glm::vec3(1.0f);
	f.dev_diffuseTex = v.dev_diffuseTex;
	if (f.dev_diffuseTex != NULL) {
		f.texcoord0 = v.texcoord0;
		f.texHeight = v.texHeight;
		f.texWidth = v.texWidth;
	}
	f.eyeNor = v.eyeNor;
	int f_idx = (int)v.pos.y * width + (int)v.pos.x;

	bool isSet = false;

	do {
		isSet = (atomicCAS(mutex + f_idx, 0, 1) == 0);
		if (isSet) {
			// Critical section
			if (v.pos.z < depth[f_idx]) {

				depth[f_idx] = v.pos.z;
				fragBuf[f_idx] = f;
			}
		}
		if (isSet) {
			mutex[f_idx] = 0;
		}
	} while (!isSet);

}

__device__
void rasterizeLine(VertexOut v1, VertexOut v2, Fragment* fragBuf, int* depth, int* mutex, int width, int height) {
	AABB aabb;
	aabb.min = glm::vec3(min(v1.pos.x, v2.pos.x), min(v1.pos.y, v2.pos.y), min(v1.pos.z, v2.pos.z));
	aabb.max = glm::vec3(max(v1.pos.x, v2.pos.x), max(v1.pos.y, v2.pos.y), max(v1.pos.z, v2.pos.z));

	// if completely off screen return
	if (aabb.max.y >= height && aabb.min.y >= height) return;
	if (aabb.max.y < 0 && aabb.min.y < 0) return;

	if (aabb.max.x >= width && aabb.min.x >= width) return;
	if (aabb.max.x < 0 && aabb.min.x < 0) return;

	// clamp bounds
	if (aabb.max.y >= height) aabb.max.y = height - 1;
	if (aabb.min.y < 0) aabb.min.y = 0;
	if (aabb.max.x >= width) aabb.max.x = width - 1;
	if (aabb.min.x < 0) aabb.min.x = 0;

	int x;
	float z;

	float m = (v2.pos.y - v1.pos.y) / (v2.pos.x - v1.pos.x);
	float a;

	Fragment f;
	f.color = glm::vec3(1.0f);

	int f_idx;

	for (int y = aabb.min.y; y <= aabb.max.y; y++) {
		//get point x on line
		x = (int)((y - v1.pos.y) / m) + (int)v1.pos.x;

		// check bounds
		if (x > aabb.max.x || x < aabb.min.x) continue;

		a = (x - v1.pos.x) / (v2.pos.x - v1.pos.x);
		f_idx = y * width + x;

		z = 1.0f / ((a / v1.pos.z) + ((1-a) / v2.pos.z));

		f.dev_diffuseTex = v1.dev_diffuseTex;
		if (f.dev_diffuseTex != NULL) {
			f.texcoord0 = z * ((a * v1.texcoord0 / v1.pos.z) + ((1 - a) * v2.texcoord0 / v2.pos.z));
			f.texHeight = v1.texHeight;
			f.texWidth = v1.texWidth;
		}
		f.eyeNor = z * ((a * v1.eyeNor / v1.pos.z) + ((1 - a) * v2.eyeNor / v2.pos.z));

		bool isSet = false;

		do {
			isSet = (atomicCAS(mutex + f_idx, 0, 1) == 0);
			if (isSet) {
				// Critical section
				if (z < depth[f_idx]) {

					depth[f_idx] = z;
					fragBuf[f_idx] = f;
				}
			}
			if (isSet) {
				mutex[f_idx] = 0;
			}
		} while (!isSet);
	}

}

__global__
void _rasterizer(int num_prim, Primitive* primitives, Fragment* fragBuf, int* depth, int* mutex, int width, int height) {
	// index id
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (idx > num_prim) return;

	Primitive p = primitives[idx];

	if (POINT_CLOUD_MODE && p.primitiveType == Triangle) {
		for (int i = 0; i < 3; i++) {
			if (p.v[i].pos.x >= width || p.v[i].pos.y >= height) continue;
			if (p.v[i].pos.x < 0 || p.v[i].pos.y < 0) continue;
			rasterizePoint(p.v[i], fragBuf, depth, mutex, width);
		}

	}
	else if (WIREFRAME_MODE && p.primitiveType == Triangle) {
		rasterizeLine(p.v[0], p.v[1], fragBuf, depth, mutex, width, height);
		rasterizeLine(p.v[0], p.v[2], fragBuf, depth, mutex, width, height);
		rasterizeLine(p.v[1], p.v[2], fragBuf, depth, mutex, width, height);
	}
	else if (p.primitiveType == Triangle) {
		rasterizeTriangle(p, fragBuf, depth, mutex, width, height);
	}
	else if (p.primitiveType == Point) {
		if (p.v[0].pos.x >= width || p.v[0].pos.y >= height) return;
		if (p.v[0].pos.x < 0 || p.v[0].pos.y < 0) return;
		rasterizePoint(p.v[0], fragBuf, depth, mutex, width);
	}
	else if (p.primitiveType == Line) {
		rasterizeLine(p.v[0], p.v[1], fragBuf, depth, mutex, width, height);
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
	dim3 num_blocks((totalNumPrimitives + 128 - 1) / 128);
	_rasterizer << <num_blocks, 128 >> > (totalNumPrimitives, dev_primitives, dev_fragmentBuffer, dev_depth, dev_mutex, width, height);


	// Copy depthbuffer colors into framebuffer
	render << <blockCount2d, blockSize2d >> > (width, height, dev_fragmentBuffer, dev_framebuffer);
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
