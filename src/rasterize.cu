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

#define TEXTURE 1
#define CORRECT_INTERP 1 
#define TEXTURE_BILINEAR 0
#define DEBUG_NORM 0
#define DEBUG_Z 0
#define DOWNSCALERATIO 3

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

	//render Mode
	enum RenderMode { Triangles, Wireframe, Points };

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
		// int texWidth, texHeight;
		// ...
		 int diffuseTexWidth, diffuseTexHeight;
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
		// VertexAttributeTexcoord texcoord0;
		TextureData* dev_diffuseTex;

		int uvStart;
		glm::vec2 uvBilinear;
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

//used for atomic depth writing
static int * dev_mutex = NULL;
//used for bloom effect post processing
static glm::vec3 * dev_framebufferAux = NULL;
static glm::vec3 * dev_framebufferDownScaled = NULL;
static glm::vec3 * dev_framebufferDownScaledAux = NULL;

/**
 * Kernel that writes the image to the OpenGL PBO directly.
 */
__global__ 
void sendImageToPBO(uchar4 *pbo, int w, int h, glm::vec3 *image,
	//used for bloomEffect
	int framebufferEdgeOffset,
	int downScaleW,
	int downScaleRatio
)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * w);

    if (x < w && y < h) {
		//frame buffer id
		int fid; 
		if (downScaleRatio == 1)
		{
			fid = x + (y * w);
		}
		else
		{
			fid = (x / downScaleRatio) + framebufferEdgeOffset
				+ ((y / downScaleRatio) + framebufferEdgeOffset) * (downScaleW + 2 * framebufferEdgeOffset);
		}
        glm::vec3 color;
        //color.x = glm::clamp(image[index].x, 0.0f, 1.0f) * 255.0;
        //color.y = glm::clamp(image[index].y, 0.0f, 1.0f) * 255.0;
        //color.z = glm::clamp(image[index].z, 0.0f, 1.0f) * 255.0;
		color.x = glm::clamp(image[fid].x, 0.0f, 1.0f) * 255.0;
		color.y = glm::clamp(image[fid].y, 0.0f, 1.0f) * 255.0;
		color.z = glm::clamp(image[fid].z, 0.0f, 1.0f) * 255.0;
        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}


__device__ glm::vec3 sampleFromTexture(TextureData* dev_diffuseTex, int uvStart) {
	return glm::vec3(dev_diffuseTex[uvStart + 0],
		dev_diffuseTex[uvStart + 1],
		dev_diffuseTex[uvStart + 2]) / 255.f;
}

__device__ glm::vec3 sampleFromTextureBilinear(TextureData* dev_diffuseTex, glm::vec2 uv, int width, int height) {
	// top right corner of grid cell
	int u = uv.x;
	int v = uv.y;

	// bottom right corner of grid cell
	int u1 = glm::clamp(u + 1, 0, width - 1);
	int v1 = glm::clamp(v + 1, 0, height - 1);

	// sample all 4 colors using indices to sample from texture
	int id00 = (u + v * width) * 3;
	int id01 = (u + v1 * width) * 3;
	int id10 = (u1 + v * width) * 3;
	int id11 = (u1 + v1 * width) * 3;
	glm::vec3 c00 = sampleFromTexture(dev_diffuseTex, id00);
	glm::vec3 c01 = sampleFromTexture(dev_diffuseTex, id01);
	glm::vec3 c10 = sampleFromTexture(dev_diffuseTex, id10);
	glm::vec3 c11 = sampleFromTexture(dev_diffuseTex, id11);

	// lerp horizontally using x fraction
	glm::vec3 lerp1 = glm::mix(c00, c01, (float)uv.x - (float)u);
	glm::vec3 lerp2 = glm::mix(c10, c11, (float)uv.x - (float)u);

	// return lerped color vertically using y fraction
	return  glm::mix(lerp1, lerp2, (float)uv.y - (float)v);
}

/** 
* Writes fragment colors to the framebuffer
*/
__global__
void render(int w, int h, Fragment *fragmentBuffer, glm::vec3 *framebuffer, 
	glm::vec3 lightPos, //for lighting calc
	int renderMode, //switch rendermode
	int framebufferEdgeOffset //bloom Effect
	)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < w && y < h) {
		int index = x + (y * w);

		//framebuffer[index] = fragmentBuffer[index].color;

		////// TODO: add your fragment shader code here

		Fragment f = fragmentBuffer[index];
		if (renderMode == RenderMode::Triangles)
		{

#if TEXTURE
			if (f.dev_diffuseTex) {
#if TEXTURE_BILINEAR
				f.color = sampleFromTextureBilinear(f.dev_diffuseTex, f.uvBilinear, f.texWidth, f.texHeight);
#else
				f.color = sampleFromTexture(f.dev_diffuseTex, f.uvStart);
#endif
			}
			else {
				f.color = glm::vec3(0.f);
			}
#endif	

#if !DEBUG_Z && !DEBUG_NORM
			glm::vec3 lightDir = glm::normalize(lightPos - f.eyePos);
			float lambert = glm::dot(lightDir, f.eyeNor);
			float ambient = 0.1f;
			float light_exp = 3.0f;
			lambert = lambert * light_exp + ambient;
			framebuffer[index] = f.color * lambert;
#else
			framebuffer[index] = f.color;
#endif
		}
		//for wireframe and points render mode, pass the color
		else if (renderMode == RenderMode::Wireframe || renderMode == RenderMode::Points)
		{
			framebuffer[index] = f.color;
		}
	}
}

// bloom effect : horizontal Gaussian blur
__global__
void gaussianBlurHorizon(int w, int h, glm::vec3 * framebufferIn,
	glm::vec3 * framebufferOut, int framebufferEdgeOffset)
{
	//using shared memory to store shared frame buffer
	__shared__ glm::vec3 framebufferIn_shared[144]; //144 = 8 * (5 + 8 + 5)
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	if (x < w && y < h)
	{
		//framebuffer index
		int fid = (x + framebufferEdgeOffset) + ((y + framebufferEdgeOffset) * (w + 2 * framebufferEdgeOffset));
		framebufferOut[fid] = glm::vec3(0.0f);
		//read framebufferIn into shared memory
		//18 = 5 * 2 + 8
		int index = threadIdx.y * 18 + threadIdx.x + 5;
		if (threadIdx.x == 0)
		{
			framebufferIn_shared[index - 5] = framebufferIn[fid - 5];
			framebufferIn_shared[index - 4] = framebufferIn[fid - 4];
			framebufferIn_shared[index - 3] = framebufferIn[fid - 3];
			framebufferIn_shared[index - 2] = framebufferIn[fid - 2];
			framebufferIn_shared[index - 1] = framebufferIn[fid - 1];
		}
		if (threadIdx.x == blockDim.x - 1)
		{
			framebufferIn_shared[index + 5] = framebufferIn[fid + 5];
			framebufferIn_shared[index + 4] = framebufferIn[fid + 4];
			framebufferIn_shared[index + 3] = framebufferIn[fid + 3];
			framebufferIn_shared[index + 2] = framebufferIn[fid + 2];
			framebufferIn_shared[index + 1] = framebufferIn[fid + 1];
		}
		framebufferIn_shared[index] = framebufferIn[fid];

		__syncthreads();

		//apply horizontal gaussian blur and write to output
		framebufferOut[fid] += framebufferIn_shared[index - 5] * 0.0093f;
		framebufferOut[fid] += framebufferIn_shared[index - 5] * 0.0093f;
		framebufferOut[fid] += framebufferIn_shared[index - 4] * 0.028002f;
		framebufferOut[fid] += framebufferIn_shared[index - 3] * 0.065984f;
		framebufferOut[fid] += framebufferIn_shared[index - 2] * 0.121703f;
		framebufferOut[fid] += framebufferIn_shared[index - 1] * 0.175713f;
		framebufferOut[fid] += framebufferIn_shared[index] * 0.198596f;
		framebufferOut[fid] += framebufferIn_shared[index + 1] * 0.175713f;
		framebufferOut[fid] += framebufferIn_shared[index + 2] * 0.121703f;
		framebufferOut[fid] += framebufferIn_shared[index + 3] * 0.065984f;
		framebufferOut[fid] += framebufferIn_shared[index + 4] * 0.028002f;
		framebufferOut[fid] += framebufferIn_shared[index + 5] * 0.0093f;
	}
}

//bloom effect : vertical gussian blur
__global__
void gaussianBlurVert(int w, int h, glm::vec3 * framebufferIn,
	glm::vec3 * framebufferOut, int framebufferEdgeOffset)
{
	//Same as horizontal

    //array size should be blocksize.x * (framebufferEdgeOffset + blocksize.y + framebufferEdgeOffset)
    // 8 * (5 + 8 + 5) -> 144
	__shared__ glm::vec3 framebuffer_in_shared[144];

	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	if (x < w && y < h) {
		//framebuffer index
		int fid = (x + framebufferEdgeOffset) + ((y + framebufferEdgeOffset) * (w + 2 * framebufferEdgeOffset));
		framebufferOut[fid] = glm::vec3(0.f);

		int numOfelementsOneRow = w + 2 * framebufferEdgeOffset;

		// blocksize.x
		// 8
		int index = (threadIdx.y + 5) * 8 + threadIdx.x;
		//write into shared memory
		if (threadIdx.y == 0) {
			// 40, 32, 24... -> 5 * blocksize.x          
			framebuffer_in_shared[index - 40] = framebufferIn[fid - 5 * numOfelementsOneRow];
			framebuffer_in_shared[index - 32] = framebufferIn[fid - 4 * numOfelementsOneRow];
			framebuffer_in_shared[index - 24] = framebufferIn[fid - 3 * numOfelementsOneRow];
			framebuffer_in_shared[index - 16] = framebufferIn[fid - 2 * numOfelementsOneRow];
			framebuffer_in_shared[index - 8] = framebufferIn[fid - 1 * numOfelementsOneRow];
		}
		if (threadIdx.y == blockDim.y - 1) {
			framebuffer_in_shared[index + 8] = framebufferIn[fid + 1 * numOfelementsOneRow];
			framebuffer_in_shared[index + 16] = framebufferIn[fid + 2 * numOfelementsOneRow];
			framebuffer_in_shared[index + 24] = framebufferIn[fid + 3 * numOfelementsOneRow];
			framebuffer_in_shared[index + 32] = framebufferIn[fid + 4 * numOfelementsOneRow];
			framebuffer_in_shared[index + 40] = framebufferIn[fid + 5 * numOfelementsOneRow];
		}
		framebuffer_in_shared[index] = framebufferIn[fid];

		__syncthreads();

		//apply vertical gaussian blur, write into output frames
		framebufferOut[fid] += framebuffer_in_shared[index - 40] * 0.0093f;
		framebufferOut[fid] += framebuffer_in_shared[index - 32] * 0.028002f;
		framebufferOut[fid] += framebuffer_in_shared[index - 24] * 0.065984f;
		framebufferOut[fid] += framebuffer_in_shared[index - 16] * 0.121703f;
		framebufferOut[fid] += framebuffer_in_shared[index - 8] * 0.175713f;
		framebufferOut[fid] += framebuffer_in_shared[index] * 0.198596f;
		framebufferOut[fid] += framebuffer_in_shared[index + 8] * 0.175713f;
		framebufferOut[fid] += framebuffer_in_shared[index + 16] * 0.121703f;
		framebufferOut[fid] += framebuffer_in_shared[index + 24] * 0.065984f;
		framebufferOut[fid] += framebuffer_in_shared[index + 32] * 0.028002f;
		framebufferOut[fid] += framebuffer_in_shared[index + 40] * 0.0093f;
	}
}

//downSample kernel
__global__
void downScaleSample(int downScaleW, int downScaleH, int downScaleRatio,
	int w, int h,
	glm::vec3 * downScale_framebuffer, glm::vec3* framebuffer,
	int framebufferEdgeOffset)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < downScaleW && y < downScaleH)
	{
		int index = (x + framebufferEdgeOffset) + ((y + framebufferEdgeOffset) * (downScaleW + 2 * framebufferEdgeOffset));
		glm::vec3& f_col = downScale_framebuffer[index];
		f_col = glm::vec3(0.0f);

		float sampleCount = (float)downScaleRatio * (float)downScaleRatio;
		int oriFramebufferX, oriFramebufferY;
		int oriFramebufferInd;
		//down sample and calculate average
		for (int i = 0; i < downScaleRatio; i++)
		{
			for (int j = 0; j < downScaleRatio; j++)
			{
				oriFramebufferX = x * downScaleRatio + i;
				oriFramebufferY = y * downScaleRatio + j;
				oriFramebufferX = glm::clamp(oriFramebufferX, 0, w - 1);
				oriFramebufferY = glm::clamp(oriFramebufferY, 0, h - 1);
				oriFramebufferInd = oriFramebufferX + (oriFramebufferY * w);
				f_col += framebuffer[oriFramebufferInd];
			}
		}
		f_col /= sampleCount;
	}
}

//bloom effect: brightness filter
__global__
void brightnessFilter(int w, int h, glm::vec3 * framebufferIn, glm::vec3 * framebufferOut)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	if (x < w && y < h)
	{
		int index = x + y * w;
		// input frame buffer at index, not reference
		glm::vec3 f_in = framebufferIn[index];
		//calculate brightness of this frame buffer
		float brightness = f_in.r * 0.2126f + f_in.g * 0.7152f + f_in.b * 0.0722f;
		//framebufferOut[index] = brightness * f_in;
		brightness /= 2.0f;
		framebufferOut[index] = glm::vec3(brightness);
	}
}

//bloom effect: final stage, combine buffers
__global__
void bloomCombineFramebuffers(int w, int h,
	glm::vec3 * mainFramebuffer, glm::vec3 * sideFramebuffer,
	glm::vec3 * framebufferOut,
	int sideFramebuffer_downScaleW,
	int sideFramebuffer_downScaleRatio, int framebufferEdgeOffset)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	if (x < w && y < h)
	{
		int mainIdx = x + (y * w);
		glm::vec3 main_col = mainFramebuffer[mainIdx];
		int sideIdx = ((x / sideFramebuffer_downScaleRatio) + framebufferEdgeOffset) +
			(((y / sideFramebuffer_downScaleRatio) + framebufferEdgeOffset) *
			(sideFramebuffer_downScaleW + 2 * framebufferEdgeOffset));
		glm::vec3 side_col = sideFramebuffer[sideIdx];

		//combination factor
		float k = 1.0f;
		//comine color and write to output
		framebufferOut[mainIdx] = main_col + k * side_col;
	}
}

//edge width in gaussian blur
int bloomGBedge = 5;
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
    //initialize auxilliary frame buffers for bloom effect
	cudaFree(dev_framebufferAux);
	cudaMalloc(&dev_framebufferAux, width * height * sizeof(glm::vec3));
	cudaMemset(dev_framebufferAux, 0, width * height * sizeof(glm::vec3));

	int downScaleRatio = DOWNSCALERATIO;
	cudaFree(dev_framebufferDownScaled);
	cudaMalloc(&dev_framebufferDownScaled,
		((width / downScaleRatio) + 2 * bloomGBedge) * ((height / downScaleRatio) + 2 * bloomGBedge) * sizeof(glm::vec3));
	cudaMemset(dev_framebufferDownScaled, 0,
		((width / downScaleRatio) + 2 * bloomGBedge) * ((height / downScaleRatio) + 2 * bloomGBedge) * sizeof(glm::vec3));
	
	cudaFree(dev_framebufferDownScaledAux);
	cudaMalloc(&dev_framebufferDownScaledAux,
		((width / downScaleRatio) + 2 * bloomGBedge) * ((height / downScaleRatio) + 2 * bloomGBedge) * sizeof(glm::vec3));
	cudaMemset(dev_framebufferDownScaledAux, 0,
		((width / downScaleRatio) + 2 * bloomGBedge) * ((height / downScaleRatio) + 2 * bloomGBedge) * sizeof(glm::vec3));



	cudaFree(dev_depth);
	cudaMalloc(&dev_depth, width * height * sizeof(int));

	cudaFree(dev_mutex);
	cudaMalloc(&dev_mutex, width * height * sizeof(Fragment));
	cudaMemset(dev_mutex, 0, width * height * sizeof(Fragment));
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
	int width, int height,
	glm::mat4 autoRotateMat4 //to enable auto rotation of the object
	)
{

	// vertex id
	int vid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (vid < numVertices) {

		// TODO: Apply vertex transformation here
		// Multiply the MVP matrix for each vertex position, this will transform everything into clipping space
		// Then divide the pos by its w element to transform into NDC space
		// Finally transform x and y to viewport space

		//use shared memory to apply auto rotation
		__shared__ glm::mat4 _autoRotateMat4;
		if (threadIdx.x == 0)
		{
			_autoRotateMat4 = autoRotateMat4;
		}
		__syncthreads();

		// TODO: Apply vertex assembly here
		// Assemble all attribute arraies into the primitive array
		VertexOut& thisDevVertexOut = primitive.dev_verticesOut[vid];
		//multiply model-view-projective matrix
		glm::vec4 worldSpacePos = MVP * _autoRotateMat4* glm::vec4(primitive.dev_position[vid], 1.0f);
		//Projective divide
		glm::vec4 NDCpos = worldSpacePos * (1.0f / worldSpacePos.w);
		//transform into pixels
		glm::vec4 PixelPos = glm::vec4(
			(NDCpos.x + 1.0f) * (float)width / 2.0f,
			(1.0f - NDCpos.y) * (float)height / 2.0f,
			NDCpos.z,
			NDCpos.w);
		//write into vertexout struct
		thisDevVertexOut.pos = PixelPos;
		//Eye space pos
		glm::vec3 eyeSpacePos = glm::vec3(MV * _autoRotateMat4 *glm::vec4(primitive.dev_position[vid], 1.0f));
		thisDevVertexOut.eyePos = eyeSpacePos;
		//Eye space normal
		glm::vec3 eyeSpaceNormal = glm::normalize(MV_normal * glm::mat3(_autoRotateMat4) *primitive.dev_normal[vid]);
		thisDevVertexOut.eyeNor = eyeSpaceNormal;
		//texture coords
		if (primitive.dev_texcoord0 != NULL)
		{
			thisDevVertexOut.texcoord0 = primitive.dev_texcoord0[vid];
		}
		else
		{
			thisDevVertexOut.texcoord0 = glm::vec2(0.0f, 0.0f);
		}
		//diffuse texture
		if (primitive.dev_diffuseTex != NULL)
		{
			thisDevVertexOut.dev_diffuseTex = primitive.dev_diffuseTex;
			thisDevVertexOut.diffuseTexHeight = primitive.diffuseTexHeight;
			thisDevVertexOut.diffuseTexWidth = primitive.diffuseTexWidth;
		}
		//else assign sequential RGB color
		else
		{
			thisDevVertexOut.col = glm::vec3(0.f);
			thisDevVertexOut.col[vid % 3] += 0.6f;
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

////helper function: fill specific fragment buffer
//__device__
//void fillFragmentBufferRoutine(Fragment& thisFragment,
//	glm::vec3 p, glm::vec3 p1, glm::vec3 p2, glm::vec3 p3,
//	VertexOut v1, VertexOut v2, VertexOut v3)
//{
//	int diffuseTexWidth = v1.diffuseTexWidth;
//	int diffuseTexHeight = v1.diffuseTexHeight;
//	TextureData* textureData = v1.dev_diffuseTex;
//
//	//positions
//	glm::vec3 eyeSpacePos_interpolated = vec3AttriInterpolate(
//		p, p1, p2, p3,
//		v1.eyePos, v2.eyePos, v3.eyePos
//	);
//	thisFragment.eyePos = eyeSpacePos_interpolated;
//
//	//normals
//	glm::vec3 eyeSpaceNor_interpolated = vec3AttriInterpolate(
//		p, p1, p2, p3,
//		v1.eyeNor, v2.eyeNor, v3.eyeNor
//	);
//	thisFragment.eyeNor = glm::normalize(eyeSpaceNor_interpolated);
//
//	//uv coordinates
//	glm::vec2 uv_interpolated = vec2AttriInterpolate(
//		p, p1, p2, p3,
//		v1.texcoord0, v2.texcoord0, v3.texcoord0
//	);
//	//get texture and color the fragment
//	if (textureData != NULL)
//	{
//		//read texture and color the fragment with it
//		glm::ivec2 texSpaceCoord = glm::ivec2(diffuseTexWidth * uv_interpolated.x,
//			diffuseTexHeight * uv_interpolated.y);
//		int texIdx = texSpaceCoord.x + diffuseTexWidth * texSpaceCoord.y;
//
//		//read from color channels
//		int textChannelsNum = 3;
//		TextureData r = textureData[texIdx * textChannelsNum];
//		TextureData g = textureData[texIdx * textChannelsNum + 1];
//		TextureData b = textureData[texIdx * textChannelsNum + 2];
//
//		thisFragment.color = glm::vec3((float)r / 255.0f,
//			(float)g / 255.0f,
//			(float)b / 255.0f);
//		//thisFragment.color = thisFragment.eyeNor;
//	}
//	else
//	{
//		thisFragment.color = glm::vec3(1.0f, 1.0f, 1.0f);
//	}
//
//}



/**
 * fill in wireframe mode
 */
__device__
void rasterizerFillWireFrame(
	Fragment* fragmentBuffer, int* depth,
	glm::vec3 p1, glm::vec3 p2, glm::vec3 p3,
	int w, int h)
{

	//2D Bounding Box
	float xMin = fminf(p1.x, fminf(p2.x, p3.x));
	float xMax = fmaxf(p1.x, fmaxf(p2.x, p3.x));
	float yMin = fminf(p1.y, fminf(p2.y, p3.y));
	float yMax = fmaxf(p1.y, fmaxf(p2.y, p3.y));

	//check bounding box with screen size
	//get start and end indices in pixel from bounding box
	float xStart = xMin < 0 ? 0 : (int)glm::floor(xMin);
	float xEnd = xMax > w ? w : (int)glm::ceil(xMax);

	float yStart = yMin < 0 ? 0 : (int)glm::floor(yMin);
	float yEnd = yMax > h ? h : (int)glm::ceil(yMax);

	glm::vec3 tris[3];
	tris[0] = p1;
	tris[1] = p2;
	tris[2] = p3;

	int fragmentIdx;

	float lineThickness = 0.08f;
	glm::vec3 wireColor = glm::vec3(0.85f, 0.85f, 0.35f);


	//scan pixel lines to fill the current primitive(triangle)
	for (int i = yStart; i <= yEnd; i++)
	{
		for (int j = xStart; j <= xEnd; j++)
		{
			glm::vec3 barycentricCoord = calculateBarycentricCoordinate(tris, glm::vec2(j, i));
			if (glm::abs(barycentricCoord.x) < lineThickness)
			{
				if (barycentricCoord.y >= 0.0f && barycentricCoord.y <= 1.0f
					&&barycentricCoord.z >= 0.0f && barycentricCoord.z <= 1.0f)
				{
					fragmentIdx = j + (i * w);
					fragmentBuffer[fragmentIdx].color = wireColor;
				}
			}
			if (glm::abs(barycentricCoord.y) < lineThickness)
			{
				if (barycentricCoord.x >= 0.0f && barycentricCoord.x <= 1.0f
					&&barycentricCoord.z >= 0.0f && barycentricCoord.z <= 1.0f)
				{
					fragmentIdx = j + (i * w);
					fragmentBuffer[fragmentIdx].color = wireColor;
				}
			}
			if (glm::abs(barycentricCoord.z) < lineThickness)
			{
				if (barycentricCoord.y >= 0.0f && barycentricCoord.y <= 1.0f
					&&barycentricCoord.x >= 0.0f && barycentricCoord.x <= 1.0f)
				{
					fragmentIdx = j + (i * w);
					fragmentBuffer[fragmentIdx].color = wireColor;
				}
			}
			
		}

	}
}

/**
 * fill in point mode
 */
__device__
void rasterizerFillPoints(
	Fragment* fragmentBuffer, int* depth,
	glm::vec3 p1, glm::vec3 p2, glm::vec3 p3,
	int w, int h)
{

	//2D Bounding Box
	float xMin = fminf(p1.x, fminf(p2.x, p3.x));
	float xMax = fmaxf(p1.x, fmaxf(p2.x, p3.x));
	float yMin = fminf(p1.y, fminf(p2.y, p3.y));
	float yMax = fmaxf(p1.y, fmaxf(p2.y, p3.y));

	//check bounding box with screen size
	//get start and end indices in pixel from bounding box
	float xStart = xMin < 0 ? 0 : (int)glm::floor(xMin);
	float xEnd = xMax > w ? w : (int)glm::ceil(xMax);

	float yStart = yMin < 0 ? 0 : (int)glm::floor(yMin);
	float yEnd = yMax > h ? h : (int)glm::ceil(yMax);

	glm::vec3 tris[3];
	tris[0] = p1;
	tris[1] = p2;
	tris[2] = p3;

	int fragmentIdx;

	float pointThickness = 0.08f;
	glm::vec3 pointColor = glm::vec3(0.35f, 0.85f, 0.85f);


	//scan pixel lines to fill the current primitive(triangle)
	for (int i = yStart; i <= yEnd; i++)
	{
		for (int j = xStart; j <= xEnd; j++)
		{
			glm::vec3 barycentricCoord = calculateBarycentricCoordinate(tris, glm::vec2(j, i));
			
			if (glm::abs(barycentricCoord.x - 1.0f) < pointThickness)
			{
				if (glm::abs(barycentricCoord.y) < pointThickness 
					&&glm::abs(barycentricCoord.z) < pointThickness)
				{
					fragmentIdx = j + (i * w);
					fragmentBuffer[fragmentIdx].color = pointColor;
				}
			}
			if (glm::abs(barycentricCoord.y - 1.0f) < pointThickness)
			{
				if (glm::abs(barycentricCoord.x) < pointThickness
					&&glm::abs(barycentricCoord.z) < pointThickness)
				{
					fragmentIdx = j + (i * w);
					fragmentBuffer[fragmentIdx].color = pointColor;
				}
			}
			if (glm::abs(barycentricCoord.z - 1.0f) < pointThickness)
			{
				if (glm::abs(barycentricCoord.y) < pointThickness
					&&glm::abs(barycentricCoord.x) < pointThickness)
				{
					fragmentIdx = j + (i * w);
					fragmentBuffer[fragmentIdx].color = pointColor;
				}
			}

		}

	}
}

/**
* fill fragment method, actually control over different renderModes
* call this function in rasterize function
*/
__global__
void rasterizeFill(int numPrimitives, int curPrimitiveBeginId, Primitive* primitives,
	Fragment* fragmentBuffer, int * depth, int w, int h, int renderMode, int* dev_mutex)
{
	int primitiveIdx = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (primitiveIdx < numPrimitives)
	{
		Primitive& thisPrimitive = primitives[primitiveIdx + curPrimitiveBeginId];
		glm::vec3 p1(thisPrimitive.v[0].pos[0], thisPrimitive.v[0].pos[1], thisPrimitive.v[0].pos[2]);
		glm::vec3 p2(thisPrimitive.v[1].pos[0], thisPrimitive.v[1].pos[1], thisPrimitive.v[1].pos[2]);
		glm::vec3 p3(thisPrimitive.v[2].pos[0], thisPrimitive.v[2].pos[1], thisPrimitive.v[2].pos[2]);
	
		//Switch between different rendermodes
		if (renderMode == RenderMode::Triangles)
		{
			//rasterizerFillTriangle(thisPrimitive,
			//	fragmentBuffer, depth, p1, p2, p3, w, h,dev_mutex);

			////2D Bounding Box
			//float xMin = fminf(p1.x, fminf(p2.x, p3.x));
			//float xMax = fmaxf(p1.x, fmaxf(p2.x, p3.x));
			//float yMin = fminf(p1.y, fminf(p2.y, p3.y));
			//float yMax = fmaxf(p1.y, fmaxf(p2.y, p3.y));

			////check bounding box with screen size
			////get start and end indices in pixel from bounding box
			//float xStart = xMin < 0 ? 0 : (int)glm::floor(xMin);
			//float xEnd = xMax > w ? w : (int)glm::ceil(xMax);

			//float yStart = yMin < 0 ? 0 : (int)glm::floor(yMin);
			//float yEnd = yMax > h ? h : (int)glm::ceil(yMax);

			glm::vec3 tri[3] = { p1,p2,p3 };
			AABB boundingBox = getAABBForTriangleWithClamp(tri, (float)w, (float)h);
			float xStart = boundingBox.min.x;
			float xEnd = boundingBox.max.x;

			float yStart = boundingBox.min.y;
			float yEnd = boundingBox.max.y;
			//scan pixel lines to fill the current primitive(triangle)
			for (int i = yStart; i <= yEnd; i++)
			{
				for (int j = xStart; j <= xEnd; j++)
				{
					// test if the pos is inside the current triangle
					glm::vec3 baryCoords = calculateBarycentricCoordinate(tri, glm::vec2(j, i));
					if (isBarycentricCoordInBounds(baryCoords))
					{

					
					/*if (isPosInTriangle(glm::vec3(j, i, 0.f), p1, p2, p3))
					{*/
						//interpolate depth 
						//depth value is in eye pace
						//const float z_interpolated = depthInterpolate(glm::vec3(j, i, 0.f), p1, p2, p3);
						const float z_interpolated = getZAtCoordinate(baryCoords, tri);
						//CAUTIOUS(basic)
						//We need to use atomic function to write depth value
						//But it only supports integers, so we multiply it by a large number to get as many digits as we can
						//Note: int32 is bwtween -2^31 and 2^31, which is 2147483648
						const int z_rounded = z_interpolated * INT_MAX;

						//atomic depth writing
						int fragmentIdx = j + (i * w);

						bool isSet;
						do {
							isSet = (atomicCAS(&dev_mutex[fragmentIdx], 0, 1) == 0);
							if (isSet) {
								if (z_rounded < depth[fragmentIdx])
								{
									depth[fragmentIdx] = z_rounded;
									//The pos of current interested fragment
									glm::vec3 p((float)j, (float)i, z_interpolated);

									//fillFragmentBufferRoutine(fragmentBuffer[fragmentIdx],
									//	p, p1, p2, p3,
									//	thisPrimitive.v[0], thisPrimitive.v[1], thisPrimitive.v[2]);
									Fragment f;

									f.eyeNor = glm::normalize(
										thisPrimitive.v[0].eyeNor * baryCoords.x +
										thisPrimitive.v[1].eyeNor * baryCoords.y +
										thisPrimitive.v[2].eyeNor * baryCoords.z);

									f.eyePos = glm::normalize(
										thisPrimitive.v[0].eyePos * baryCoords.x +
										thisPrimitive.v[1].eyePos * baryCoords.y +
										thisPrimitive.v[2].eyePos * baryCoords.z);
#if CORRECT_INTERP
									// get correct depth to use for interpolation
									float newBaryDepth = 1.f / (1 / thisPrimitive.v[0].pos.z * baryCoords.x +
										1 / thisPrimitive.v[1].pos.z * baryCoords.y +
										1 / thisPrimitive.v[2].pos.z * baryCoords.z);

									f.color = glm::normalize(newBaryDepth * (thisPrimitive.v[0].col * baryCoords.x / thisPrimitive.v[0].pos.z +
										thisPrimitive.v[1].col * baryCoords.y / thisPrimitive.v[1].pos.z +
										thisPrimitive.v[2].col * baryCoords.z / thisPrimitive.v[2].pos.z));
#else
									f.color = glm::normalize(thisPrimitive.v[0].col * baryCoords.x +
										thisPrimitive.v[1].col * baryCoords.y +
										thisPrimitive.v[2].col * baryCoords.z);
#endif


									/***** Debug views handling ****/
#if DEBUG_Z
									f.color = glm::abs(glm::vec3(1.f - baryDepth));

#endif
#if DEBUG_NORM
									f.color = f.eyeNor;
#endif

									/***** Texture handling ****/
#if TEXTURE
#if CORRECT_INTERP
						// get the starting index of the color in the tex buffer
									glm::vec2 uv = newBaryDepth * (
										thisPrimitive.v[0].texcoord0 * baryCoords.x / thisPrimitive.v[0].pos.z +
										thisPrimitive.v[1].texcoord0 * baryCoords.y / thisPrimitive.v[1].pos.z +
										thisPrimitive.v[2].texcoord0 * baryCoords.z / thisPrimitive.v[2].pos.z);

#else
									glm::vec2 uv = thisPrimitive.v[0].texcoord0 * baryCoords.x +
										thisPrimitive.v[1].texcoord0 * baryCoords.y +
										thisPrimitive.v[2].texcoord0 * baryCoords.z;
#endif
									uv.x *= thisPrimitive.v[0].diffuseTexWidth;
									uv.y *= thisPrimitive.v[0].diffuseTexHeight;
#if TEXTURE_BILINEAR
									f.uvBilinear = uv;
									f.diffuseTexWidth = thisPrimitive.v[0].diffuseTexWidth;
									f.diffuseTexHeight = thisPrimitive.v[0].diffuseTexHeight;
#endif
									// actual index into tex buffer is 1D, multip by 3 since every 3 floats (rgb) is a color
									f.uvStart = ((int)uv.x + (int)uv.y * thisPrimitive.v[0].diffuseTexWidth) * 3;
									f.dev_diffuseTex = thisPrimitive.v[0].dev_diffuseTex;
#endif
									fragmentBuffer[fragmentIdx] = f;
								}
							}
							if (isSet)
							{
								dev_mutex[fragmentIdx] = 0;
							}
						} while (!isSet);

					}
				}

			}

		}
		if (renderMode == RenderMode::Wireframe)
		{
			rasterizerFillWireFrame(fragmentBuffer,
				depth, p1, p2, p3, w, h);
		}
		if (renderMode == RenderMode::Points)
		{
			rasterizerFillPoints(fragmentBuffer, depth, p1, p2, p3, w, h);
		}
	}
}

/**
 * Perform rasterization.
 */
void rasterize(uchar4 *pbo, const glm::mat4 & MVP, const glm::mat4 & MV, const glm::mat3 MV_normal,
	int renderMode, glm::mat4 autoRotateMat4,
	bool bloomEffect) {
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

				_vertexTransformAndAssembly << < numBlocksForVertices, numThreadsPerBlock >> >(p->numVertices, *p, MVP, MV, MV_normal, width, height, autoRotateMat4);
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
	
	curPrimitiveBeginId = 0;
	dim3 numThreadsPerBlock(128);

	auto it = mesh2PrimitivesMap.begin();
	auto itEnd = mesh2PrimitivesMap.end();

	for (; it != itEnd; ++it)
	{
		auto p = (it->second).begin();
		auto pEnd = (it->second).end();
		for (; p != pEnd; ++p)
		{
			//launch kernel to fill fragments
			dim3 numBlocksForPrimitives((p->numPrimitives + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
			rasterizeFill<<<numBlocksForPrimitives, numThreadsPerBlock>>>(p->numPrimitives, curPrimitiveBeginId,dev_primitives, dev_fragmentBuffer,
				dev_depth, width, height,renderMode, dev_mutex);
			checkCUDAError("rasterizerFillPrimitive error!");
			cudaDeviceSynchronize();
			curPrimitiveBeginId += p->numPrimitives;
		}
	}

    // Copy depthbuffer colors into framebuffer
	// Modified: need more info in fragment shader such as light position
	glm::vec3 singlePointLight(3.0f, 6.0f, -5.0f);
	render << <blockCount2d, blockSize2d >> >(width, height, dev_fragmentBuffer, dev_framebuffer, singlePointLight, renderMode, bloomGBedge);
	checkCUDAError("fragment shader");
    
	if (bloomEffect)
	{
		//BLOOM EFFECT POST PROCESSING

		//BRIGHTNESS FITLER
		brightnessFilter<<<blockCount2d, blockSize2d>>>(width, height, dev_framebuffer, dev_framebufferAux);
		//DOWN SCALE
		int downScaleRatio = DOWNSCALERATIO;
		dim3 blockCount2dDownScaled((width / downScaleRatio - 1) / blockSize2d.x + 1,
			(height / downScaleRatio - 1) / blockSize2d.y + 1);
		downScaleSample<<<blockCount2dDownScaled,blockSize2d>>>(width / downScaleRatio, height / downScaleRatio, downScaleRatio,
			width, height,
			dev_framebufferDownScaled, dev_framebufferAux,
			bloomGBedge);
		//GAUSSIAN BLUR
		gaussianBlurHorizon << <blockCount2dDownScaled, blockSize2d >> > (width / downScaleRatio, height / downScaleRatio,
			dev_framebufferDownScaled, dev_framebufferDownScaledAux,
			bloomGBedge);
		gaussianBlurVert << <blockCount2dDownScaled, blockSize2d >> > (width / downScaleRatio, height / downScaleRatio,
			dev_framebufferDownScaledAux, dev_framebufferDownScaled,
			bloomGBedge);
		//FINAL COMBINE
		bloomCombineFramebuffers << <blockCount2d, blockSize2d >> > (width, height,
			dev_framebuffer, dev_framebufferDownScaled,
			dev_framebufferAux,
			width / downScaleRatio,
			downScaleRatio,
			bloomGBedge);
		checkCUDAError("bloom effect");

		// Copy post-processed framebuffer into OpenGL buffer for OpenGL previewing
		sendImageToPBO << <blockCount2d, blockSize2d >> > (pbo, width, height, dev_framebufferAux, bloomGBedge, width,1);
		checkCUDAError("copy render result to pbo");
	}
	
	else
	{
		// Copy framebuffer into OpenGL buffer for OpenGL previewing
		sendImageToPBO << <blockCount2d, blockSize2d >> > (pbo, width, height, dev_framebuffer,bloomGBedge, width, 1);
		checkCUDAError("copy render result to pbo");
	}
	
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

	//free extra buffers
	cudaFree(dev_framebufferAux);
	dev_framebufferAux = NULL;

	cudaFree(dev_framebufferDownScaled);
	dev_framebufferDownScaled = NULL;

	cudaFree(dev_framebufferDownScaledAux);
	dev_framebufferDownScaled = NULL;

	cudaFree(dev_depth);
	dev_depth = NULL;



    checkCUDAError("rasterize Free");
}
