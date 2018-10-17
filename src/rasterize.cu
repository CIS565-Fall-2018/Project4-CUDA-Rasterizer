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
#include <thrust/remove.h>
#include <thrust/execution_policy.h>

#define SSAA_RES 1
#define CORRECT_PROSPECTIVE 1
#define BACKFACE_CULLING 1

#define LINE 0
#define POINT 0
#define TRIANGLE 1

#ifndef imax
#define imax(a, b) (((a) > (b)) ? (a) : (b))
#endif

#ifndef imin
#define imin(a, b) (((a) < (b)) ? (a) : (b))
#endif
#define SCREENGAMMA 2.2

template<typename T>
__host__ __device__

void swap(T &a, T &b) {
    T tmp(a);
    a = b;
    b = tmp;
}

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

        // The attributes listed below might be useful,
        // but always feel free to modify on your own

        glm::vec3 eyePos;    // eye space position used for shading
        glm::vec3 eyeNor;    // eye space normal used for shading, cuz normal will go wrong after perspective transformation
        glm::vec3 col;
        glm::vec2 texcoord0;
        TextureData *dev_diffuseTex = NULL;
        int texWidth, texHeight;
        // ...
    };

    struct Primitive {
        PrimitiveType primitiveType = Triangle;    // C++ 11 init
        VertexOut v[3];
        bool cull = false;
    };

    struct Fragment {
        glm::vec3 color;

        // The attributes listed below might be useful,
        // but always feel free to modify on your own

        glm::vec3 eyePos;    // eye space position used for shading
        glm::vec3 eyeNor;
        VertexAttributeTexcoord texcoord0;
        TextureData *dev_diffuseTex;
        int texWidth, texHeight;
        // ...
    };

    struct PrimitiveDevBufPointers {
        int primitiveMode;    //from tinygltfloader macro
        PrimitiveType primitiveType;
        int numPrimitives;
        int numIndices;
        int numVertices;

        // Vertex In, const after loaded
        VertexIndex *dev_indices;
        VertexAttributePosition *dev_position;
        VertexAttributeNormal *dev_normal;
        VertexAttributeTexcoord *dev_texcoord0;

        // Materials, add more attributes when needed
        TextureData *dev_diffuseTex;
        int diffuseTexWidth;
        int diffuseTexHeight;
        // TextureData* dev_specularTex;
        // TextureData* dev_normalTex;
        // ...

        // Vertex Out, vertex used for rasterization, this is changing every frame
        VertexOut *dev_verticesOut;

    };

}

static std::map <std::string, std::vector<PrimitiveDevBufPointers>> mesh2PrimitivesMap;


static int width = 0;
static int height = 0;
static int screen_width = 0;
static int screen_height = 0;
static int totalNumPrimitives = 0;

static Primitive *dev_primitives = NULL;
static Fragment *dev_fragmentBuffer = NULL;
static glm::vec3 *dev_framebuffer = NULL;

static int *dev_depth = NULL;    // you might need this buffer when doing depth test

static cudaEvent_t start, stop;

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
        for (int i = 0; i < SSAA_RES; i++) {
            for (int j = 0; j < SSAA_RES; j++) {
                int ss_index = x * SSAA_RES + i + (y * SSAA_RES + j) * w * SSAA_RES;
                color.x = glm::clamp(image[ss_index].x, 0.0f, 1.0f) * 255.f;
                color.y = glm::clamp(image[ss_index].y, 0.0f, 1.0f) * 255.f;
                color.z = glm::clamp(image[ss_index].z, 0.0f, 1.0f) * 255.f;
            }
        }
        color /= (SSAA_RES * SSAA_RES);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

__host__ __device__

glm::vec3 getRGBColor(const int idx, const TextureData *texture) {
    return glm::vec3(texture[idx] / 255.0f, texture[idx + 1] / 255.0f, texture[idx + 2] / 255.0f);
}


__host__ __device__

glm::vec3 textureMapping(const int w, const int h, const glm::vec2 uv, const TextureData *texture) {
    // bilinear interpolation wikipedia
    float _x = w * uv.x;
    float _y = h * uv.y;
    int x = (int) _x;
    int y = (int) _y;

    glm::vec3 q00 = getRGBColor(3 * (x + y * w), texture);
    glm::vec3 q10 = getRGBColor(3 * (x + 1 + y * w), texture);
    glm::vec3 q01 = getRGBColor(3 * (x + (y + 1) * w), texture);
    glm::vec3 q11 = getRGBColor(3 * (x + 1 + (y + 1) * w), texture);

    float dx = _x - x;
    float dy = _y - y;

    return (q00 * (1.f - dx) * (1.f - dy) + q10 * (1.f - dy) * dx + q01 * (1.f - dx) * dy + q11 * dx * dy);
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

        // texture mapping
#if TRIANGLE
        // the bug comes from place where there is no texture
        if (fragmentBuffer[index].dev_diffuseTex != NULL) {
            glm::vec3 diffuseTexture = textureMapping(fragmentBuffer[index].texWidth, fragmentBuffer[index].texHeight,
                                            fragmentBuffer[index].texcoord0,
                                            fragmentBuffer[index].dev_diffuseTex);
            // wikipedia blin phong shading model
            glm::vec3 lightPos = glm::vec3(50.f);
            glm::vec3 lightColor = glm::vec3(1.0f);
            glm::vec3 ambientColor = glm::vec3(0.9f);
            glm::vec3 specular = glm::vec3(0.9f);
            float lightPower = 1.2;

            glm::vec3 lightDir = glm::normalize(lightPos - fragmentBuffer[index].eyePos);
            glm::vec3 eyeDir = glm::normalize(-fragmentBuffer[index].eyePos);
            float lambertian = imax(glm::dot(fragmentBuffer[index].eyeNor, lightDir), 0);

            specular *= pow(imax(glm::dot(glm::normalize(lightDir + eyeDir), fragmentBuffer[index].eyeNor), 0), 16.0f);

            glm::vec3 color =
                    ambientColor * 0.1f * lightColor + (diffuseTexture * lambertian + specular) * lightColor * lightPower;

            color = pow(color, glm::vec3(1.f / SCREENGAMMA));

            framebuffer[index] = color;
        } else {
            framebuffer[index] = fragmentBuffer[index].color;
        }

#else
        framebuffer[index] = fragmentBuffer[index].color;
#endif
    }
}

/**
* Called once at the beginning of the program to allocate memory.
*/
void rasterizeInit(int w, int h) {
    width = w * SSAA_RES;
    height = h * SSAA_RES;
    screen_width = w;
    screen_height = h;
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
void initDepth(int w, int h, int *depth) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < w && y < h) {
        int index = x + (y * w);
        depth[index] = INT_MAX;
    }
}


/**
* kern function with support for stride to sometimes replace cudaMemcpy
* One thread is responsible for copying one component
*/
__global__
void _deviceBufferCopy(int N, BufferByte *dev_dst, const BufferByte *dev_src, int n, int byteStride, int byteOffset,
                       int componentTypeByteSize) {

    // Attribute (vec3 position)
    // component (3 * float)
    // byte (4 * byte)

    // id of component
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (i < N) {
        int count = i / n;
        int offset = i - count * n;    // which component of the attribute

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
        VertexAttributePosition *position,
        VertexAttributeNormal *normal,
        glm::mat4 MV, glm::mat3 MV_normal) {

    // vertex id
    int vid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (vid < numVertices) {
        position[vid] = glm::vec3(MV * glm::vec4(position[vid], 1.0f));
        normal[vid] = glm::normalize(MV_normal * normal[vid]);
    }
}

glm::mat4 getMatrixFromNodeMatrixVector(const tinygltf::Node &n) {

    glm::mat4 curMatrix(1.0);

    const std::vector<double> &m = n.matrix;
    if (m.size() > 0) {
        // matrix, copy it

        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                curMatrix[i][j] = (float) m.at(4 * i + j);
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

void traverseNode(
        std::map <std::string, glm::mat4> &n2m,
        const tinygltf::Scene &scene,
        const std::string &nodeString,
        const glm::mat4 &parentMatrix
) {
    const tinygltf::Node &n = scene.nodes.at(nodeString);
    glm::mat4 M = parentMatrix * getMatrixFromNodeMatrixVector(n);
    n2m.insert(std::pair<std::string, glm::mat4>(nodeString, M));

    auto it = n.children.begin();
    auto itEnd = n.children.end();

    for (; it != itEnd; ++it) {
        traverseNode(n2m, scene, *it, M);
    }
}

void rasterizeSetBuffers(const tinygltf::Scene &scene) {

    totalNumPrimitives = 0;

    std::map < std::string, BufferByte * > bufferViewDevPointers;

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

            BufferByte *dev_bufferView;
            cudaMalloc(&dev_bufferView, bufferView.byteLength);
            cudaMemcpy(dev_bufferView, &buffer.data.front() + bufferView.byteOffset, bufferView.byteLength,
                       cudaMemcpyHostToDevice);

            checkCUDAError("Set BufferView Device Mem");

            bufferViewDevPointers.insert(std::make_pair(key, dev_bufferView));

        }
    }



    // 2. for each mesh:
    //		for each primitive:
    //			build device buffer of indices, materail, and each attributes
    //			and store these pointers in a map
    {

        std::map <std::string, glm::mat4> nodeString2Matrix;
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

            const tinygltf::Node &N = scene.nodes.at(itNode->first);
            const glm::mat4 &matrix = itNode->second;
            const glm::mat3 &matrixNormal = glm::transpose(glm::inverse(glm::mat3(matrix)));

            auto itMeshName = N.meshes.begin();
            auto itEndMeshName = N.meshes.end();

            for (; itMeshName != itEndMeshName; ++itMeshName) {

                const tinygltf::Mesh &mesh = scene.meshes.at(*itMeshName);

                auto res = mesh2PrimitivesMap.insert(std::pair < std::string, std::vector < PrimitiveDevBufPointers
                        >> (mesh.name, std::vector<PrimitiveDevBufPointers>()));
                std::vector <PrimitiveDevBufPointers> &primitiveVector = (res.first)->second;

                // for each primitive
                for (size_t i = 0; i < mesh.primitives.size(); i++) {
                    const tinygltf::Primitive &primitive = mesh.primitives[i];

                    if (primitive.indices.empty())
                        return;

                    VertexIndex *dev_indices = NULL;
                    VertexAttributePosition *dev_position = NULL;
                    VertexAttributeNormal *dev_normal = NULL;
                    VertexAttributeTexcoord *dev_texcoord0 = NULL;

                    // ----------Indices-------------

                    const tinygltf::Accessor &indexAccessor = scene.accessors.at(primitive.indices);
                    const tinygltf::BufferView &bufferView = scene.bufferViews.at(indexAccessor.bufferView);
                    BufferByte *dev_bufferView = bufferViewDevPointers.at(indexAccessor.bufferView);

                    // assume type is SCALAR for indices
                    int n = 1;
                    int numIndices = indexAccessor.count;
                    int componentTypeByteSize = sizeof(VertexIndex);
                    int byteLength = numIndices * n * componentTypeByteSize;

                    dim3 numThreadsPerBlock(128);
                    dim3 numBlocks((numIndices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
                    cudaMalloc(&dev_indices, byteLength);
                    _deviceBufferCopy << < numBlocks, numThreadsPerBlock >> > (
                            numIndices,
                                    (BufferByte *) dev_indices,
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
                        } else if (accessor.type == TINYGLTF_TYPE_VEC2) {
                            n = 2;
                        } else if (accessor.type == TINYGLTF_TYPE_VEC3) {
                            n = 3;
                        } else if (accessor.type == TINYGLTF_TYPE_VEC4) {
                            n = 4;
                        }

                        BufferByte *dev_bufferView = bufferViewDevPointers.at(accessor.bufferView);
                        BufferByte **dev_attribute = NULL;

                        numVertices = accessor.count;
                        int componentTypeByteSize;

                        // Note: since the type of our attribute array (dev_position) is static (float32)
                        // We assume the glTF model attribute type are 5126(FLOAT) here

                        if (it->first.compare("POSITION") == 0) {
                            componentTypeByteSize = sizeof(VertexAttributePosition) / n;
                            dev_attribute = (BufferByte **) &dev_position;
                        } else if (it->first.compare("NORMAL") == 0) {
                            componentTypeByteSize = sizeof(VertexAttributeNormal) / n;
                            dev_attribute = (BufferByte **) &dev_normal;
                        } else if (it->first.compare("TEXCOORD_0") == 0) {
                            componentTypeByteSize = sizeof(VertexAttributeTexcoord) / n;
                            dev_attribute = (BufferByte **) &dev_texcoord0;
                        }

                        std::cout << accessor.bufferView << "  -  " << it->second << "  -  " << it->first << '\n';

                        dim3 numThreadsPerBlock(128);
                        dim3 numBlocks((n * numVertices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
                        int byteLength = numVertices * n * componentTypeByteSize;
                        cudaMalloc(dev_attribute, byteLength);

                        _deviceBufferCopy << < numBlocks, numThreadsPerBlock >> > (
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
                    VertexOut *dev_vertexOut;
                    cudaMalloc(&dev_vertexOut, numVertices * sizeof(VertexOut));
                    checkCUDAError("Malloc VertexOut Buffer");

                    // ----------Materials-------------

                    // You can only worry about this part once you started to
                    // implement textures for your rasterizer
                    TextureData *dev_diffuseTex = NULL;
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
                    _nodeMatrixTransform << < numBlocksNodeTransform, numThreadsPerBlock >> > (
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

                            dev_vertexOut    //VertexOut
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

        std::map<std::string, BufferByte *>::const_iterator it(bufferViewDevPointers.begin());
        std::map<std::string, BufferByte *>::const_iterator itEnd(bufferViewDevPointers.end());

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


        // Multiply the MVP matrix for each vertex position, this will transform everything into clipping space
        glm::vec4 pos = MVP * glm::vec4(primitive.dev_position[vid], 1.0f);
        // Then divide the pos by its w element to transform into NDC space
        pos /= pos.w;
        // Finally transform x and y to viewport space
        pos.x = (float) width * (1.f - pos.x) / 2.f;
        pos.y = (float) height * (1.f - pos.y) / 2.f;

        // Assemble all attribute arraies into the primitive array
        primitive.dev_verticesOut[vid].pos = pos;
        primitive.dev_verticesOut[vid].eyePos = glm::vec3(MV * glm::vec4(primitive.dev_position[vid], 1.0f));
        primitive.dev_verticesOut[vid].eyeNor = glm::normalize(MV_normal * primitive.dev_normal[vid]);

        if (primitive.dev_diffuseTex != NULL) {
            primitive.dev_verticesOut[vid].dev_diffuseTex = primitive.dev_diffuseTex;
            primitive.dev_verticesOut[vid].texcoord0 = primitive.dev_texcoord0[vid];
            primitive.dev_verticesOut[vid].texHeight = primitive.diffuseTexHeight;
            primitive.dev_verticesOut[vid].texWidth = primitive.diffuseTexWidth;

        } else {
            primitive.dev_verticesOut[vid].dev_diffuseTex = NULL;

        }
    }
}


static int curPrimitiveBeginId = 0;

__global__
void _primitiveAssembly(int numIndices, int curPrimitiveBeginId, Primitive *dev_primitives,
                        PrimitiveDevBufPointers primitive) {

    // index id
    int iid = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (iid < numIndices) {


        // This is primitive assembly for triangles

        int pid;    // id for cur primitives vector
        if (primitive.primitiveMode == TINYGLTF_MODE_TRIANGLES) {
            pid = iid / (int) primitive.primitiveType;
            dev_primitives[pid + curPrimitiveBeginId].v[iid % (int) primitive.primitiveType]
                    = primitive.dev_verticesOut[primitive.dev_indices[iid]];
            dev_primitives[pid + curPrimitiveBeginId].v[iid % (int) primitive.primitiveType].col
                    = glm::vec3(0.9f);
        }


        // TODO: other primitive types (point, line)
    }

}

struct primitive_culling {
    __host__ __device__

    bool operator()(const Primitive &p) {
        return p.cull;
    }
};

// wikipedia back-face culling
__global__ void _backfaceCulling(const int numPrims, Primitive *dev_primitives) {
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index < numPrims) {
        glm::vec3 v1 = dev_primitives[index].v[1].eyePos - dev_primitives[index].v[0].eyePos;
        glm::vec3 v2 = dev_primitives[index].v[2].eyePos - dev_primitives[index].v[0].eyePos;
        glm::vec3 normal = glm::cross(v1, v2);
        dev_primitives[index].cull = normal.z > 0;
    }
}

// rasterization for points and lines
// reference: http://www.cs.cornell.edu/courses/cs4620/2010fa/lectures/07pipeline.pdf
__global__ void _rasterizePoint(const int numPrims, const int height, const int width,
                                Primitive *dev_primitives, Fragment *dev_fragment) {
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index < numPrims) {
        Primitive curr_prim = dev_primitives[index];
        glm::vec3 tri[3] = {glm::vec3(curr_prim.v[0].pos),
                            glm::vec3(curr_prim.v[1].pos), glm::vec3(curr_prim.v[2].pos)};
        for (int i = 0; i < 3; i++) {
            int x = (int) tri[i].x;
            int y = (int) tri[i].y;
            if (x >= 0 && x < width && y >= 0 && y < height) {
                int pixel = x + y * width;
                dev_fragment[pixel].color = glm::vec3(0.4f, 0.8f, 0.4f);
            }
        }
    }

}


// bresenham algorithm to draw line
// https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm
__host__ __device__

void _bresenham(glm::vec3 pt1, glm::vec3 pt2, const int height, const int width,
                Fragment *dev_fragment) {
    float x1 = glm::clamp(pt1[0], 0.f, (float) (width - 1));
    float x2 = glm::clamp(pt2[0], 0.f, (float) (width - 1));
    float y1 = glm::clamp(pt1[1], 0.f, (float) (height - 1));
    float y2 = glm::clamp(pt2[1], 0.f, (float) (height - 1));
    const glm::vec3 color(0.8f, 0.8f, 0.8f);

    bool swapped = (fabs(x2 - x1) < fabs(y2 - y1));

    if (swapped) {
        swap(x1, y1);
        swap(x2, y2);
    }

    if (x1 > x2) {
        swap(x1, x2);
        swap(y1, y2);
    }

    const float dx = x2 - x1;
    const float dy = fabs(y2 - y1);

    float err = dx / 2.0f;
    const int step_size = (y1 < y2) ? 1 : -1;
    int y = (int) y1;

    int idx;

    for (int x = (int) x1; x < (int) x2; x++) {
        if (swapped) {
            idx = y + x * width;
            dev_fragment[idx].color = color;
        } else {
            idx = x + y * width;
            dev_fragment[idx].color = color;
        }
        err -= dy;
        if (err < 0) {
            y += step_size;
            err += dx;
        }
    }

}

__global__ void _rasterizeLine(const int numPrims, const int height, const int width,
                               Primitive *dev_primitives, Fragment *dev_fragment) {
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index < numPrims) {
        Primitive curr_prim = dev_primitives[index];
        glm::vec3 tri[3] = {glm::vec3(curr_prim.v[0].pos),
                            glm::vec3(curr_prim.v[1].pos), glm::vec3(curr_prim.v[2].pos)};

        _bresenham(tri[0], tri[1], height, width, dev_fragment);
        _bresenham(tri[0], tri[2], height, width, dev_fragment);
        _bresenham(tri[1], tri[2], height, width, dev_fragment);
    }

}


__global__ void _rasterizeTraingle(const int numPrims, const int height, const int width,
                                   Primitive *dev_primitives, int *dev_depth, Fragment *dev_fragment) {
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index < numPrims) {
        Primitive curr_prim = dev_primitives[index];
        glm::vec3 tri[3] = {glm::vec3(curr_prim.v[0].pos),
                            glm::vec3(curr_prim.v[1].pos), glm::vec3(curr_prim.v[2].pos)};

        AABB curr_box = getAABBForTriangle(tri);

        /* reference: cis 460 slides
        * scratchpixel.com perspective correct interpolation vertex attributes & wikipedia texture mapping
        */
        for (int i = imax(curr_box.min.x, 0); i < imin(curr_box.max.x, width); i++) {
            for (int j = imax(curr_box.min.y, 0); j < imin(curr_box.max.y, height); j++) {
                glm::vec3 bary_coord = calculateBarycentricCoordinate(tri, glm::vec2(i, j));
                int pixel = j * width + i;// huge bug here omg!
                if (isBarycentricCoordInBounds(bary_coord)) {
                    // use color with smallest z-coordinate
                    int depth = static_cast<int>(glm::clamp(-getZAtCoordinate(bary_coord, tri), -1.f, 1.f) * INT_MAX);
                    atomicMin(&dev_depth[pixel], depth);

                    if (depth == dev_depth[pixel]) {
                        dev_fragment[pixel].eyeNor = glm::normalize(bary_coord.x * curr_prim.v[0].eyeNor
                                                                    + bary_coord.y * curr_prim.v[1].eyeNor +
                                                                    bary_coord.z * curr_prim.v[2].eyeNor);
                        dev_fragment[pixel].color = bary_coord.x * curr_prim.v[0].col
                                                    + bary_coord.y * curr_prim.v[1].col +
                                                    bary_coord.z * curr_prim.v[2].col;

                        dev_fragment[pixel].eyePos = bary_coord.x * curr_prim.v[0].eyePos
                                                     + bary_coord.y * curr_prim.v[1].eyePos +
                                                     bary_coord.z * curr_prim.v[2].eyePos;


#if CORRECT_PROSPECTIVE
                        glm::vec3 eyePosition[3] = {glm::vec3(curr_prim.v[0].eyePos),
                                                    glm::vec3(curr_prim.v[1].eyePos), glm::vec3(curr_prim.v[2].eyePos)};

                        float z = computeOneOverZ(bary_coord, eyePosition);

                        glm::vec3 uv[3] = {glm::vec3(curr_prim.v[0].texcoord0, 0.f),
                                           glm::vec3(curr_prim.v[1].texcoord0, 0.f),
                                           glm::vec3(curr_prim.v[2].texcoord0, 0.f)};

                        if (curr_prim.v[0].dev_diffuseTex != NULL) {
                            dev_fragment[pixel].dev_diffuseTex = curr_prim.v[0].dev_diffuseTex;
                            dev_fragment[pixel].texHeight = curr_prim.v[0].texHeight;
                            dev_fragment[pixel].texWidth = curr_prim.v[0].texWidth;
                            dev_fragment[pixel].texcoord0 = glm::vec2(
                                    correctCoordPerspective(z, bary_coord, eyePosition, uv));
                        } else {
                            dev_fragment[pixel].dev_diffuseTex = NULL;
                        }

#else
                        if (curr_prim.v[0].dev_diffuseTex != NULL) {
                            dev_fragment[pixel].dev_diffuseTex = curr_prim.v[0].dev_diffuseTex;
                            dev_fragment[pixel].texHeight = curr_prim.v[0].texHeight;
                            dev_fragment[pixel].texWidth = curr_prim.v[0].texWidth;
                            dev_fragment[pixel].texcoord0 = bary_coord.x * curr_prim.v[0].texcoord0
                                + bary_coord.y * curr_prim.v[1].texcoord0 +
                                bary_coord.z * curr_prim.v[2].texcoord0;
                        }
                        else {
                            dev_fragment[pixel].dev_diffuseTex = NULL;
                        }
#endif
                    }

                }
            }
        }
    }
}


/**
* Perform rasterization.
*/
void rasterize(uchar4 *pbo, const glm::mat4 &MVP, const glm::mat4 &MV, const glm::mat3 MV_normal, int primitive_type) {
    int sideLength2d = 8;
    dim3 blockSize2d(sideLength2d, sideLength2d);
    dim3 blockCount2d((width - 1) / blockSize2d.x + 1,
                      (height - 1) / blockSize2d.y + 1);


    // Execute your rasterization pipeline here
    // (See README for rasterization pipeline outline.)
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float miliseconds = 0;

    // Vertex Process & primitive assembly
    dim3 numThreadsPerBlock(128);
    {
        curPrimitiveBeginId = 0;


        auto it = mesh2PrimitivesMap.begin();
        auto itEnd = mesh2PrimitivesMap.end();

        for (; it != itEnd; ++it) {
            auto p = (it->second).begin();    // each primitive
            auto pEnd = (it->second).end();
            for (; p != pEnd; ++p) {
                dim3 numBlocksForVertices((p->numVertices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
                dim3 numBlocksForIndices((p->numIndices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);

                cudaEventRecord(start);
                _vertexTransformAndAssembly << < numBlocksForVertices, numThreadsPerBlock >> >
                                                                       (p->numVertices, *p, MVP, MV, MV_normal, width, height);
                checkCUDAError("Vertex Processing");
                cudaDeviceSynchronize();

                cudaEventRecord(stop);
                cudaEventSynchronize(stop);
                cudaEventElapsedTime(&miliseconds, start, stop);

                cudaEventRecord(start);
                _primitiveAssembly << < numBlocksForIndices, numThreadsPerBlock >> >
                                                             (p->numIndices,
                                                                     curPrimitiveBeginId,
                                                                     dev_primitives,
                                                                     *p);
                checkCUDAError("Primitive Assembly");
                cudaEventRecord(stop);
                cudaEventSynchronize(stop);
                cudaEventElapsedTime(&miliseconds, start, stop);

                curPrimitiveBeginId += p->numPrimitives;
            }
        }

        checkCUDAError("Vertex Processing and Primitive Assembly");
    }

    cudaMemset(dev_fragmentBuffer, 0, width * height * sizeof(Fragment));

    cudaEventRecord(start);
    initDepth << < blockCount2d, blockSize2d >> > (width, height, dev_depth);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&miliseconds, start, stop);

    // backface culling
    dim3 numBlocksForPrimitives;
#if TRIANGLE && BACKFACE_CULLING
    numBlocksForPrimitives = dim3((curPrimitiveBeginId + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);

    cudaEventCreate(start);
    _backfaceCulling << < numBlocksForPrimitives, numThreadsPerBlock >> > (curPrimitiveBeginId, dev_primitives);
    Primitive *culled_primitives = thrust::partition(thrust::device, dev_primitives,
                                                     dev_primitives + curPrimitiveBeginId, primitive_culling());
    checkCUDAError("BackFace culling error");
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&miliseconds, start, stop);
    curPrimitiveBeginId = (int) (culled_primitives - dev_primitives);
#endif

    // TODO: rasterize
    numBlocksForPrimitives = dim3((curPrimitiveBeginId + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
    cudaEventCreate(start);
#if POINT
    _rasterizePoint << < numBlocksForPrimitives, numThreadsPerBlock >> > (curPrimitiveBeginId, height, width,
        dev_primitives, dev_fragmentBuffer);
    checkCUDAError("Point rasterization error");

#elif LINE
    _rasterizeLine << < numBlocksForPrimitives, numThreadsPerBlock >> > (curPrimitiveBeginId, height, width,
        dev_primitives, dev_fragmentBuffer);
    checkCUDAError("Line rasterization error");

#elif TRIANGLE
    _rasterizeTraingle << < numBlocksForPrimitives, numThreadsPerBlock >> > (curPrimitiveBeginId, height, width,
            dev_primitives, dev_depth, dev_fragmentBuffer);
    checkCUDAError("Traingle rasterization error");
#endif
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&miliseconds, start, stop);

    // Copy depthbuffer colors into framebuffer
    cudaEventRecord(start);
    render << < blockCount2d, blockSize2d >> > (width, height, dev_fragmentBuffer, dev_framebuffer);
    checkCUDAError("fragment shader");
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&miliseconds, start, stop);

    cudaEventRecord(start)
    // Copy framebuffer into OpenGL buffer for OpenGL previewing
    sendImageToPBO << < blockCount2d, blockSize2d >> > (pbo, screen_width, screen_height, dev_framebuffer);
    checkCUDAError("copy render result to pbo");
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&miliseconds, start, stop);
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