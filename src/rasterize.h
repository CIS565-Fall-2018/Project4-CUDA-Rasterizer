/**
 * @file      rasterize.h
 * @brief     CUDA-accelerated rasterization pipeline.
 * @authors   Skeleton code: Yining Karl Li, Kai Ninomiya, Shuai Shao (Shrek)
 * @date      2012-2016
 * @copyright University of Pennsylvania & STUDENT
 */

#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>

namespace tinygltf{
	class Scene;
}

#define USE_Tiles = true;
#define TILE_SIZE = 16;

void rasterizeInit(int width, int height, int tilePixelSize);
void rasterizeSetBuffers(const tinygltf::Scene & scene);
void rasterizeSetTileBuffers();

void rasterize(uchar4 *pbo, const glm::mat4 & MVP, const glm::mat4 & MV, const glm::mat3 MV_normal, int renderMode);
void rasterizeFree();
