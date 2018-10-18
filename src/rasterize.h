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

#define FXAA 0
#define FXAA_DEBUG_PASSTHROUGH 0
#define FXAA_DEBUG_HORZVERT    0
#define FXAA_DEBUG_PAIR        0
#define FXAA_DEBUG_EDGEPOS     0
#define FXAA_DEBUG_OFFSET      0

#define FXAA_EDGE_THRESHOLD_MIN 0.0625
#define FXAA_EDGE_THRESHOLD     0.375

#define FXAA_SUBPIX_TRIM       .333f
#define FXAA_SUBPIX_TRIM_SCALE 1.f
#define FXAA_SUBPIX_CAP        .75f

#define FXAA_SEARCH_STEPS        12
#define FXAA_SEARCH_ACCELERATION 2
#define FXAA_SEARCH_THRESHOLD    0.25f

#define COLOR_RED    (glm::vec3(1, 0, 0))
#define COLOR_BLUE   (glm::vec3(0, 0, 1))
#define COLOR_YELLOW (glm::vec3(1, 1, 0))
#define COLOR_GREEN  (glm::vec3(0, 1, 0))

namespace tinygltf{
	class Scene;
}


void rasterizeInit(int width, int height);
void rasterizeSetBuffers(const tinygltf::Scene & scene);

void rasterize(uchar4 *pbo, const glm::mat4 & MVP, const glm::mat4 & MV, const glm::mat3 MV_normal);
void rasterizeFree();
