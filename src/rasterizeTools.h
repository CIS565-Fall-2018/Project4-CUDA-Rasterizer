/**
 * @file      rasterizeTools.h
 * @brief     Tools/utility functions for rasterization.
 * @authors   Yining Karl Li
 * @date      2012-2015
 * @copyright University of Pennsylvania
 */

#pragma once

#include <cmath>
#include <glm/glm.hpp>
#include <util/utilityCore.hpp>

struct AABB {
    glm::vec3 min;
    glm::vec3 max;
};

/**
 * Multiplies a glm::mat4 matrix and a vec4.
 */
__host__ __device__ static
glm::vec3 multiplyMV(glm::mat4 m, glm::vec4 v) {
    return glm::vec3(m * v);
}

// CHECKITOUT
/**
 * Finds the axis aligned bounding box for a given triangle.
 */
__host__ __device__ static
AABB getAABBForTriangleWithClamp(const glm::vec3 tri[3], float width, float height) {
    AABB aabb;
    aabb.min = glm::vec3(
            max(min(min(tri[0].x, tri[1].x), tri[2].x),0.0f),
            max(min(min(tri[0].y, tri[1].y), tri[2].y),0.0f),
            min(min(tri[0].z, tri[1].z), tri[2].z));
    aabb.max = glm::vec3(
            min(max(max(tri[0].x, tri[1].x), tri[2].x),width),
            min(max(max(tri[0].y, tri[1].y), tri[2].y),height),
            max(max(tri[0].z, tri[1].z), tri[2].z));
    return aabb;
}

// CHECKITOUT
/**
 * Calculate the signed area of a given triangle.
 */
__host__ __device__ static
float calculateSignedArea(glm::vec3 tri[3]) {
    return 0.5 * ((tri[2].x - tri[0].x) * (tri[1].y - tri[0].y) - (tri[1].x - tri[0].x) * (tri[2].y - tri[0].y));
}

// CHECKITOUT
/**
 * Helper function for calculating barycentric coordinates.
 */
__host__ __device__ static
float calculateBarycentricCoordinateValue(glm::vec2 a, glm::vec2 b, glm::vec2 c,  glm::vec3 tri[3]) {
    glm::vec3 baryTri[3];
    baryTri[0] = glm::vec3(a, 0);
    baryTri[1] = glm::vec3(b, 0);
    baryTri[2] = glm::vec3(c, 0);
    return calculateSignedArea(baryTri) / calculateSignedArea(tri);
}

// CHECKITOUT
/**
 * Calculate barycentric coordinates.
 */
__host__ __device__ static
glm::vec3 calculateBarycentricCoordinate(glm::vec3 tri[3], glm::vec2 point) {
    float beta  = calculateBarycentricCoordinateValue(glm::vec2(tri[0].x, tri[0].y), point, glm::vec2(tri[2].x, tri[2].y), tri);
    float gamma = calculateBarycentricCoordinateValue(glm::vec2(tri[0].x, tri[0].y), glm::vec2(tri[1].x, tri[1].y), point, tri);
    float alpha = 1.0 - beta - gamma;
    return glm::vec3(alpha, beta, gamma);
}

// CHECKITOUT
/**
 * Check if a barycentric coordinate is within the boundaries of a triangle.
 */
__host__ __device__ static
bool isBarycentricCoordInBounds(const glm::vec3 barycentricCoord) {
    return barycentricCoord.x >= 0.0 && barycentricCoord.x <= 1.0 &&
           barycentricCoord.y >= 0.0 && barycentricCoord.y <= 1.0 &&
           barycentricCoord.z >= 0.0 && barycentricCoord.z <= 1.0;
}

// CHECKITOUT
/**
 * For a given barycentric coordinate, compute the corresponding z position
 * (i.e. depth) on the triangle.
 */
__host__ __device__ static
float getZAtCoordinate(const glm::vec3 barycentricCoord, const glm::vec3 tri[3]) {
    return -(barycentricCoord.x * tri[0].z
           + barycentricCoord.y * tri[1].z
           + barycentricCoord.z * tri[2].z);
}

//HELPER FUNCTIONS

/**
 * Helper function, determine whether a point is inside a triangle
 */
__device__
bool isPosInTriangle(glm::vec3 p, glm::vec3 pos1, glm::vec3 pos2, glm::vec3 pos3)
{
	glm::vec3 v(p[0], p[1], 0.f);
	glm::vec3 v1(pos1[0], pos1[1], 0.f);
	glm::vec3 v2(pos2[0], pos2[1], 0.f);
	glm::vec3 v3(pos3[0], pos3[1], 0.f);

	//compute areas for barycentric coordinates
	float area = 0.5f * glm::length(glm::cross(v1 - v2, v3 - v2));
	float area1 = 0.5f * glm::length(glm::cross(v - v2, v3 - v2));
	float area2 = 0.5f * glm::length(glm::cross(v - v3, v1 - v3));
	float area3 = 0.5f * glm::length(glm::cross(v - v1, v2 - v1));

	//check the sum of three (signed) areas against the whole area
	return glm::abs(area1 + area2 + area3 - area) < 0.001f;
}

/**
 * Helper function, interpolate depth value
 * Argument list: p is in screen space: (pixel.x, pixel.y, 0)
 * pos1, pos2, pos3 are in combination of screen space and camera space: (pixel.x, pixel.y, eyeSpace.z)
 */

__device__
float depthInterpolate(glm::vec3 p,
	glm::vec3 pos1, glm::vec3 pos2, glm::vec3 pos3)
{
	glm::vec3 v1(pos1[0], pos1[1], 0.f);
	glm::vec3 v2(pos2[0], pos2[1], 0.f);
	glm::vec3 v3(pos3[0], pos3[1], 0.f);

	//compute areas for barycentric coordinates
	float area = 0.5f * glm::length(glm::cross(v1 - v2, v3 - v2));
	float area1 = 0.5f * glm::length(glm::cross(p - v2, v3 - v2));
	float area2 = 0.5f * glm::length(glm::cross(p - v3, v1 - v3));
	float area3 = 0.5f * glm::length(glm::cross(p - v1, v2 - v1));

	//calculate interpolated z:
	float z_inverse_interpolated = area1 / (pos1[2] * area) + area2 / (pos2[2] * area) + area3 / (pos3[2] * area);
	return 1.0f / z_inverse_interpolated;
}

/**
 * Helper function, interpolate any vec2 vertex attributes, mainly for texture coords
 * Argument list: p, pos1, pos2, pos3 are in combination of screen space and camera space: (pixel.x, pixel.y, eyeSpace.z)
 * Attributes to interpolate are attri1, attri2 and attri3
 */
__device__
glm::vec2 vec2AttriInterpolate(glm::vec3 p,
	glm::vec3 pos1, glm::vec3 pos2, glm::vec3 pos3,
	glm::vec2 attri1, glm::vec2 attri2, glm::vec2 attri3)
{
	glm::vec3 v(p[0], p[1], 0.f);
	glm::vec3 v1(pos1[0], pos1[1], 0.f);
	glm::vec3 v2(pos2[0], pos2[1], 0.f);
	glm::vec3 v3(pos3[0], pos3[1], 0.f);

	//compute areas for barycentric coordinates
	float area = 0.5f * glm::length(glm::cross(v1 - v2, v3 - v2));
	float area1 = 0.5f * glm::length(glm::cross(v - v2, v3 - v2));
	float area2 = 0.5f * glm::length(glm::cross(v - v3, v1 - v3));
	float area3 = 0.5f * glm::length(glm::cross(v - v1, v2 - v1));

	//formula:
	// A/Z = Sigma (Ai/Zi * Si/S)
	glm::vec2 AttriIntepolate =
		(attri1 / pos1[2]) * (area1 / area)
		+ (attri2 / pos2[2]) * (area2 / area)
		+ (attri3 / pos3[2]) * (area3 / area);
	return AttriIntepolate * p[2];
}

/**
 * Helper function, interpolate any vec3 vertex attributes
 * Argument list: p, pos1, pos2, pos3 are in combination of screen space and camera space: (pixel.x, pixel.y, eyeSpace.z)
 * Attributes to interpolate are attri1, attri2 and attri3
 * basicall same as vec2 attribute interpolation, but with vec3
 */
__device__
glm::vec3 vec3AttriInterpolate(glm::vec3 p,
	glm::vec3 pos1, glm::vec3 pos2, glm::vec3 pos3,
	glm::vec3 attri1, glm::vec3 attri2, glm::vec3 attri3)
{
	glm::vec3 v(p[0], p[1], 0.f);
	glm::vec3 v1(pos1[0], pos1[1], 0.f);
	glm::vec3 v2(pos2[0], pos2[1], 0.f);
	glm::vec3 v3(pos3[0], pos3[1], 0.f);

	//compute areas for barycentric coordinates
	float area = 0.5f * glm::length(glm::cross(v1 - v2, v3 - v2));
	float area1 = 0.5f * glm::length(glm::cross(v - v2, v3 - v2));
	float area2 = 0.5f * glm::length(glm::cross(v - v3, v1 - v3));
	float area3 = 0.5f * glm::length(glm::cross(v - v1, v2 - v1));

	//formula:
	// A/Z = Sigma (Ai/Zi * Si/S)
	glm::vec3 AttriIntepolate =
		(attri1 / pos1[2]) * (area1 / area)
		+ (attri2 / pos2[2]) * (area2 / area)
		+ (attri3 / pos3[2]) * (area3 / area);
	return AttriIntepolate * p[2];
}
