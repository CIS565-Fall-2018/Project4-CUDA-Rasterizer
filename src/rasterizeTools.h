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
AABB getAABBForTriangle(const glm::vec3 tri[3]) {
    AABB aabb;
    aabb.min = glm::vec3(
            min(min(tri[0].x, tri[1].x), tri[2].x),
            min(min(tri[0].y, tri[1].y), tri[2].y),
            min(min(tri[0].z, tri[1].z), tri[2].z));
    aabb.max = glm::vec3(
            max(max(tri[0].x, tri[1].x), tri[2].x),
            max(max(tri[0].y, tri[1].y), tri[2].y),
            max(max(tri[0].z, tri[1].z), tri[2].z));
    return aabb;
}

// CHECKITOUT
/**
 * Calculate the signed area of a given triangle.
 */
__host__ __device__ static
float calculateSignedArea(const glm::vec3 tri[3]) {
    return 0.5 * ((tri[2].x - tri[0].x) * (tri[1].y - tri[0].y) - (tri[1].x - tri[0].x) * (tri[2].y - tri[0].y));
}

// CHECKITOUT
/**
 * Helper function for calculating barycentric coordinates.
 */
__host__ __device__ static
float calculateBarycentricCoordinateValue(glm::vec2 a, glm::vec2 b, glm::vec2 c, const glm::vec3 tri[3]) {
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
glm::vec3 calculateBarycentricCoordinate(const glm::vec3 tri[3], glm::vec2 point) {
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

__host__ __device__ static
glm::vec3 BCInterpVector(glm::vec3 bc, glm::vec3 vectors[3]) {
	/*glm::vec3 tri1[3] = { tri[0], tri[1], bc };
	glm::vec3 tri2[3] = { tri[1], tri[2], bc };
	glm::vec3 tri3[3] = { tri[0], tri[2], bc };
	float s = calculateSignedArea(tri);
	float s1 = calculateSignedArea(tri1);
	float s2 = calculateSignedArea(tri2);
	float s3 = calculateSignedArea(tri3);

	return vectors[0] * s2 / s + vectors[1] * s3 / s + vectors[2] * s1 / s;*/

	return glm::normalize(bc.x * vectors[0] + bc.y * vectors[1] + bc.z * vectors[2]);
}

__host__ __device__ static
glm::vec2 PCInterpUV(glm::vec3 bc, glm::vec3 eyePositions[3], glm::vec2 UVs[3]) {
	

	glm::vec2 tz = bc.x * UVs[0] / eyePositions[0].z + bc.y * UVs[1] / eyePositions[1].z + bc.z * UVs[2] / eyePositions[2].z;
	float cz = bc.x / eyePositions[0].z + bc.y / eyePositions[1].z + bc.z / eyePositions[2].z;
	return tz / cz;

}

__host__ __device__ static
bool isBackface(const glm::vec3 tri[3]) {
	glm::vec3 v1 = tri[1] - tri[0];
	glm::vec3 v2 = tri[2] - tri[0];
	glm::vec3 nor = glm::cross(v1, v2);
	glm::vec3 eyeDir = glm::vec3(0.0f, 0.0f, -1.0f);
	if (glm::dot(nor, eyeDir) < 0) {
		return true;
	}
	return false;
}