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
	float beta =  calculateBarycentricCoordinateValue(glm::vec2(tri[0].x, tri[0].y), point, glm::vec2(tri[2].x, tri[2].y), tri);
	float gamma = calculateBarycentricCoordinateValue(glm::vec2(tri[0].x, tri[0].y), glm::vec2(tri[1].x, tri[1].y), point, tri);
    float alpha = 1.0 - beta - gamma;
    return glm::vec3(alpha, beta, gamma);
}


// CHECKITOUT
/**
 * Check if a barycentric coordinate is within the boundaries of a triangle.
 */
__host__ __device__ static
bool isBarycentricCoordInBounds(const glm::vec3 barycentricCoord, int primitiveType, float epsilon) {
	switch (primitiveType) {
	case 3:
		// triangle
		return barycentricCoord.x >= 0.0 && barycentricCoord.x <= 1.0 &&
			barycentricCoord.y >= 0.0 && barycentricCoord.y <= 1.0 &&
			barycentricCoord.z >= 0.0 && barycentricCoord.z <= 1.0;
	case 1:
		// point
		return ((barycentricCoord.x >= 0.0 && barycentricCoord.x <= epsilon &&				
				barycentricCoord.y >= 0.0 && barycentricCoord.y <= epsilon &&
				barycentricCoord.z >= 1.0 - epsilon && barycentricCoord.z <= 1.0) || 
				(barycentricCoord.x >= 0.0 && barycentricCoord.x <= epsilon &&
				barycentricCoord.y >= 1.0 - epsilon && barycentricCoord.y <= 1.0 &&
				barycentricCoord.z >= 0.0 && barycentricCoord.z <= epsilon) || 
				(barycentricCoord.x >= 1.0 - epsilon && barycentricCoord.x <= 1.0 &&
				barycentricCoord.y >= 0.0 && barycentricCoord.y <= epsilon &&
				barycentricCoord.z >= 0.0 && barycentricCoord.z <= epsilon));

	case 2:
		// line
		return ((barycentricCoord.x <= epsilon && barycentricCoord.x >= 0.0 &&
				barycentricCoord.y >= 0.0 && barycentricCoord.y <= 1.0 &&
				barycentricCoord.z >= 0.0 && barycentricCoord.z <= 1.0) ||
				(barycentricCoord.x <= 1.0 && barycentricCoord.x >= 0.0 &&
				barycentricCoord.y >= 0.0 && barycentricCoord.y <= epsilon &&
				barycentricCoord.z >= 0.0 && barycentricCoord.z <= 1.0) ||
				(barycentricCoord.x <= 1.0 && barycentricCoord.x >= 0.0 &&
				barycentricCoord.y >= 0.0 && barycentricCoord.y <= 1.0 &&
				barycentricCoord.z >= 0.0 && barycentricCoord.z <= epsilon));
	default:
		return false;
	}
}

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
float getPerspectiveZAtCoordinate(const glm::vec3 barycentricCoord, const glm::vec3 etri[3]) {
	return -1.0f / (barycentricCoord.x / etri[0].z + barycentricCoord.y / etri[1].z + barycentricCoord.z / etri[2].z);
}


__host__ __device__ static
glm::vec3 barycentricInterpolation(const glm::vec3 baryCoord, const glm::vec3 attr[3])
{
	return baryCoord.x * attr[0] + baryCoord.y * attr[1] + baryCoord.z * attr[2];
}

__host__ __device__ static
glm::vec3 perspectiveCorrectBCIterpolation(const glm::vec3 baryCoord, const glm::vec3 etri[3], glm::vec3 attr[3], float z)
{
	return (baryCoord.x * attr[0] / etri[0].z + baryCoord.y * attr[1] / etri[1].z + baryCoord.z * attr[2] / etri[2].z) * z;
}

__host__ __device__ static
glm::vec3 barycentricInterpolation(const glm::vec3 baryCoord, const glm::vec3 attr0, const glm::vec3 attr1, const glm::vec3 attr2)
{
	return baryCoord.x * attr0 + baryCoord.y * attr1 + baryCoord.z * attr2;
}

__host__ __device__ static
glm::vec3 perspectiveCorrectBCIterpolation(const glm::vec3 baryCoord, const glm::vec3 etri[3], const glm::vec3 attr0, const glm::vec3 attr1, const glm::vec3 attr2, float z)
{
	return (baryCoord.x * attr0 / etri[0].z + baryCoord.y * attr1 / etri[1].z + baryCoord.z * attr2 / etri[2].z) * z;
}

__host__ __device__ static
glm::vec2 barycentricInterpolation(const glm::vec3 baryCoord, const glm::vec2 attr[3])
{
	return baryCoord.x * attr[0] + baryCoord.y * attr[1] + baryCoord.z * attr[2];
}

__host__ __device__ static
glm::vec2 perspectiveCorrectBCIterpolation(const glm::vec3 baryCoord, const glm::vec3 etri[3], glm::vec2 attr[3], float z)
{
	return (baryCoord.x * attr[0] / etri[0].z + baryCoord.y * attr[1] / etri[1].z + baryCoord.z * attr[2] / etri[2].z) * z;
}

__host__ __device__ static
glm::vec2 barycentricInterpolation(const glm::vec3 baryCoord, const glm::vec2 attr0, const glm::vec2 attr1, const glm::vec2 attr2)
{
	return baryCoord.x * attr0 + baryCoord.y * attr1 + baryCoord.z * attr2;
}

__host__ __device__ static
glm::vec2 perspectiveCorrectBCIterpolation(const glm::vec3 baryCoord, const glm::vec3 etri[3], const glm::vec2 attr0, const glm::vec2 attr1, const glm::vec2 attr2, float z)
{
	return (baryCoord.x * attr0 / etri[0].z + baryCoord.y * attr1 / etri[1].z + baryCoord.z * attr2 / etri[2].z) * z;
}