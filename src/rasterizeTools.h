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

struct BoundingBox {
  glm::vec2 min;
  glm::vec2 max;
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
            glm::min(glm::min(tri[0].x, tri[1].x), tri[2].x),
      glm::min(glm::min(tri[0].y, tri[1].y), tri[2].y),
      glm::min(glm::min(tri[0].z, tri[1].z), tri[2].z));
    aabb.max = glm::vec3(
      glm::max(glm::max(tri[0].x, tri[1].x), tri[2].x),
      glm::max(glm::max(tri[0].y, tri[1].y), tri[2].y),
      glm::max(glm::max(tri[0].z, tri[1].z), tri[2].z));
    return aabb;
}

__host__ __device__ static
BoundingBox getBoundingBoxForTriangle(const glm::vec2 p0, const glm::vec2 p1, const glm::vec2 p2) {
  BoundingBox aabb;
  aabb.min = glm::vec2(
    glm::min(glm::min(p0.x, p1.x), p2.x),
    glm::min(glm::min(p0.y, p1.y), p2.y));
  aabb.max = glm::vec2(
    glm::max(glm::max(p0.x, p1.x), p2.x),
    glm::max(glm::max(p0.y, p1.y), p2.y));
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

__host__ __device__ static
float calculateSignedArea(const glm::vec2 p0, const glm::vec2 p1, const glm::vec2 p2) {
  return 0.5 * ((p2.x - p0.x) * (p1.y - p0.y) - (p1.x - p0.x) * (p2.y - p0.y));
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

__host__ __device__ static float calculateBarycentricCoordinateValue(glm::vec2 a, glm::vec2 b, glm::vec2 c, float totalSignedArea) {
  glm::vec3 baryTri[3];
  baryTri[0] = glm::vec3(a, 0);
  baryTri[1] = glm::vec3(b, 0);
  baryTri[2] = glm::vec3(c, 0);
  return calculateSignedArea(baryTri) / totalSignedArea;
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

__host__ __device__ static
glm::vec3 calculateBarycentricCoordinate(const glm::vec2 p0, const glm::vec2 p1, const glm::vec2 p2, glm::vec2 point) {
  const float totalArea = calculateSignedArea(p0, p1, p2);

  const float beta  = calculateBarycentricCoordinateValue(glm::vec2(p0.x, p0.y), point, glm::vec2(p2.x, p2.y), totalArea);
  const float gamma = calculateBarycentricCoordinateValue(glm::vec2(p0.x, p0.y), glm::vec2(p1.x, p1.y), point, totalArea);
  const float alpha = 1.0 - beta - gamma;
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
