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
#include <chrono>

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

__host__ __device__ static
glm::vec4 NDCToScreenSpace(const glm::vec4* v, int width, int height)
{
	glm::vec4 screenCoords(0.f, 0.f, 0.f, 1.f);
	screenCoords[0] = ((*v)[0] + 1.f) * width / 2.f;
	screenCoords[1] = (1.f - (*v)[1]) * height / 2.f;
	screenCoords[2] = ((*v)[2] + 1.f) / 2.f;;
	return screenCoords;
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

#define POINT_TOLERANCE 5.f
#define LINE_TOLERANCE 0.02f

#define IS_SAME(a,b) abs(a - b) < POINT_TOLERANCE

/**
* Check if a barycentric coordinate is on the boundary of a triangle.
*/
__host__ __device__ static
bool isBarycentricCoordOnBoundary(const glm::vec3 barycentricCoord) 
{
    return isBarycentricCoordInBounds(barycentricCoord) && (barycentricCoord.x <= LINE_TOLERANCE || barycentricCoord.y <= LINE_TOLERANCE ||
        barycentricCoord.z <= LINE_TOLERANCE);
}

/**
* Check if a barycentric coordinate is points of a triangle.
*/
__host__ __device__ static
bool isBarycentricCoordOnVertices(const glm::vec3 tri[3], glm::vec2 point) 
{
   return (IS_SAME(point.x, tri[0].x) && IS_SAME(point.y, tri[0].y)) || 
       (IS_SAME(point.x, tri[1].x) && IS_SAME(point.y, tri[1].y)) || 
       (IS_SAME(point.x, tri[2].x) && IS_SAME(point.y, tri[2].y));
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


/**
* Sample a given texture at the given UV using no filtering
*/
__host__ __device__ static
glm::vec3 sampleTextureSimple(const unsigned char* textureData, int u, int v, int textureWidth, int bytesPerPixel) 
{
	const int startPixelIndex = int(u + v * textureWidth) * bytesPerPixel;
	return glm::vec3(textureData[startPixelIndex], textureData[startPixelIndex + 1], textureData[startPixelIndex + 2]);
}

/**
* Sample a given texture at the given UV using bilinear filtering
*/
__host__ __device__ static
glm::vec3 sampleTextureBiLinear(const unsigned char* textureData, int u, int v, const glm::vec2& mixRatio, int textureWidth, int bytesPerPixel) 
{
	int sampleIndex = (u +  v * textureWidth) * bytesPerPixel;
	const glm::vec3 sample1 = glm::vec3(textureData[sampleIndex], textureData[sampleIndex + 1], textureData[sampleIndex + 2]);

	sampleIndex = (u + 1 +  v * textureWidth) * bytesPerPixel;
	const glm::vec3 sample2 = glm::vec3(textureData[sampleIndex], textureData[sampleIndex + 1], textureData[sampleIndex + 2]);

	sampleIndex = (u +  (v + 1) * textureWidth) * bytesPerPixel;
	const glm::vec3 sample3 = glm::vec3(textureData[sampleIndex], textureData[sampleIndex + 1], textureData[sampleIndex + 2]);

	sampleIndex = (u + 1 +  (v + 1) * textureWidth) * bytesPerPixel;
	const glm::vec3 sample4 = glm::vec3(textureData[sampleIndex], textureData[sampleIndex + 1], textureData[sampleIndex + 2]);

	const glm::vec3 mixInX = glm::mix(sample2, sample4, mixRatio.x);
	const glm::vec3 mixInY = glm::mix(sample1, sample3, mixRatio.x);

	return glm::mix(mixInX, mixInY, mixRatio.y) / 255.f;
}


/**
* This class is used for timing the performance
* Uncopyable and unmovable
*
* Adapted from WindyDarian(https://github.com/WindyDarian)
*/
class PerformanceTimer
{
public:
	PerformanceTimer()
	{
		cudaEventCreate(&event_start);
		cudaEventCreate(&event_end);
	}

	~PerformanceTimer()
	{
		cudaEventDestroy(event_start);
		cudaEventDestroy(event_end);
	}

	void startCpuTimer()
	{
		if (cpu_timer_started) { throw std::runtime_error("CPU timer already started"); }
		cpu_timer_started = true;

		time_start_cpu = std::chrono::high_resolution_clock::now();
	}

	void endCpuTimer()
	{
		time_end_cpu = std::chrono::high_resolution_clock::now();

		if (!cpu_timer_started) { throw std::runtime_error("CPU timer not started"); }

		std::chrono::duration<double, std::milli> duro = time_end_cpu - time_start_cpu;
		prev_elapsed_time_cpu_milliseconds =
			static_cast<decltype(prev_elapsed_time_cpu_milliseconds)>(duro.count());

		cpu_timer_started = false;
	}

	void startGpuTimer()
	{
		if (gpu_timer_started) { throw std::runtime_error("GPU timer already started"); }
		gpu_timer_started = true;

		cudaEventRecord(event_start);
	}

	void endGpuTimer()
	{
		cudaEventRecord(event_end);
		cudaEventSynchronize(event_end);

		if (!gpu_timer_started) { throw std::runtime_error("GPU timer not started"); }

		cudaEventElapsedTime(&prev_elapsed_time_gpu_milliseconds, event_start, event_end);
		gpu_timer_started = false;
	}

	float getCpuElapsedTimeForPreviousOperation() //noexcept //(damn I need VS 2015
	{
		return prev_elapsed_time_cpu_milliseconds;
	}

	float getGpuElapsedTimeForPreviousOperation() //noexcept
	{
		return prev_elapsed_time_gpu_milliseconds;
	}

	// remove copy and move functions
	PerformanceTimer(const PerformanceTimer&) = delete;
	PerformanceTimer(PerformanceTimer&&) = delete;
	PerformanceTimer& operator=(const PerformanceTimer&) = delete;
	PerformanceTimer& operator=(PerformanceTimer&&) = delete;

private:
	cudaEvent_t event_start = nullptr;
	cudaEvent_t event_end = nullptr;

	using time_point_t = std::chrono::high_resolution_clock::time_point;
	time_point_t time_start_cpu;
	time_point_t time_end_cpu;

	bool cpu_timer_started = false;
	bool gpu_timer_started = false;

	float prev_elapsed_time_cpu_milliseconds = 0.f;
	float prev_elapsed_time_gpu_milliseconds = 0.f;
};
