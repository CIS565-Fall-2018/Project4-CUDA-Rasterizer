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
#include <chrono>

namespace tinygltf{
	class Scene;
}


void rasterizeInit(int width, int height);
void rasterizeSetBuffers(const tinygltf::Scene & scene);

void rasterize(uchar4 *pbo, const glm::mat4 & MVP, const glm::mat4 & MV, const glm::mat3 MV_normal);
void rasterizeFree();

class PerformanceTimer
{
	/**
	* This class is used for timing the performance
	* Uncopyable and unmovable
	*
	* Adapted from WindyDarian(https://github.com/WindyDarian)
	*/
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

PerformanceTimer& timer();
