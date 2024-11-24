#pragma once

#include <string>
#include <chrono>  
#include <iostream>
#include "Utils.h"

// Timer utilities for evaluating execution times, include nanosecond, millisecond, and second resolutions

namespace Benchmark
{
	class Timer
	{
		std::string Name;
	public:
		Timer(const std::string& InName, bool bStartRunning = false) : Name(InName), start_time_(std::chrono::high_resolution_clock::now()), is_running_(bStartRunning) {}
		Timer(const std::string& InName) : Name(InName), start_time_(std::chrono::high_resolution_clock::now()), is_running_(false) {}
		Timer() : start_time_(std::chrono::high_resolution_clock::now()), is_running_(false) {}

		using Seconds = std::chrono::seconds;
		using Microseconds = std::chrono::microseconds;
		using Milliseconds = std::chrono::milliseconds;
		using Nanoseconds = std::chrono::nanoseconds;

		void Start()
		{
			if (!is_running_)
			{
				start_time_ = std::chrono::high_resolution_clock::now();
				is_running_ = true;
			}
		}

		void Stop(bool bReport = false)
		{
			if (is_running_)
			{
				end_time_ = std::chrono::high_resolution_clock::now();
				is_running_ = false;
			}

			if (bReport)
			{
				ReportMilliseconds();
			}
		}

		void Reset()
		{
			start_time_ = std::chrono::high_resolution_clock::now();
			end_time_ = start_time_;
			is_running_ = false;
		}

		void ReportMilliseconds()
		{
			if (is_running_)
			{
				auto current_time = std::chrono::high_resolution_clock::now();
				auto elapsed_time = current_time - start_time_;
				NEAT::LogMessage(NEAT::LogLevel::Info, Name + " Elapsed time: " + std::to_string(std::chrono::duration_cast<Milliseconds>(elapsed_time).count()) + "ms");
			}
			else
			{
				auto elapsed_time = end_time_ - start_time_;
				NEAT::LogMessage(NEAT::LogLevel::Info, Name + " Elapsed time: " + std::to_string(std::chrono::duration_cast<Milliseconds>(elapsed_time).count()) + "ms");
			}
		}

		void ReportNanoseconds()
		{
			if (is_running_)
			{
				auto current_time = std::chrono::high_resolution_clock::now();
				auto elapsed_time = current_time - start_time_;
				NEAT::LogMessage(NEAT::LogLevel::Info, Name + " Elapsed time: " + std::to_string(std::chrono::duration_cast<Nanoseconds>(elapsed_time).count()) + "ms");
			}
			else
			{
				auto elapsed_time = end_time_ - start_time_;
				NEAT::LogMessage(NEAT::LogLevel::Info, Name + " Elapsed time: " + std::to_string(std::chrono::duration_cast<Nanoseconds>(elapsed_time).count()) + "ms");
			}
		}

		uint64 GetMillisecondsElapsed() const
		{
			if (is_running_)
			{
				auto current_time = std::chrono::high_resolution_clock::now();
				auto elapsed_time = current_time - start_time_;
				return std::chrono::duration_cast<Milliseconds>(elapsed_time).count();
			}

			auto elapsed_time = end_time_ - start_time_;
			return std::chrono::duration_cast<Milliseconds>(elapsed_time).count();
		}

		uint64 GetNanosecondsElapsed() const
		{
			if (is_running_) return uint64(std::chrono::duration_cast<Nanoseconds>(std::chrono::high_resolution_clock::now() - start_time_).count());
			return uint64(std::chrono::duration_cast<Nanoseconds>(end_time_ - start_time_).count());
		}

	private:
		std::chrono::high_resolution_clock::time_point start_time_;
		std::chrono::high_resolution_clock::time_point end_time_;
		bool is_running_;
	};
}