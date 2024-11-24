#pragma once

#include <vector>  
#include <string>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <cmath>
#include "Array.h"

namespace NEAT
{
	enum class EAggregation 
	{
		Mean,
		Median,
		Max,
		Min,
		Sum,
		Count,
		Product,
		Variance,
		StandardDeviation,
		Percentile25,
		Percentile75,
		MAX
	};

	namespace Aggregation
	{
		static std::string ToString(EAggregation Method)
		{
			switch (Method)
			{
			case EAggregation::Mean: return "EAggregation::Mean";
			case EAggregation::Median: return "EAggregation::Median";
			case EAggregation::Max: return "EAggregation::Max";
			case EAggregation::Min: return "EAggregation::Min";
			case EAggregation::Sum: return "EAggregation::Sum";
			case EAggregation::Count: return "EAggregation::Count";
			case EAggregation::Product: return "EAggregation::Product";
			case EAggregation::Variance: return "EAggregation::Variance";
			case EAggregation::StandardDeviation: return "EAggregation::StandardDeviation";
			case EAggregation::Percentile25: return "EAggregation::Percentile25";
			case EAggregation::Percentile75: return "EAggregation::Percentile75";
			default: return "Unknown";
			}
		}

		static EAggregation FromString(const std::string& String)
		{
			if (String == "EAggregation::Mean") return EAggregation::Mean;
			if (String == "EAggregation::Median") return EAggregation::Median;
			if (String == "EAggregation::Max") return EAggregation::Max;
			if (String == "EAggregation::Min") return EAggregation::Min;
			if (String == "EAggregation::Sum") return EAggregation::Sum;
			if (String == "EAggregation::Count") return EAggregation::Count;
			if (String == "EAggregation::Product") return EAggregation::Product;
			if (String == "EAggregation::Variance") return EAggregation::Variance;
			if (String == "EAggregation::StandardDeviation") return EAggregation::StandardDeviation;
			if (String == "EAggregation::Percentile25") return EAggregation::Percentile25;
			if (String == "EAggregation::Percentile75") return EAggregation::Percentile75;
			return EAggregation::MAX;
		}

		template<typename T>
		double Mean(const TArray<T>& Values)
		{
			if (Values.Num() == 0) return T(0); // Return 0 if the vector is empty
			return std::accumulate(Values.begin(), Values.end(), 0.0) / Values.Num();
		}

		template<typename T>
		double Median(const TArray<T>& Values)
		{
			if (Values.Num() == 0) return T(0); // Return 0 if the vector is empty
			TArray<double> SortedValues = Values;
			SortedValues.Sort();
			int n = SortedValues.Num();
			if (n % 2 == 1) return SortedValues[n / 2];
			else return (SortedValues[n / 2 - 1] + SortedValues[n / 2]) / 2;
		}

		template<typename T>
		double Max(const TArray<T>& Values)
		{
			if (Values.Num() == 0) return T(0); // Return 0 if the vector is empty
			return *std::max_element(Values.begin(), Values.end());
		}

		template<typename T>
		double Min(const TArray<T>& Values)
		{
			if (Values.IsEmpty()) return T(0); // Return 0 if the vector is empty
			return *std::min_element(Values.begin(), Values.end());
		}

		template<typename T>
		double Sum(const TArray<T>& Values)
		{
			return std::accumulate(Values.begin(), Values.end(), 0.0);
		}

		template<typename T>
		double Count(const TArray<T>& Values)
		{
			return double(Values.Num());
		}

		template<typename T>
		double Product(const TArray<T>& Values)
		{
			return std::accumulate(Values.begin(), Values.end(), 1.0, std::multiplies<double>());
		}

		template<typename T>
		double Variance(const TArray<T>& Values)
		{
			if (Values.Num() == 0) return T(0); // Return 0 if the vector is empty
			double Mean = Aggregation::Mean<T>(Values);
			double Sum = std::accumulate(Values.begin(), Values.end(), 0.0, [Mean](double Sum, double Value)
			{
				return Sum + (Value - Mean) * (Value - Mean);
			});
			return Sum / Values.Num();
		}

		template<typename T>
		double StandardDeviation(const TArray<T>& Values)
		{
			if (Values.Num() == 0) return T(0); // Return 0 if the vector is empty
			return std::sqrt(Variance(Values));
		}

		template<typename T>
		double Percentile25(const TArray<T>& Values)
		{
			if (Values.Num() < 4) return T(0); // Return 0 if the vector is empty
			TArray<double> SortedValues = Values;
			SortedValues.Sort();
			int n = SortedValues.Num();
			return SortedValues[n / 4];
		}

		template<typename T>
		double Percentile75(const TArray<T>& Values)
		{
			if (Values.Num() < 4) return T(0); // Return 0 if the vector is empty
			TArray<double> SortedValues = Values;
			SortedValues.Sort();
			int n = SortedValues.Num();
			return SortedValues[3 * n / 4];
		}

		template <typename T>
		double Aggregate(const T& Values, EAggregation Method)
		{
			switch (Method)
			{
			case EAggregation::Mean: return Mean(Values);
			case EAggregation::Median: return Median(Values);
			case EAggregation::Max: return Max(Values);
			case EAggregation::Min: return Min(Values);
			case EAggregation::Sum: return Sum(Values);
			case EAggregation::Count: return Count(Values);
			case EAggregation::Variance: return Variance(Values);
			case EAggregation::StandardDeviation: return StandardDeviation(Values);
			case EAggregation::Percentile25: return Percentile25(Values);
			case EAggregation::Percentile75: return Percentile75(Values);
			default: throw std::invalid_argument("Invalid aggregation method");
			}
		}
	}
}