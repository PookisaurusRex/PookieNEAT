#include "Math.h"
#include "Array.h"
#include "Aggregations.h"

namespace NEAT
{
	namespace Fitness
	{
		namespace Regression
		{
			double MeanSquaredError(const TArray<double>& Predictions, const TArray<double>& Targets)
			{
				if (Predictions.Num() != Targets.Num()) return 0.0;

				double Error = 0.0;
				for (auto Idx = 0, StopIdx = Predictions.Num(); Idx != StopIdx; ++Idx)
				{
					Error += Math::Pow(Predictions[Idx] - Targets[Idx], 2.0);
				}

				return 1.0 - (Error / Predictions.Num());
			}

			double MeanAbsoluteError(const TArray<double>& Predictions, const TArray<double>& Targets)
			{
				if (Predictions.Num() != Targets.Num()) return 0.0;

				double Error = 0.0;
				for (auto Idx = 0, StopIdx = Predictions.Num(); Idx != StopIdx; ++Idx)
				{
					Error += Math::Abs(Predictions[Idx] - Targets[Idx]);
				}

				return 1.0 - (Error / Predictions.Num());
			}

			double RootMeanSquaredError(const TArray<double>& Predictions, const TArray<double>& Targets)
			{
				return std::sqrt(MeanSquaredError(Predictions, Targets));
			}

			double R2(const TArray<double>& Predictions, const TArray<double>& Targets)
			{
				if (Predictions.Num() != Targets.Num()) return 0.0;

				double SST = 0.0;
				double SSR = 0.0;
				double Mean = Aggregation::Mean(Targets);
				for (auto Idx = 0, StopIdx = Predictions.Num(); Idx != StopIdx; ++Idx)
				{
					SST += Math::Pow(Targets[Idx] - Mean, 2.0);
					SSR += Math::Pow(Predictions[Idx] - Targets[Idx], 2.0);
				}

				return 1.0 - SSR / SST;
			}
		}
	}
}