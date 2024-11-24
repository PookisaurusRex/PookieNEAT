#pragma once

#include "Trainer.h"

namespace NEAT 
{
	class Reporter 
	{
	public:
		virtual void Report() = 0;
	};

	class PopulationReporter : public Reporter 
	{
		const Trainer* TrackedTrainer = nullptr;

	public:
		PopulationReporter(Trainer* InTrackedTrainer) : TrackedTrainer(InTrackedTrainer) {}
		void Report() override;
	};

	class BestGenomeReporter : public Reporter
	{
		const Trainer* TrackedTrainer = nullptr;
	public:
		BestGenomeReporter(Trainer* InTrackedTrainer) : TrackedTrainer(InTrackedTrainer) {}
		void Report() override;
	};

	class NewBestGenomeReporter : public Reporter
	{
		GenomePtr BestGenome = nullptr;
		uint64 Generation = 0;
	public:
		NewBestGenomeReporter(GenomePtr InBestGenome, uint64 InGeneration) : BestGenome(InBestGenome), Generation(InGeneration) { }
		void Report() override;
	};

} // namespace NEAT