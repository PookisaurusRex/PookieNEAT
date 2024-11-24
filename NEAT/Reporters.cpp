#include "Reporters.h"
#include "Trainer.h"
#include "Species.h"
#include "Genome.h"
#include "Utils.h"  

namespace NEAT
{
	void PopulationReporter::Report() 
	{
		LogMessage(LogLevel::Info, "Population Health Report: Generation " + std::to_string(TrackedTrainer->Generation));
		LogMessage(LogLevel::Info, "  Number of Species: " + std::to_string(TrackedTrainer->Species.Num()));
		LogMessage(LogLevel::Info, "  Number of Genomes: " + std::to_string(TrackedTrainer->Population.Num()));
		LogMessage(LogLevel::Info, "  Average Genome Distance: " + std::to_string(TrackedTrainer->AverageDistance));
		LogMessage(LogLevel::Info, "  Best Fitness: " + std::to_string(TrackedTrainer->BestGenome.Fitness));

		for (const SpeciesPtr& Specie : TrackedTrainer->Species) 
		{
			if (Specie->Genomes.IsEmpty()) continue;
			auto BestGenome = Specie->GetBestGenome();
			LogMessage(LogLevel::Info, "  Species " + std::to_string(Specie->ID) + ":");
			LogMessage(LogLevel::Info, "   Number of Genomes: " + std::to_string(Specie->Genomes.Num()));
			LogMessage(LogLevel::Info, "   Best Fitness: " + std::to_string(BestGenome ? BestGenome->Fitness : 0));
			LogMessage(LogLevel::Info, "   Adjusted Fitness: " + std::to_string(Specie->AdjustedFitness));
			LogMessage(LogLevel::Info, "   Representative Fitness: " + std::to_string(Specie->Representative->Fitness));
			LogMessage(LogLevel::Info, "   Average Genome Distance: " + std::to_string(Specie->GetAverageGenomeDistance()));
			LogMessage(LogLevel::Info, "   Stagnation: " + std::to_string(Specie->Stagnation));
		}

		/*for (const GenomePtr& Genome : TrackedTrainer->Population)
		{
			LogMessage(LogLevel::Info, "  Genome " + std::to_string(Genome->ID) + ":");
			LogMessage(LogLevel::Info, "   Fitness: " + std::to_string(Genome->Fitness));
			LogMessage(LogLevel::Info, "   Adjusted Fitness: " + std::to_string(Genome->AdjustedFitness));
			LogMessage(LogLevel::Info, "   Species: " + std::to_string(Genome->SpeciesID));
		}*/
	}

	void BestGenomeReporter::Report()
	{
		if (TrackedTrainer->bHasBestGenome)
		{
			LogMessage(LogLevel::Info, "Best Genome Report: Generation(" + std::to_string(TrackedTrainer->Generation) + ")");
			LogMessage(LogLevel::Info, "  Genome ID: " + std::to_string(TrackedTrainer->BestGenome.ID) + "  Species ID: " + std::to_string(TrackedTrainer->BestGenome.SpeciesID) + "  Fitness: " + std::to_string(TrackedTrainer->BestGenome.Fitness));
		}
	}

	void NewBestGenomeReporter::Report()
	{
		if (BestGenome)
		{
			LogMessage(LogLevel::Info, "New Best Genome Found: Generation(" + std::to_string(Generation) + ")");
			LogMessage(LogLevel::Info, "  Genome ID: " + std::to_string(BestGenome->ID) + "  Species ID: " + std::to_string(BestGenome->SpeciesID) + "  Fitness: " + std::to_string(BestGenome->Fitness));
		}
	}
} // namespace NEAT