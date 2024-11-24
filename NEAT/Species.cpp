#include "Species.h"
#include "Genome.h"	
#include "Utils.h"
#include "Math.h"

namespace NEAT
{
	static unsigned NextSpeciesID = 0;

	Species::Species(const GenomePtr& InRepresentative, const ConfigPtr& InConfig)
		: Config(InConfig)
		, Representative(InRepresentative)
		, ID(++NextSpeciesID)
	{
	}

	Species::~Species()
	{
	}

	double Species::GetAverageGenomeDistance() const
	{
		double TotalDistance = 0.0;
		int NumGenomes = Genomes.Num();

		for (int i = 0; i < NumGenomes; i++) {
			for (int j = i + 1; j < NumGenomes; j++) 
			{
				double Distance = Distance::Calculate(Genomes[i], Genomes[j], Config);
				TotalDistance += Distance;
			}
		}

		double averageDistance = TotalDistance / (NumGenomes * (NumGenomes - 1) / 2.0);
		return averageDistance;
	}

	GenomePtr Species::GetRandomGenome() const
	{
		if (Genomes.IsEmpty()) return nullptr;
		int RandomIndex = GetRandomInt(0, Genomes.Num() - 1);
		return Genomes[RandomIndex];
	}

	GenomePtr Species::GetBestGenome() const
	{
		if (Genomes.IsEmpty()) return nullptr;
		GenomePtr BestGenome = nullptr;
		double BestFitness = -std::numeric_limits<double>::max();
		for (const GenomePtr& Genome : Genomes)
		{
			if (Genome->Fitness > BestFitness)
			{
				BestFitness = Genome->Fitness;
				BestGenome = Genome;
			}
		}
		return BestGenome;
	}

	void Species::AddGenome(const GenomePtr& Genome)
	{
		Genomes.Add(Genome);
	}

	void Species::RemoveGenome(const GenomePtr& Genome)
	{
		Genomes.Remove(Genome);
	}

	void Species::ClearGenomes()
	{
		Genomes.Reset();
	}
}