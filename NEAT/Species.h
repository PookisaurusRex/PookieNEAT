#pragma once

#include <memory>
#include "Array.h"

namespace NEAT
{
	class Genome;
	using GenomePtr = std::shared_ptr<NEAT::Genome>;

	class Config;
	//using ConfigPtr = std::shared_ptr<const NEAT::Config>;
	using ConfigPtr = std::shared_ptr<NEAT::Config>;

	class Species
	{
	public:
		const ConfigPtr Config = nullptr;
		GenomePtr Representative = nullptr;

		TArray<GenomePtr> Genomes;
		double BestAdjustedFitness = 0.0;
		double AdjustedFitness = 0.0;
		int DesiredPopulationSize = 0;
		unsigned Stagnation = 0;
		bool IsStagnant = false;
		unsigned ID = 0;

		Species(const GenomePtr& InRepresentative, const ConfigPtr& Config);
		~Species();

		double GetAverageGenomeDistance() const;
		
		GenomePtr GetRandomGenome() const;
		GenomePtr GetBestGenome() const;

		void AddGenome(const GenomePtr& Genome);
		void RemoveGenome(const GenomePtr& Genome);
		void ClearGenomes();

		bool IsEmpty() const { return Genomes.IsEmpty(); }
		int GetNum() const { return Genomes.Num(); }
	};
}