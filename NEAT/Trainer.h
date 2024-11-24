#pragma once

#include <vector>
#include <memory>
#include <string>
#include "Types.h"
#include "Genome.h"

namespace NEAT
{
	class Config;
	//using ConfigPtr = std::shared_ptr<const NEAT::Config>;
	using ConfigPtr = std::shared_ptr<NEAT::Config>;

	class Species;
	using SpeciesPtr = std::shared_ptr<NEAT::Species>;

	class Genome;
	using GenomePtr = std::shared_ptr<NEAT::Genome>;

	class Trainer
	{
	public:
		Trainer(const ConfigPtr& InConfig) : Config(InConfig) { }
		virtual ~Trainer() = default;

		virtual void Initialize(); // Called once before training begins
		virtual bool ContinueTraining(); // Returns true if the training loop should continue
		virtual void EvaluatePopulation(); // Evaluates the fitness of each genome in the population
		virtual void CheckForStagnation(); // Checks for species that have stagnated
		virtual void SpeciatePopulation(); // Calculates the adjusted fitness of each genome in the population
		virtual void PromoteEliteGenomes(); // Promotes the elite genomes to the next generation
		virtual void UpdateReproductionCounts(); // Updates the reproduction count of each species, based on its adjusted fitness, so that all species get some offspring but the best performing species get more
		virtual void ReproduceSpecies(); // Removes the lowest performing genomes from the population
		virtual void MutateOffspring(); // Reproduces and mutates the population

		virtual int GetNumInputs() const = 0; // Returns the number of inputs for the neural network
		virtual int GetNumOutputs() const = 0; // Returns the number of outputs for the neural network

		void UpdateReproductionCounts_Method1();
		void UpdateReproductionCounts_Method2();
		void UpdateReproductionCounts_Method3();

		void SpeciatePopulation_Method1();
		void SpeciatePopulation_Method2();

		void EvaluatePopulationThread(int ThreadID);
		void SpeciatePopulationThread(int ThreadID);

		virtual double Evaluate(const GenomePtr& Genome) = 0; // Evaluates the fitness of a single genome

		void RepopulateFromGenome(const GenomePtr& Genome); // Clones the genome and then mutates it, with a single original copy
		void LoadPopulation(const std::string& Filename); // Loads the entire population from a file, in a human-readable format that was saved earlier
		void SavePopulation(const std::string& Filename); // Saves the entire population to a file, in a human-readable format that can also be read back in later
		void SaveGenome(const std::string& Filename, const GenomePtr& Genome); // Serializes the genome to a file, in a human-readable format that can also be read back in later
		GenomePtr LoadGenome(const std::string& Filename); // Deserializes the genome from a file, in a human-readable format that was saved earlier
		void SaveBestGenome(); // Saves the best genome to a file, in a human-readable format that can also be read back in later
		void Train(); // Runs the training loop until ShouldContinueTraining returns false

		SpeciesPtr GetSpeciesByID(uint64 ID) const; // Returns the species with the given ID
		GenomePtr GetGenomeByID(uint64 ID) const; // Returns the genome with the given ID

		void SerializePopulationInfo(const std::string& Filename); // Serializes the population info to a file, in a human-readable format that can also be read back in later

		TArray<GenomePtr> Population;
		TArray<GenomePtr> Unspeciated; // Only used for speciation
		TArray<SpeciesPtr> ActiveSpecies; // Only used for speciation
		TArray<SpeciesPtr> Species;
		bool bHasBestGenome = false;
		NEAT::Genome BestGenome;
		ConfigPtr Config = nullptr;
		unsigned Generation = 0;
		double AverageDistance = 0.0;
		double DistanceCalculations = 0;
	};
}