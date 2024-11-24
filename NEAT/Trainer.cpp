#include "Trainer.h"
#include "Genome.h"
#include "Species.h"
#include "Config.h"
#include "Genes.h"
#include "Math.h"
#include "Reproduction.h"
#include "Mutations.h"
#include "Reporters.h"
#include "Network.h"
#include "Utils.h"
#include "Timer.h"
#include <algorithm>
#include <iostream>
#include <fstream>
#include <thread>
#include <mutex>

// Called once before training begins, using Config settings to initialize the population
void NEAT::Trainer::Initialize() 
{
	// Clear the existing population and species
	Population.Reset();
	Species.Reset();
	BestGenome = NEAT::Genome(Config);
	Generation = 0;

	Innovations.Reset(Config->NumInputs + Config->NumOutputs + Config->NumHidden + 1); // Reset the innovation tracker, with the number of inputs, outputs, and hidden nodes, plus one for the bias node
	
	std::vector<GenomePairing::Offspring> InitialPopulation; // Create the initial population
	for (unsigned Idx = 0; Idx < Config->PopulationSize; ++Idx)
	{
		InitialPopulation.push_back(GenomePairing::Offspring(Config));
	}

	for (auto& Pairing : InitialPopulation)
	{
		Population.Add(Pairing.GetChild());
	}

	LogMessage(LogLevel::Info, "Trainer initialized with population size: " + std::to_string(Population.Num()));
	LogMessage(LogLevel::Info, "Starting training for " + std::to_string(Config->MaxGenerations) + " generations");

	// Create the initial species
	//SpeciatePopulation();
}

// Returns true if the training loop should continue, using the generation count and the fitness of the best genome in the population to determine if training should continue
// Using data made available from the Config object
bool NEAT::Trainer::ContinueTraining() 
{
	if (bHasBestGenome && Config->StoppingFitness > 0 && BestGenome.Fitness >= Config->StoppingFitness) return false;
	if (Config->MaxGenerations > 0 && Generation >= Config->MaxGenerations) return false;
	return true;
}

// Evaluates the fitness of each genome in the population
/*
void NEAT::Trainer::EvaluatePopulation()
{
	LogMessage(LogLevel::Info, "Generation " + std::to_string(Generation) + ": Evaluating population");

	for (auto& Genome : Population)
	{
		Genome->SetFitness(Evaluate(Genome));
	}

	// Check for new best genome
	for (auto& Genome : Population)
	{
		if (!BestGenome || Genome->Fitness > BestGenome->Fitness)
		{
			BestGenome = Genome;
		}
	}

	// After evaluating the fitness of each genome in the population, calculate the adjusted fitness of each species.
	for (auto& Specie : Species)
	{
		Specie->AdjustedFitness = 0.0;
		for (auto& Genome : Specie->Genomes)
		{
			Specie->AdjustedFitness += Genome->Fitness / Specie->Genomes.size();
		}
	}
}*/

void NEAT::Trainer::EvaluatePopulation() 
{
	if (Config->MultithreadedEvaluation)  // Multithreaded evaluation  
	{
		std::vector<std::thread> Threads;
		for (int Idx = 0, StopIdx = Config->NumThreads; Idx != StopIdx; ++Idx)
		{
			Threads.emplace_back([this, Idx]() { EvaluatePopulationThread(Idx);	});
		}

		for (auto& Thread : Threads) Thread.join();
	}
	else 
	{
		EvaluatePopulationThread(0); // Single-threaded evaluation  
	}

	// Check for new best genome
	for (auto& Genome : Population)
	{
		if (!bHasBestGenome || Genome->Fitness > BestGenome.Fitness)
		{
			BestGenome = *Genome;
			BestGenome.Config = Config;
			bHasBestGenome = true;
			NEAT::NewBestGenomeReporter NewBestGenomeReporter(Genome, Generation);
			NewBestGenomeReporter.Report();
		}
	}

	// After evaluating the fitness of each genome in the population, calculate the adjusted fitness of each species.
	for (auto& Specie : Species)
	{
		Specie->AdjustedFitness = 0.0;
		for (auto& Genome : Specie->Genomes)
		{
			Specie->AdjustedFitness += Genome->Fitness / Specie->Genomes.Num();
		}
	}
}

void NEAT::Trainer::EvaluatePopulationThread(int ThreadID)
{
	int StartIdx = ThreadID * (Population.Num() / Config->NumThreads);
	int EndIdx = (ThreadID + 1) * (Population.Num() / Config->NumThreads);
	if (ThreadID == Config->NumThreads - 1) 
	{
		EndIdx = Population.Num();
	}

	for (int Idx = StartIdx; Idx != EndIdx; ++Idx)
	{
		GenomePtr Genome = Population[Idx];
		Genome->Fitness = Evaluate(Genome);
	}
}

// Checks for species that have stagnated
void NEAT::Trainer::CheckForStagnation() 
{
	for (auto& Specie : Species)
	{
		if (Specie->IsStagnant) continue;
		if (Specie->AdjustedFitness > Specie->BestAdjustedFitness)
		{
			Specie->BestAdjustedFitness = Specie->AdjustedFitness;
			Specie->Stagnation = 0;
		}
		else if (/*Specie->Genomes.IsEmpty() && */!Specie->Representative)
		{
			Specie->IsStagnant = true;
		}
		else
		{
			Specie->Stagnation++;
			if (Population.Num() > (2.0 * Config->PopulationSize) && Specie->Stagnation >= (Config->MaxStagnation / 3))
			{
				Specie->IsStagnant = true;
			}
			else if (Specie->Stagnation >= Config->MaxStagnation)
			{
				Specie->IsStagnant = true;
			}
		}
	}

	if (Species.Num() <= 1) return;

	auto SpeciesCopy = Species;
	for (const auto& Specie : SpeciesCopy)
	{
		if (!Specie->IsStagnant) continue;
		for (const auto& Genome : Specie->Genomes)
		{
			Population.Remove(Genome);
		}
		Population.Remove(Specie->Representative);
	}

	Species.RemoveByPredicate([](const SpeciesPtr& Specie) { return Specie->IsStagnant || (!Specie->Representative && Specie->Genomes.IsEmpty()); });

	if (Population.IsEmpty())
	{
		std::vector<GenomePairing::Offspring> InitialPopulation;
		for (unsigned Idx = 0; Idx < Config->PopulationSize; ++Idx)
		{
			InitialPopulation.push_back(GenomePairing::Offspring(Config));
		}

		for (auto& Pairing : InitialPopulation)
		{
			Population.Add(Pairing.GetChild());
		}
	}
}

// Calculates the adjusted fitness of each genome in the population
void NEAT::Trainer::SpeciatePopulation()
{
	SpeciatePopulation_Method1();
	//SpeciatePopulation_Method2();
}

void NEAT::Trainer::SpeciatePopulation_Method1()
{
	bool bNoSpecies = Species.IsEmpty();
	if (bNoSpecies)
	{
		// Create a new species for the first genome  
		auto NewSpecies = std::make_shared<NEAT::Species>(Population[0], Config);
		NewSpecies->AddGenome(Population[0]);
		Population[0]->SpeciesID = NewSpecies->ID;
		Species.Add(NewSpecies);
	}

	for (auto& Specie : Species)
	{
		if (!Specie->Genomes.IsEmpty())
		{
			Specie->Representative = Config->ChooseBestRepresentative ? Specie->GetBestGenome() : Specie->GetRandomGenome();
		}

		Specie->Genomes.Reset();
	}

	AverageDistance = 0.0;
	DistanceCalculations = 0;
	double DistanceSum = 0.0;

	for (auto i = bNoSpecies ? 1 : 0; i < Population.Num(); ++i)
	{
		auto& Genome = Population[i];
		bool bSpeciesFound = false;
		for (auto& Specie : Species)
		{
			double Distance = Distance::Calculate(Specie->Representative, Genome, Config);
			DistanceSum += Distance;
			DistanceCalculations++;
			if (Distance < Config->SpeciationDistanceThreshold)
			{
				Specie->AddGenome(Genome);
				Genome->SpeciesID = Specie->ID;
				bSpeciesFound = true;
				break;
			}
		}

		if (!bSpeciesFound)
		{
			auto NewSpecies = std::make_shared<NEAT::Species>(Genome, Config);
			NewSpecies->AddGenome(Genome);
			Genome->SpeciesID = NewSpecies->ID;
			Species.Add(NewSpecies);
		}
	}

	AverageDistance = DistanceSum / DistanceCalculations;
}

void NEAT::Trainer::SpeciatePopulation_Method2()
{
	bool bNoSpecies = Species.IsEmpty();
	if (bNoSpecies)
	{
		// Create a new species for the first genome  
		auto NewSpecies = std::make_shared<NEAT::Species>(Population[0], Config);
		NewSpecies->AddGenome(Population[0]);
		Population[0]->SpeciesID = NewSpecies->ID;
		Species.Add(NewSpecies);
	}

	for (auto& Specie : Species)
	{
		if (Config->ChooseBestRepresentative) Specie->Representative = Specie->GetBestGenome();
		else Specie->Representative = Specie->GetRandomGenome();
		Specie->Genomes.Reset();
	}

	AverageDistance = 0.0;
	DistanceCalculations = 0;
	double DistanceSum = 0.0;

	for (auto i = bNoSpecies ? 1 : 0; i < Population.Num(); ++i)
	{
		auto& Genome = Population[i];
		bool bSpeciesFound = false;
		for (auto& Specie : Species)
		{
			double Distance = Distance::Calculate(Specie->Representative, Genome, Config);
			DistanceSum += Distance;
			DistanceCalculations++;
			if (Distance < Config->SpeciationDistanceThreshold)
			{
				Specie->AddGenome(Genome);
				Genome->SpeciesID = Specie->ID;
				bSpeciesFound = true;
				break;
			}
		}

		if (!bSpeciesFound)
		{
			auto NewSpecies = std::make_shared<NEAT::Species>(Genome, Config);
			NewSpecies->AddGenome(Genome);
			Genome->SpeciesID = NewSpecies->ID;
			Species.Add(NewSpecies);
		}
	}

	AverageDistance = DistanceSum / DistanceCalculations;

	// Update the representative genome for each species  
	for (auto& Specie : Species)
	{
		if (Config->ChooseBestRepresentative) Specie->Representative = Specie->GetBestGenome();
		else Specie->Representative = Specie->GetRandomGenome();
	}

	if (Species.IsEmpty())
	{
		Species.Add(std::make_shared<NEAT::Species>(Population[0], Config)); // Create a new species for the first genome
	}

	// Clear existing species assignments
	for (auto& Genome : Population)	Genome->SpeciesID = 0;
	Unspeciated = Population; // Copy the population to the Unspeciated list
	ActiveSpecies = Species;

	while (!Unspeciated.IsEmpty())
	{
		if (Config->MultithreadedEvaluation)  // Multithreaded evaluation  
		{
			std::vector<std::thread> Threads;
			for (int Idx = 1, StopIdx = Config->NumThreads; Idx != StopIdx; ++Idx)
			{
				Threads.emplace_back([this, Idx]() { SpeciatePopulationThread(Idx);	});
			}

			for (auto& Thread : Threads) Thread.join();
		}
		else
		{
			//SpeciatePopulation_Method1();
			SpeciatePopulationThread(0); // Single-threaded evaluation  
		}

		Unspeciated = Unspeciated.FilterByPredicate([](const auto& Genome) { return Genome->SpeciesID == 0; }); // Filter out genomes that were not assigned to a species
		ActiveSpecies.Reset(1); // Clear the ActiveSpecies list, it's not possible to match to any of the old species
		if (!Unspeciated.IsEmpty())
		{
			Species.Add(std::make_shared<NEAT::Species>(Unspeciated[rand() % Unspeciated.Num()], Config)); // Create a new species for a random genome that was not assigned to a species
		}
	}
}

void NEAT::Trainer::SpeciatePopulationThread(int ThreadID)
{
	int StartIdx = ThreadID * (Unspeciated.Num() / Config->NumThreads);
	int EndIdx = (ThreadID + 1) * (Unspeciated.Num() / Config->NumThreads);
	if (ThreadID == Config->NumThreads - 1)
	{
		EndIdx = Unspeciated.Num();
	}

	for (int Idx = StartIdx; Idx != EndIdx; ++Idx)
	{
		auto& Genome = Unspeciated[Idx];
		// Calculate the best distance based on the ActiveSpecies list
		for (auto& Specie : ActiveSpecies)
		{
			double Distance = Distance::Calculate(Specie->Representative, Genome, Config);
			if (Distance < Config->SpeciationDistanceThreshold)
			{
				Genome->SpeciesID = Specie->ID;
				break;
			}
		}
	}
}

void NEAT::Trainer::PromoteEliteGenomes() // Promotes the elite genomes to the next generation
{
	for (auto& Genome : Population) Genome->bElite = false;

	for (auto& Specie : Species)
	{
		if (!Specie || Specie->Genomes.IsEmpty()) continue;
		Specie->Genomes.Sort([](const GenomePtr& A, const GenomePtr& B) { return A->Fitness > B->Fitness; });
		for (int Idx = 0, StopIdx = Math::Min(Config->SpeciesElitism, Specie->Genomes.Num()); Idx != StopIdx; ++Idx)
		{
			auto EliteGenome = Specie->Genomes[Idx];
			EliteGenome->bElite = true;
		}
	}
}

// Updates the reproduction count of each species, based on its adjusted fitness, so that all species get some offspring but the best performing species get more
void NEAT::Trainer::UpdateReproductionCounts()
{
	UpdateReproductionCounts_Method3();
}

void NEAT::Trainer::UpdateReproductionCounts_Method1()
{
	if (Species.IsEmpty()) return;
	double TotalAdjustedFitness = 0.0;
	for (auto& Specie : Species)
	{
		TotalAdjustedFitness += Specie->AdjustedFitness;
	}

	for (auto& Specie : Species)
	{
		Specie->DesiredPopulationSize = int(std::floor(Specie->AdjustedFitness / TotalAdjustedFitness * Config->PopulationSize));
		Specie->DesiredPopulationSize = Math::Max<int>(Specie->DesiredPopulationSize, Config->MinSpeciesSize, 0);
	}

	//	If there are any remaining offspring to give out, give them out round-robin, starting with the best species
	auto RemainingOffspring = Config->PopulationSize - std::accumulate(Species.begin(), Species.end(), 0, [](auto Sum, const auto& Specie) { return Sum + Specie->DesiredPopulationSize; });
	while (RemainingOffspring > 0)
	{
		for (auto& Specie : Species)
		{
			Specie->DesiredPopulationSize++;
			RemainingOffspring--;
			if (RemainingOffspring == 0) break;
		}
	}
}

void NEAT::Trainer::UpdateReproductionCounts_Method2()
{
	if (Species.IsEmpty()) return;
	double TotalAdjustedFitness = 0.0;
	for (auto& Specie : Species)
	{
		TotalAdjustedFitness += Specie->AdjustedFitness;
	}

	// Calculate desired populations based on the adjusted fitness of each species, and the Config settings, but only allow populations to be as low as the minimum species size  
	// and only populations to change by a maximum of 20% from the previous generation  
	for (auto& Specie : Species)
	{
		double DesiredPopulation = Specie->AdjustedFitness / TotalAdjustedFitness * Config->PopulationSize;
		DesiredPopulation = Math::Max<double>(double(DesiredPopulation), double(Config->MinSpeciesSize), 0.0);
		double PopulationChange = DesiredPopulation - Specie->DesiredPopulationSize;
		double MaxChange = 0.2 * Specie->DesiredPopulationSize;
		if (PopulationChange > MaxChange)
		{
			DesiredPopulation = Specie->DesiredPopulationSize + MaxChange;
		}
		else if (PopulationChange < -MaxChange)
		{
			DesiredPopulation = Specie->DesiredPopulationSize - MaxChange;
		}
		Specie->DesiredPopulationSize = int(std::round(DesiredPopulation));
	}

	// If there are any remaining offspring to give out, give them out round-robin, starting with the best species  
	auto RemainingOffspring = Config->PopulationSize - std::accumulate(Species.begin(), Species.end(), 0, [](int Sum, const SpeciesPtr& Specie) { return Sum + Specie->DesiredPopulationSize; });
	while (RemainingOffspring > 0)
	{
		for (auto& Specie : Species)
		{
			Specie->DesiredPopulationSize++;
			RemainingOffspring--;
			if (RemainingOffspring == 0) break;
		}
	}
}

void NEAT::Trainer::UpdateReproductionCounts_Method3()
{
	if (Species.IsEmpty()) return;
	uint64 MinimumPopulationSize = Species.Num() * Config->MinSpeciesSize;
	if (MinimumPopulationSize > Config->PopulationSize)
	{
		// If the minimum population size is greater than the total population size, then only allot the minimum species size to each species
		// Subtract current species size from the reproduction count so that the final species population is equal to the minimum species size
		for (auto& Specie : Species)
		{
			Specie->DesiredPopulationSize = Config->MinSpeciesSize;
		}
		return;
	}

	// Shift all adjusted fitness values to positive values
	double MinAdjustedFitness = 0.0;
	for (auto& Specie : Species)
	{
		MinAdjustedFitness = Math::Min(MinAdjustedFitness, Specie->AdjustedFitness);
	}

	MinAdjustedFitness = Math::Min(MinAdjustedFitness, 0.0);
	MinAdjustedFitness = -MinAdjustedFitness;

	for (auto& Specie : Species)
	{
		Specie->AdjustedFitness += MinAdjustedFitness;
	}

	// Sort the species by adjusted fitness
	//Species.Sort([](const SpeciesPtr& A, const SpeciesPtr& B) { return A->AdjustedFitness > B->AdjustedFitness; });

	double TotalAdjustedFitness = 0.0;
	for (auto& Specie : Species) TotalAdjustedFitness += Specie->AdjustedFitness;

	TotalAdjustedFitness = Math::Max(TotalAdjustedFitness, 0.0);

	// Calculate the number of offspring to give to each species
	for (auto& Specie : Species)
	{
		if (TotalAdjustedFitness != 0.0)
		{
			int DesiredSpeciesSize = int(Math::Floor(Specie->AdjustedFitness / TotalAdjustedFitness * Config->PopulationSize)); // Calculate the desired species size based on the species adjusted fitness
			Specie->DesiredPopulationSize = Math::Max(DesiredSpeciesSize, Config->MinSpeciesSize); // Subtract the current species size from the desired species size
		}
		else
		{
			Specie->DesiredPopulationSize = Config->MinSpeciesSize;
		}
	}

	// If there are any remaining offspring to give out, give them out round-robin, starting with the best species
	auto TotalDesiredPopulation = 0;
	for (const auto& Specie : Species)
	{
		TotalDesiredPopulation += Specie->DesiredPopulationSize;
	}

	auto RemainingOffspring = Math::Max<int>(Config->PopulationSize - TotalDesiredPopulation, 0);
	while (RemainingOffspring > 0)
	{
		for (auto& Specie : Species)
		{
			if (RemainingOffspring == 0) return;
			Specie->DesiredPopulationSize++;
			RemainingOffspring--;
		}
	}

	auto NewDesiredPopulationSize = 0;
	for (const auto& Specie : Species) NewDesiredPopulationSize += Specie->DesiredPopulationSize;

	if (NewDesiredPopulationSize <= Config->PopulationSize) return;

	auto SpeciesWithExcess = Species.FilterByPredicate([this](const auto& Specie) {return Specie->DesiredPopulationSize > Config->MinSpeciesSize; });
	if (SpeciesWithExcess.IsEmpty()) return;

	int SpecieIdx = 0;
	while (NewDesiredPopulationSize > Config->PopulationSize)
	{
		if (!SpeciesWithExcess.IsValidIndex(SpecieIdx))
		{
			SpecieIdx = ++SpecieIdx % SpeciesWithExcess.Num();
			continue;
		}

		auto Specie = SpeciesWithExcess[SpecieIdx];
		SpecieIdx = ++SpecieIdx % SpeciesWithExcess.Num();
		if (Specie->DesiredPopulationSize <= Config->MinSpeciesSize)
		{
			SpeciesWithExcess.Remove(Specie); 
			SpecieIdx--;
			continue;
		}

		Specie->DesiredPopulationSize -= 1;
		NewDesiredPopulationSize -= 1;
	}
}

// Removes the lowest performing genomes from the population, based on the reproduction count of each species, and the Config settings (e.g. Elitism)
// and also generates offspring for the next generation with the Reproduction.h content
void NEAT::Trainer::ReproduceSpecies() 
{
	// Remove the lowest performing genomes from the population
	/*for (auto& Specie : Species)
	{
		std::sort(Specie->Genomes.begin(), Specie->Genomes.end(), [](const GenomePtr& A, const GenomePtr& B) { return A->Fitness > B->Fitness; });
		Specie->Genomes.erase(Specie->Genomes.begin() + Config->SpeciesElitism, Specie->Genomes.end());
	}*/

	UpdateReproductionCounts();
	PromoteEliteGenomes();

	// Remove genomes from the species following the Config->SpeciesElitism setting, Config->SurvivalRate, and Config->MinSpeciesSize, and Config->CullingMethod
	for (auto& Specie : Species)
	{
		if (Specie->Genomes.Num() <= Config->MinSpeciesSize) continue;

		// Sort the genomes in the species by fitness
		Specie->Genomes.Sort([](const auto& LHS, const auto& RHS) 
		{ 
			if (LHS->Fitness != RHS->Fitness) return LHS->Fitness >= RHS->Fitness;
			/*auto GeneCountLHS = LHS->Genotype.Connections.Num() + LHS->Genotype.Nodes.Num();
			auto GeneCountRHS = RHS->Genotype.Connections.Num() + RHS->Genotype.Nodes.Num();
			if (GeneCountLHS != GeneCountRHS) return GeneCountLHS <= GeneCountRHS;*/
			return LHS->ID >= RHS->ID;
		});

		// Remove the lowest performing genomes from the species
		double NumToCullDouble = Specie->Genomes.Num() * Config->SurvivalRate;
		size_t NumToCull = Math::Floor<size_t>(NumToCullDouble);
		if (NumToCull < Config->MinSpeciesSize) NumToCull = Config->MinSpeciesSize;
		if (NumToCull > Specie->Genomes.Num() - Config->SpeciesElitism) NumToCull = Specie->Genomes.Num() - Config->SpeciesElitism;

		Specie->Genomes = CullingMethod::CullPopulation(Specie->Genomes, NumToCull, Config->CullingMethod);
		if (Specie->Genomes.IsEmpty())
			break;
	}

	// Generate offspring for the next generation
	for (auto& Specie : Species)
	{
		if (Specie->IsEmpty()) continue;

		// Select the best genome from the species to carry over
		auto BestGenome = Specie->GetBestGenome();

		// Generate offspring for the species
		int ReproductionCount = Math::Max(Specie->DesiredPopulationSize - Specie->GetNum(), 0);
		TArray<GenomePairing::Offspring> Offspring = GenomePairing::Reproduce(Specie->Genomes, ReproductionCount, Config);
		for (auto& Pairing : Offspring)
		{
			auto Child = Pairing.GetChild();
			Child->SpeciesID  = Specie->ID;
			Specie->AddGenome(Child);
		}
	}

	// Replace the old population with the new population
	Population.Reset();
	for (const auto& Specie : Species)
	{
		Population.Append(Specie->Genomes);
	}

	Generation++;

	int PopulationSize = Population.Num();
	int IntendedSize = Config->PopulationSize;
	if (Config->ReintroduceBestGenome && Generation % Config->ReintroductionPeriod == 0)
	{
		auto ReintroducedBest = std::make_shared<NEAT::Genome>(BestGenome);
		ReintroducedBest->ID = Genome::GenerateNewGenomeID();
		ReintroducedBest->Config = Config;
		ReintroducedBest->SpeciesID = 0;
		Population.Add(ReintroducedBest);
	}
}

// Mutates the offspring, with the Config settings (e.g. MutationRates) and the Mutations.h content	
void NEAT::Trainer::MutateOffspring()
{
	for (auto& Genome : Population)
	{
		if (Genome->bElite) continue; // Don't mutate the elites of each species
		if (Math::Random<double>(1.0) >= Config->MutationRate) continue; // Skip the mutation step if the mutation rate is not met
		Genome->Genotype.Mutate(Config); // Check the Config->MutationRates to see if we should perform each type of mutation
	}
}

// Clones the genome and then mutates it, with a single original copy
void NEAT::Trainer::RepopulateFromGenome(const GenomePtr& Genome) 
{
    // Clone the genome
    GenomePtr ClonedGenome = std::make_shared<NEAT::Genome>(*Genome);

	// Clear the population and add the cloned genome
	Population.Reset();
	Population.Add(ClonedGenome);

	// Mutate the genome and fill out an Offspring list as if initializing from scratch
	std::vector<GenomePairing::Offspring> InitialPopulation;
	for (unsigned Idx = 0; Idx < Config->PopulationSize; ++Idx)
	{
		InitialPopulation.push_back(GenomePairing::Offspring(Config, ClonedGenome));
	}

	for (auto& Pairing : InitialPopulation)
	{
		Population.Add(Pairing.GetChild());
	}

	// Create the initial species
	SpeciatePopulation();
}

// Clones the genome and then mutates it, with a single original copy
void NEAT::Trainer::LoadPopulation(const std::string& Filename)
{
    std::ifstream File(Filename);
    if (!File.is_open())
    {
        std::cout << "Failed to open file for loading the population." << std::endl;
        return;
    }

    std::string SerializedGenome;
    while (std::getline(File, SerializedGenome))
    {
        GenomePtr LoadedGenome = std::make_shared<NEAT::Genome>(Config);
        if (!LoadedGenome->Genotype.Deserialize(SerializedGenome))
        {
            std::cout << "Failed to deserialize the genome." << std::endl;
            return;
        }

        Population.Add(LoadedGenome);
    }

    File.close();
}

void NEAT::Trainer::SaveGenome(const std::string& Filename, const GenomePtr& Genome)
{
    std::ofstream File(Filename);
    if (File.is_open())
    {
		auto GenotypeCopy = Genome->Genotype;
		GenotypeCopy.Prune();
		GenotypeCopy.ReduceGeneKeys();
        File << GenotypeCopy.Serialize() << "\n";
        File.close();
    }
    else
    {
        std::cout << "Failed to open file for saving the genome." << std::endl;
    }
}

NEAT::GenomePtr NEAT::Trainer::LoadGenome(const std::string& Filename)
{
    std::ifstream File(Filename);
    if (!File.is_open())
    {
        std::cout << "Failed to open file for loading the genome." << std::endl;
        return nullptr;
    }

    std::string SerializedGenome;
    std::getline(File, SerializedGenome);

    GenomePtr LoadedGenome = std::make_shared<NEAT::Genome>(Config);
    if (!LoadedGenome->Genotype.Deserialize(SerializedGenome))
    {
        std::cout << "Failed to deserialize the genome." << std::endl;
        return nullptr;
    }

    File.close();
    return LoadedGenome;
}

// Saves the entire population to a file, in a human-readable format that can also be read back in later. Serializes the entire population to a file with the genome's Serialize() function
void NEAT::Trainer::SavePopulation(const std::string& Filename) 
{
	std::ofstream File(Filename);
	if (!File.is_open())  return;

	File << "Population Size: " << Population.Num() << "\n";
	for (const auto& Genome : Population) 
	{
		File << Genome->Genotype.Serialize() << "\n";
	}

	File.close();
}

void NEAT::Trainer::SaveBestGenome() // Saves the best genome to a file, in a human-readable format that can also be read back in later
{
    if (bHasBestGenome)
    {
        std::ofstream File("best_genome.txt");
        if (File.is_open())
        {
			auto GenotypeCopy = BestGenome.Genotype;
			GenotypeCopy.Prune();
			GenotypeCopy.ReduceGeneKeys();
            File << GenotypeCopy.Serialize() << "\n";
            File.close();
        }
        else
        {
            std::cout << "Failed to open file for saving the best genome." << std::endl;
        }
    }
    else
    {
        std::cout << "No best genome found." << std::endl;
    }
}

void NEAT::Trainer::Train() // Runs the training loop until ShouldContinueTraining returns false  
{
	// Create a unique filename with timestamp  
	time_t now = time(0);
	tm ltm;
	localtime_s(&ltm, &now);
	char timestamp[20];
	strftime(timestamp, 20, "%Y%m%d_%H%M%S", &ltm);
	std::string PopulationMetadata = "TrainingMetadata/population_info_" + std::string(timestamp) + ".json";

	bool bLogTiming = false;

	Initialize();
	PopulationReporter PopReporter(this);
	BestGenomeReporter BestReporter(this);
	while (ContinueTraining())
	{
		Benchmark::Timer EvaluationTimer("Evaluation", Config->LogEvaluation);
		EvaluatePopulation();
		EvaluationTimer.Stop(Config->LogEvaluation);

		Benchmark::Timer StagnationTimer("Stagnation", bLogTiming);
		CheckForStagnation();
		StagnationTimer.Stop(bLogTiming);

		Benchmark::Timer SpeciateTimer("Speciate", bLogTiming);
		SpeciatePopulation();
		SpeciateTimer.Stop(bLogTiming);

		if (Generation % 100 == 0) PopReporter.Report();
		//if (Generation % 10 == 0) BestReporter.Report();

		Benchmark::Timer ReproduceTimer("Reproduce", bLogTiming);
		ReproduceSpecies();
		ReproduceTimer.Stop(bLogTiming);

		Benchmark::Timer MutateTimer("Mutate", bLogTiming);
		MutateOffspring();
		MutateTimer.Stop(bLogTiming);

		/*if (Generation % 10 == 0)*/ SerializePopulationInfo(PopulationMetadata);
	}
}

NEAT::SpeciesPtr NEAT::Trainer::GetSpeciesByID(uint64 ID) const // Returns the species with the given ID
{
	for (const auto& Specie : Species)
	{
		if (Specie->ID == ID) return Specie;
	}
	return nullptr;
}

NEAT::GenomePtr NEAT::Trainer::GetGenomeByID(uint64 ID) const // Returns the genome with the given ID
{
	for (const auto& Genome : Population)
	{
		if (Genome->ID == ID) return Genome;
	}
	return nullptr;
}

void NEAT::Trainer::SerializePopulationInfo(const std::string& Filename)
{
	std::time_t Timestamp = std::time(0); // Get the current timestamp

	// Check if the file is empty  
	std::ifstream CheckFile(Filename);
	bool isEmpty = CheckFile.peek() == std::ifstream::traits_type::eof();
	CheckFile.close();

	if (isEmpty)
	{
		// Create the file and any necessary folders that don't already exist, without filesystem  
		std::ofstream File(Filename, std::ios_base::app);
		if (!File.is_open()) return;

		File << "["; // If empty, add a "[" to the beginning of the file
		// Serialize the population info to the file in JSON format
		File << "{\"timestamp\": " << Timestamp << ", \"species\": [" << std::endl;
		auto LastSpecies = Species.Last();
		for (const auto& Specie : Species)
		{
			File << "  {\"id\": " << Specie->ID << ", \"size\": " << Specie->Genomes.Num() << ", \"stagnation\": " << Specie->Stagnation << ", \"adjusted_fitness\": " << Specie->AdjustedFitness << "}";
			if (Specie == LastSpecies) File << std::endl;
			else File << "," << std::endl;
		}
		File << "]}]";
		File.close(); // Add this line to close the file and release memory
	}
	else
	{
		// If not empty, remove the last "]" and add a ","  
		std::fstream File(Filename, std::ios_base::in | std::ios_base::out);
		File.seekg(-1, std::ios_base::end);
		File << "," << std::endl;
		// Serialize the population info to the file in JSON format
		File << "{\"timestamp\": " << Timestamp << ", \"species\": [" << std::endl;
		auto LastSpecies = Species.Last();
		for (const auto& Specie : Species)
		{
			File << "  {\"id\": " << Specie->ID << ", \"size\": " << Specie->Genomes.Num() << ", \"stagnation\": " << Specie->Stagnation << ", \"adjusted_fitness\": " << Specie->AdjustedFitness << "}";
			if (Specie == LastSpecies) File << std::endl;
			else File << "," << std::endl;
		}
		File << "]}]";
		File.close(); // Add this line to close the file and release memory
	}
}