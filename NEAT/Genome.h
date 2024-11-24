#pragma once

#include <string>  
#include <memory>
#include <sstream>
#include "Genotype.h"

namespace NEAT 
{
	class NeuralNetwork;
	using NeuralNetworkPtr = std::shared_ptr<NeuralNetwork>;

	class Genome;
	using GenomePtr = std::shared_ptr<Genome>;

	class Genome 
	{
	public:
		uint64 ID = 0;
		uint64 SpeciesID = 0;
		NEAT::Genotype Genotype;
		ConfigPtr Config = nullptr;
		double AdjustedFitness = 0.0;
		double Fitness = 0.0;
		bool bElite = false;

		static unsigned GenerateNewGenomeID() { static unsigned NewestID = 0; return ++NewestID; }
		Genome(Genome&& Other) noexcept : ID(std::move(Other.ID)), SpeciesID(std::move(Other.SpeciesID)), Genotype(std::move(Other.Genotype)), Config(std::move(Other.Config)), AdjustedFitness(std::move(Other.AdjustedFitness)), Fitness(std::move(Other.Fitness)), bElite(std::move(Other.bElite)) { }
		Genome(const Genome& Other) : ID(Other.ID), SpeciesID(Other.SpeciesID), Genotype(Other.Genotype), Config(Other.Config), AdjustedFitness(Other.AdjustedFitness), Fitness(Other.Fitness), bElite(Other.bElite) { }
		Genome(const ConfigPtr& InConfig, const NEAT::Genotype& InGenotype) : Config(InConfig), Genotype(InGenotype) { }
		Genome(const ConfigPtr& InConfig) : Config(InConfig) { }
		Genome() = default;
		~Genome() { }


		Genome& operator=(const Genome& Other)
		{
			ID = Other.ID;
			SpeciesID = Other.SpeciesID;
			Genotype = Other.Genotype;
			AdjustedFitness = Other.AdjustedFitness;
			Fitness = Other.Fitness;
			bElite = Other.bElite;
			return *this;
		}

		Genome& operator=(Genome&& Other) noexcept
		{
			ID = std::move(Other.ID);
			SpeciesID = std::move(Other.SpeciesID);
			Genotype = std::move(Other.Genotype);
			AdjustedFitness = std::move(Other.AdjustedFitness);
			Fitness = std::move(Other.Fitness);
			bElite = std::move(Other.bElite);
			return *this;
		}

		int GetNumInputs() const { return Config ? (Config->NumInputs + 1) : 0; } // +1 for bias node
		int GetNumOutputs() const { return Config ? Config->NumOutputs : 0; }
		int GetNumHidden() const { return GetNumNodes() - GetNumInputs() - GetNumOutputs(); }

		int GetNumGenes() const { return Genotype.Connections.Num() + Genotype.Nodes.Num(); }
		int GetNumConnections() const { return Genotype.Connections.Num(); }
		int GetNumNodes() const { return Genotype.Nodes.Num(); }

		TArray<ConnectionGene> GetConnections() const { return Genotype.Connections.GetValues(); }
		TArray<NodeGene> GetNodes() const { return Genotype.Nodes.GetValues(); }

		const ConnectionGene* GetConnectionByID(uint64 InID) const;
		const NodeGene* GetNodeByID(uint64 InID) const;
		ConnectionGene* GetConnectionByID(uint64 InID);
		NodeGene* GetNodeByID(uint64 InID);

		NeuralNetworkPtr CreateNeuralNetwork() const;
	};
} // namespace NEAT  