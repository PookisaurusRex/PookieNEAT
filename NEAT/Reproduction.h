#pragma once

#include <vector>  
#include <string>  
#include <memory>
#include "Array.h"

namespace NEAT
{
	class Genome;
	using GenomePtr = std::shared_ptr<NEAT::Genome>;

	class Config;
	//using ConfigPtr = std::shared_ptr<const NEAT::Config>;
	using ConfigPtr = std::shared_ptr<NEAT::Config>;

	enum class EInitialTopology
	{
		None, // No initial connections between neurons
		Sparse, // Sparse initial connections between neurons
		Full, // Full initial connections between neurons
		Tree, // Tree-like initial connections between neurons
	};

	namespace InitialTopology
	{
		std::string ToString(EInitialTopology Topology);

		static EInitialTopology FromString(const std::string& Topology)
		{
			if (Topology == "EInitialTopology::None") return EInitialTopology::None;
			if (Topology == "EInitialTopology::Sparse") return EInitialTopology::Sparse;
			if (Topology == "EInitialTopology::Full") return EInitialTopology::Full;
			if (Topology == "EInitialTopology::Tree") return EInitialTopology::Tree;
			return EInitialTopology::None;
		}

		GenomePtr InitializeFromParents(const GenomePtr& Parent1, const GenomePtr& Parent2); // Initialize a genome from two parents
		GenomePtr InitializeFromParent(const GenomePtr& Parent); // Initialize a genome from a single parent
		GenomePtr InitializeGenome(const ConfigPtr& Config); // Initialize a single genome from Config defaults

		void None(GenomePtr Genome); // Initialize connections for an initially unconnected neural network
		void Sparse(GenomePtr Genome); // Initialize connections for a sparsely connected neural network
		void Full(GenomePtr Genome); // Initialize connections from every Input to every Output, from every Input to every Hidden, and from every Hidden to every Output
		void Tree(GenomePtr Genome); // Initialize connections for a tree-like neural network
	}

	enum class ECrossoverType
	{
		Uniform, // Uniform crossover method
		Average, // Average crossover method
		Multipoint, // Multi-point crossover method
		SinglePoint, // Single-point crossover method
		TwoPoint, // Two-point crossover method
	};

	namespace CrossoverType
	{
		static std::string ToString(ECrossoverType Method)
		{
			switch (Method)
			{
			case ECrossoverType::Uniform: return "ECrossoverMethod::Uniform";
			case ECrossoverType::Multipoint: return "ECrossoverMethod::Multipoint";
			case ECrossoverType::SinglePoint: return "ECrossoverMethod::SinglePoint";
			case ECrossoverType::TwoPoint: return "ECrossoverMethod::TwoPoint";
			default: return "ECrossoverMethod::Unknown";
			}
		}

		static ECrossoverType FromString(const std::string& Method)
		{
			if (Method == "ECrossoverMethod::Uniform") return ECrossoverType::Uniform;
			if (Method == "ECrossoverMethod::Multipoint") return ECrossoverType::Multipoint;
			if (Method == "ECrossoverMethod::SinglePoint") return ECrossoverType::SinglePoint;
			if (Method == "ECrossoverMethod::TwoPoint") return ECrossoverType::TwoPoint;
			return ECrossoverType::Uniform;
		}

		GenomePtr Uniform(const GenomePtr& Parent1, const GenomePtr& Parent2); // Initialize a genome from two parents using Uniform crossover
		GenomePtr SinglePoint(const GenomePtr& Parent1, const GenomePtr& Parent2); // Initialize a genome from two parents using Single Point crossover
		GenomePtr TwoPoint(const GenomePtr& Parent1, const GenomePtr& Parent2); // Initialize a genome from two parents using Two Point crossover
		GenomePtr Multipoint(const GenomePtr& Parent1, const GenomePtr& Parent2); // Initialize a genome from two parents using Multipoint crossover
	}

	enum class ECullingMethod
	{
		RouletteWheel, // Selects individuals based on their fitness proportion, may not work well with very large fitness differences.  
		Random, // Fast, simple, and efficient. Encourages diversity. May not always select the best individuals if Population size too small.
		Boltzmann, // Similar to roulette wheel selection, but uses a temperature parameter to control the selection pressure.
		Elitism, // Selects the top N individuals based on their fitness.
		Rank, // Similar to roulette wheel selection, but uses a ranking system to reduce the impact of very large fitness differences.
	};

	namespace CullingMethod
	{
		std::string ToString(ECullingMethod Method);

		ECullingMethod FromString(const std::string& Method);

		TArray<GenomePtr> CullPopulation(const TArray<GenomePtr>& Population, size_t N, ECullingMethod Method);
		TArray<GenomePtr> RouletteWheel(const TArray<GenomePtr>& Population, size_t N);
		TArray<GenomePtr> Random(const TArray<GenomePtr>& Population, size_t N);
		TArray<GenomePtr> Boltzmann(const TArray<GenomePtr>& Population, size_t N);
		TArray<GenomePtr> Elitism(const TArray<GenomePtr>& Population, size_t N);
		TArray<GenomePtr> Rank(const TArray<GenomePtr>& Population, size_t N);
	}

	enum class EGenomePairing
	{
		Random, // Pair genomes randomly  
		Fittest, // Pair the fittest genomes with each other  
		Weakest, // Pair the weakest genomes with each other  
		Alternating, // Pair the fittest genome with the weakest, the second fittest with the second weakest, and so on  
		SimilarFitness, // Pair genomes with similar fitness values  
		DissimilarFitness, // Pair genomes with dissimilar fitness values  
		Proximity, // Pair genomes based on their proximity in the search space  
		Diversity, // Pair genomes to maximize diversity in the offspring  
	};

	namespace GenomePairing
	{
		static std::string ToString(EGenomePairing Method)
		{
			switch (Method)
			{
			case EGenomePairing::Random: return "EGenomePairing::Random";
			case EGenomePairing::Fittest: return "EGenomePairing::Fittest";
			case EGenomePairing::Weakest: return "EGenomePairing::Weakest";
			case EGenomePairing::Alternating: return "EGenomePairing::Alternating";
			case EGenomePairing::SimilarFitness: return "EGenomePairing::SimilarFitness";
			case EGenomePairing::DissimilarFitness: return "EGenomePairing::DissimilarFitness";
			case EGenomePairing::Proximity: return "EGenomePairing::Proximity";
			case EGenomePairing::Diversity: return "EGenomePairing::Diversity";
			default: return "EGenomePairing::Unknown";
			}
		}

		static EGenomePairing FromString(const std::string& Method)
		{
			if (Method == "EGenomePairing::Random") return EGenomePairing::Random;
			if (Method == "EGenomePairing::Fittest") return EGenomePairing::Fittest;
			if (Method == "EGenomePairing::Weakest") return EGenomePairing::Weakest;
			if (Method == "EGenomePairing::Alternating") return EGenomePairing::Alternating;
			if (Method == "EGenomePairing::SimilarFitness") return EGenomePairing::SimilarFitness;
			if (Method == "EGenomePairing::DissimilarFitness") return EGenomePairing::DissimilarFitness;
			if (Method == "EGenomePairing::Proximity") return EGenomePairing::Proximity;
			if (Method == "EGenomePairing::Diversity") return EGenomePairing::Diversity;
			return EGenomePairing::Random;
		}

		class Offspring
		{
		public:
			Offspring(ConfigPtr InConfig, GenomePtr InParent1, GenomePtr InParent2) : Config(std::move(InConfig)), Parent1(std::move(InParent1)), Parent2(std::move(InParent2)) { }
			Offspring(ConfigPtr InConfig, GenomePtr InParent1) : Config(std::move(InConfig)), Parent1(std::move(InParent1)), Parent2(nullptr) { }
			Offspring(ConfigPtr InConfig) : Config(std::move(InConfig)), Parent1(nullptr), Parent2(nullptr) { }
			ConfigPtr Config = nullptr;
			GenomePtr Parent1 = nullptr;
			GenomePtr Parent2 = nullptr;
			GenomePtr GetChild() const;
		};

		TArray<Offspring> Random(const TArray<GenomePtr>& Population, int ReproductionCount, const ConfigPtr& Config);
		TArray<Offspring> Fittest(const TArray<GenomePtr>& Population, int ReproductionCount, const ConfigPtr& Config);
		TArray<Offspring> Weakest(const TArray<GenomePtr>& Population, int ReproductionCount, const ConfigPtr& Config);
		TArray<Offspring> Alternating(const TArray<GenomePtr>& Population, int ReproductionCount, const ConfigPtr& Config);
		TArray<Offspring> SimilarFitness(const TArray<GenomePtr>& Population, int ReproductionCount, const ConfigPtr& Config);
		TArray<Offspring> DissimilarFitness(const TArray<GenomePtr>& Population, int ReproductionCount, const ConfigPtr& Config);
		TArray<Offspring> Proximity(const TArray<GenomePtr>& Population, int ReproductionCount, const ConfigPtr& Config);
		TArray<Offspring> Diversity(const TArray<GenomePtr>& Population, int ReproductionCount, const ConfigPtr& Config);

		// Uses information from the Config to determine the best pairing method, crossover rate, population size, 
		// and other parameters, to generate a list of offspring to add to the population. If crossover does not 
		// happen, the offspring should be 50/50 chance of random initializations or asexual reproduction with mutation.
		TArray<Offspring> Reproduce(const TArray<GenomePtr>& Population, int ReproductionCount, const ConfigPtr& Config);
	}

	enum class EDistance
	{
		Euclidean, // This method calculates the distance between two genomes as the square root of the sum of the squared differences between corresponding connection weights.
		Manhattan, // This method calculates the distance between two genomes as the sum of the absolute differences between corresponding connection weights.
		Chebyshev, // This method calculates the distance between two genomes as the maximum absolute difference between corresponding connection weights.
	};

	namespace Distance
	{
		static std::string ToString(EDistance Method)
		{
			switch (Method)
			{
			case EDistance::Euclidean: return "EDistance::Euclidean";
			case EDistance::Manhattan: return "EDistance::Manhattan";
			case EDistance::Chebyshev: return "EDistance::Chebyshev";
			default: return "EDistance::Unknown";
			}
		}

		static EDistance FromString(const std::string& Method)
		{
			if (Method == "EDistance::Euclidean") return EDistance::Euclidean;
			if (Method == "EDistance::Manhattan") return EDistance::Manhattan;
			if (Method == "EDistance::Chebyshev") return EDistance::Chebyshev;
			return EDistance::Euclidean;
		}

		double Euclidean(const GenomePtr& Genome1, const GenomePtr& Genome2, const ConfigPtr& Config);
		double Manhattan(const GenomePtr& Genome1, const GenomePtr& Genome2, const ConfigPtr& Config);
		double Chebyshev(const GenomePtr& Genome1, const GenomePtr& Genome2, const ConfigPtr& Config);

		double Calculate(const GenomePtr& Genome1, const GenomePtr& Genome2, const ConfigPtr& Config);
	}
}