#pragma once

#include <memory>
#include <string>
#include "Types.h"
#include "Array.h"
#include "Activations.h"
#include "Aggregations.h"
#include "Reproduction.h"

// This configuration file is the central hub for a NeuroEvolution of Augmenting Topologies(NEAT) algorithm implementation.
// It defines the core parameters and settings that govern the behavior of the NEAT algorithm, including the structure of the neural networks, the evolutionary process, and the fitness evaluation.
// As the foundation of the NEAT algorithm, this configuration file should be used as a reference point for the implementation of other modules, ensuring consistency and coherence throughout the project.
// By modifying the parameters and settings in this file, developers can fine - tune the performance of the NEAT algorithm and adapt it to specific problem domains.

#define INDEX_NONE -1

namespace NEAT
{
	// Configuration class  
	class Config
	{
	public:
		///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		// General settings  
		///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		// Population size: The number of genomes in the population. A higher value allows for more diverse solutions but increases computational cost.  
		uint64 PopulationSize = 137;

		// Number of generations to run: The number of generations to evolve the population. A higher value allows for more complex solutions but increases computational cost.  
		uint64 MaxGenerations = 1000;

		// Random seed for initializing random number generator: The seed value used to initialize the random number generator. A different value will result in different evolutionary paths.  
		int RandomSeed = 137;

		// Enable or disable verbose mode: A flag to enable or disable verbose mode, which controls the level of output during evolution. Enabling verbose mode can be useful for debugging but may slow down evolution.  
		bool VerboseMode = false;

		// Output directory for saving logs and output files: The directory where logs and output files will be saved. Make sure the directory exists and is writable.  
		std::string OutputDirectory = ".";

		// Interval at which to save checkpoints of the population: The interval at which to save checkpoints of the population. A lower value allows for more frequent checkpoints but increases disk usage.  
		int CheckpointInterval = 10;

		// Stopping condition: if any genome reaches this fitness, the algorithm will stop.
		double StoppingFitness = 0.0;

		// Whether to reset network activations between evaluations; this is useful for problems where the network state should/shouldn't persist between evaluations.
		bool ResetNetworkActivations = true;

		int MultithreadedEvaluation = 1;
		int NumThreads = 16;

		bool ReintroduceBestGenome = true;
		int ReintroductionPeriod = 25;

		///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		// Genome settings  
		///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		// Number of input neurons: The number of input neurons in the neural network. This value should match the number of inputs in the problem.  
		int NumInputs = 0;

		// Number of output neurons: The number of output neurons in the neural network. This value should match the number of outputs in the problem.  
		int NumOutputs = 0;

		// Number of hidden neurons: The number of hidden neurons in the neural network. A higher value allows for more complex solutions but may increase computational cost.
		int NumHidden = 0;

		// Initial connection state: The initial state of connections in the neural network. This value determines the initial connectivity of the network.
		EInitialTopology InitialTopology = EInitialTopology::None;

		// Probability of an initial connection between any given input and output, when using EInitialTopology::Sparse
		double InitialConnectionProbability = 0.6;

		///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		// Genome Distance Calculation settings 
		///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		// Distance threshold for genome distance calculation: The threshold value used to determine the maximum distance between two genomes for them to be considered compatible. A higher value allows for more diverse genomes.
		double SpeciationDistanceThreshold = 12.75;

		// Excess coefficient for genome distance calculation: The coefficient value used for calculating the distance between genomes based on excess genes. A higher value increases the sensitivity of the distance calculation.
		double ExcessGeneCoefficient = 0.95;

		// Disjoint coefficient for genome distance calculation: The coefficient value used for calculating the distance between genomes based on disjoint genes. A higher value increases the sensitivity of the distance calculation.
		double DisjointGeneCoefficient = 0.75;

		// Matching coefficient for genome distance calculation: The coefficient value used for calculating the distance between genomes based on matching genes. A higher value increases the sensitivity of the distance calculation.
		double MatchingGeneCoefficient = 0.65;

		// Weight coefficient for genome distance calculation: The coefficient value used for calculating the distance between genomes based on weight differences. A higher value increases the sensitivity of the distance calculation.
		double DistanceExponent = 1.0;

		// Distance method for genome distance calculation: The method used for calculating the distance between genomes (0: Euclidean, 1: Manhattan, 2: Chebyshev). Different methods may be more suitable for different problems.
		EDistance DistanceMethod = EDistance::Euclidean;

		///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		// Species settings  
		///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		// Species elitism: The number of elite genomes to preserve in each species. A higher value allows for more competitive species but may increase computational cost.
		int SpeciesElitism = 2;

		// Species survival threshold: The threshold value used to determine the number of genomes to survive in each species. A higher value allows for more competitive species but may increase computational cost.
		double SurvivalRate = 0.8;

		// Maximum stagnation generations: The maximum number of generations a species can remain stagnant before being penalized. A higher value allows for more stable species but may not be sufficient for complex problems.
		unsigned MaxStagnation = 27;

		// If true, species will be represented by their most fit genome each iteration, otherwise representative is chosen at random. The difference being that the most fit genome is more likely to be chosen as a parent.
		bool ChooseBestRepresentative = false;

		// Minimum species size
		int MinSpeciesSize = 5;

		///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		// Mutation settings  
		///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		// Mutation rate: The overall mutation rate for the population. A higher value allows for more exploration but may increase computational cost.  
		double MutationRate = 0.8;

		// Add node rate: The rate at which new neurons are added to the neural network during evolution. A higher value allows for more complex solutions but may increase computational cost.  
		double AddNodeMutationRate = 0.04;

		// Remove node rate: The rate at which neurons are removed from the neural network during evolution. A higher value allows for simpler solutions but may not be sufficient for complex problems.
		double RemoveNodeMutationRate = 0.0;

		// Add connection mutation rate: The rate at which new connections are added to the neural network during evolution. A higher value allows for more complex solutions but may increase computational cost.  
		double AddConnectionMutationRate = 0.08;

		// Remove neuron mutation rate: The rate at which neurons are removed from the neural network during evolution. A higher value allows for simpler solutions but may not be sufficient for complex problems.  
		double RemoveConnectionMutationRate = 0.01;

		// Weight mutation rate: The rate at which weights are mutated during evolution. A higher value allows for more exploration but may increase computational cost.  
		double WeightMutationRate = 0.6;

		// Weight mutation power: The power value used to control the magnitude of weight mutations. A higher value allows for more significant mutations but may increase computational cost.  
		double WeightMutationVariance = 1.0;

		// Minimum connection weight: The minimum weight value for connections between neurons. A lower value allows for weaker connections but may not be sufficient for complex problems.
		double MinConnectionWeight = -100.0;

		// Maximum connection weight: The maximum weight value for connections between neurons. A higher value allows for stronger connections but may increase computational cost.
		double MaxConnectionWeight = 100.0;

		// Minimum node bias: The minimum bias value for neurons in the neural network. A smaller value allows for weaker biases but may not be sufficient for complex problems.
		double MinNodeBias = -100.0;

		// Maximum node bias: The maximum bias value for neurons in the neural network. A larger value allows for stronger biases but may increase computational cost.
		double MaxNodeBias = 100.0;

		// Bias mutation rate: The rate at which biases are mutated during evolution. A higher value allows for more exploration but may increase computational cost.  
		double BiasMutationRate = 0.5;

		// Bias mutation power: The power value used to control the magnitude of bias mutations. A higher value allows for more significant mutations but may increase computational cost.  
		double BiasMutationVariance = 1.0;

		// Activation function mutation rate: The rate at which activation functions are mutated during evolution. A higher value allows for more exploration but may increase computational cost.  
		double ActivationFunctionMutationRate = 0.02;

		// Default activation function: The default activation function used for neurons in the neural network. This value should match the problem domain.
		EActivation DefaultActivationFunction = EActivation::Sigmoid;

		// Supported activation functions: The list of activation functions supported by the NEAT algorithm. Different activation functions may be more suitable for different problems.
		TArray<EActivation> SupportedActivationFunctions = 
		{ 
			EActivation::Sigmoid, 
			EActivation::Tanh, 
			EActivation::Relu, 
			EActivation::Absolute, 
			EActivation::Step, 
			EActivation::Gaussian, 
			EActivation::Inverse, 
			EActivation::Linear ,
			EActivation::BentIdentity,
			EActivation::Swish,
			EActivation::LeakyRelu,
			EActivation::BipolarSigmoid,
		};

		// Aggregation function mutation rate: The rate at which aggregation functions are mutated during evolution. A higher value allows for more exploration but may increase computational cost.  
		double AggregationFunctionMutationRate = 0.01;

		// Default aggregation function: The default aggregation function used for neurons in the neural network. This value should match the problem domain.
		EAggregation DefaultAggregationFunction = EAggregation::Product;

		// Supported aggregation functions: The list of aggregation functions supported by the NEAT algorithm. Different aggregation functions may be more suitable for different problems.
		TArray<EAggregation> SupportedAggregationFunctions = 
		{ 
			EAggregation::Mean, 
			EAggregation::Median, 
			EAggregation::Sum, 
			EAggregation::Max, 
			EAggregation::Min, 
			EAggregation::Count, 
			EAggregation::Product 
		};

		// Enable mutation rate: The rate at which enable mutations occur during evolution. A higher value allows for more exploration but may increase computational cost.  
		double EnableMutationRate = 0.03;

		// Single mutation: A flag to enable or disable single mutations. Enabling single mutations can be useful for problems with simple solutions but may not be sufficient for complex problems.
		bool SingleMutation = false;

		///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		// Reproduction settings  
		///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		// Crossover rate: The rate at which crossover occurs during evolution. A higher value allows for more exploration but may increase computational cost.  
		double CrossoverRate = 0.8;

		// Crossover method: The method used for crossover (0: uniform, 1: single-point, 2: two-point). Different methods may be more suitable for different problems.  
		ECrossoverType CrossoverType = ECrossoverType::SinglePoint;

		// Number of crossover points for multi-point crossover: The number of crossover points used for multi-point crossover. A higher value allows for more complex solutions but may increase computational cost.  
		int CrossoverPoints = 2;

		// Culling method: The method used to select genomes for reproduction (roulette wheel, Random, Boltzmann, etc). Different methods may be more suitable for different problems.
		ECullingMethod CullingMethod = ECullingMethod::Elitism;

		// Pairing method: The method used to pair surviving genomes together for crossover reproduction
		EGenomePairing PairingMethod = EGenomePairing::Random;

		///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		// Logging settings  
		///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		// Log network evaluation steps: A flag to enable or disable logging of network evaluation steps. Enabling logging can be useful for debugging but may increase disk usage.
		bool LogEvaluation = false;

		// Log fitness values: A flag to enable or disable logging of fitness values. Enabling logging can be useful for debugging but may increase disk usage.  
		bool LogFitness = false;

		// Log genome information: A flag to enable or disable logging of genome information. Enabling logging can be useful for debugging but may increase disk usage.  
		bool LogGenomes = false;

		// Log file name: The file name used for logging. Make sure the file exists and is writable.  
		std::string LogFile = "";

		// Constructor  
		Config();

		// Load configuration from file  
		void LoadFromFile(const std::string& Filename);

		// Save configuration to file  
		void SaveToFile(const std::string& Filename);

		static ConfigPtr CreateDefaultConfig() 
		{
			return std::make_shared<Config>();
		}
	};

	//using ConfigPtr = std::shared_ptr<const NEAT::Config>;
	using ConfigPtr = std::shared_ptr<NEAT::Config>;
} // namespace NEAT