#pragma once

#include <unordered_map>
#include <vector>  
#include <memory>
#include "Genes.h"
#include "Config.h"
#include "Genome.h"
#include "Array.h"
#include "Utils.h"

namespace NEAT
{
	class NeuralNetwork
	{
	public:
		struct NeuronNode
		{
			uint64 ID = 0;
			double Bias = 0.0;
			double Activation = 0.0;
			EActivation ActivationType;
			EAggregation AggregationType;

			NeuronNode(uint64 InID, EActivation InActivationType, EAggregation InAggregationType, double InBias);
			NeuronNode(NEAT::NodeGene Node);
			virtual ~NeuronNode();

			void Activate(const class NeuralNetwork* Network);
		};
		using NeuronPtr = std::shared_ptr<NeuronNode>;

		struct NeuronConnection
		{
			NeuronPtr Input = nullptr;
			NeuronPtr Output = nullptr;
			double Weight = 1.0;

			double GetWeightedInput() const;

			explicit NeuronConnection(const NeuronPtr& InInput, const NeuronPtr& InOutput, double InWeight) : Input(InInput), Output(InOutput), Weight(InWeight) 
			{ 
				if (!Input) BREAKPOINT();
				if (!Output) BREAKPOINT();
			}

			NeuronConnection(const NeuronPtr& InInput, const NeuronPtr& InOutput) : Input(Input), Output(InOutput) { }
		};
		using ConnectionPtr = std::shared_ptr<NeuronConnection>;

		ConfigPtr Config = nullptr;
		TArray<NeuronPtr> InputNeurons;
		TArray<NeuronPtr> HiddenNeurons;
		TArray<NeuronPtr> OutputNeurons;
		TArray<ConnectionPtr> Connections;

		NeuralNetwork(GenomePtr Genome);
		virtual ~NeuralNetwork() {}

		TArray<double> GetWeightedInputs(uint64 NeuronID) const;
		TArray<double> Evaluate(const TArray<double>& Inputs);
		NeuronPtr GetNeuronByID(uint64 ID) const;
	};
}