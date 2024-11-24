#include "Network.h"
#include "Aggregations.h"
#include "Activations.h"
#include "Genome.h"
#include "Genes.h"
#include "Utils.h"
#include "Math.h"
#include <limits>
#include <cmath>

namespace NEAT {

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
NeuralNetwork::NeuronNode::NeuronNode(uint64 InID, EActivation InActivation, EAggregation InAggregation, double InBias)
	: ID(InID), Activation(0.0), ActivationType(InActivation), AggregationType(InAggregation), Bias(InBias)
{
}

NeuralNetwork::NeuronNode::NeuronNode(NEAT::NodeGene Node) : NeuronNode(Node.ID, Node.Activation, Node.Aggregation, Node.Bias)
{
}

NeuralNetwork::NeuronNode::~NeuronNode()
{
}

void NeuralNetwork::NeuronNode::Activate(const NeuralNetwork* Network)
{
	double NewActivation = Aggregation::Aggregate(Network->GetWeightedInputs(ID), AggregationType);
	Activation = Activation::Activate(NewActivation, ActivationType);
	Activation = Math::IsNaN(Activation) ? 0.f : Activation;
	Activation = Math::IsFinite(Activation) ? Activation : 0.f;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
double NeuralNetwork::NeuronConnection::GetWeightedInput() const
{
	return (Input->Activation + Input->Bias) * Weight;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
NeuralNetwork::NeuralNetwork(GenomePtr Genome) : Config(Genome->Config)
{
	const auto& InputNodeIDs = Genome->Genotype.GetFilteredNodeKeys([](const auto& Pair) { return Pair.second.Type == ENodeType::Input; });
	const auto& HiddenNodeIDs = Genome->Genotype.GetFilteredNodeKeys([](const auto& Pair) { return Pair.second.Type == ENodeType::Hidden; });
	const auto& OutputNodeIDs = Genome->Genotype.GetFilteredNodeKeys([](const auto& Pair) { return Pair.second.Type == ENodeType::Output; });

	for (const auto& NodeID : InputNodeIDs)
	{
		const auto& Node = Genome->Genotype.Nodes[NodeID];
		InputNeurons.Add(std::make_shared<NeuronNode>(Node));
	}

	for (const auto& NodeID : HiddenNodeIDs)
	{
		const auto& Node = Genome->Genotype.Nodes[NodeID];
		HiddenNeurons.Add(std::make_shared<NeuronNode>(Node));
	}

	for (const auto& NodeID : OutputNodeIDs)
	{
		const auto& Node = Genome->Genotype.Nodes[NodeID];
		OutputNeurons.Add(std::make_shared<NeuronNode>(Node));
	}

	for (const auto& Connection : Genome->Genotype.Connections)
	{
		auto InputNeuron = GetNeuronByID(Connection.second.Input);
		auto OutputNeuron = GetNeuronByID(Connection.second.Output);
		Connections.Add(std::make_shared<NeuronConnection>(InputNeuron, OutputNeuron, Connection.second.Weight));
	}
}

TArray<double> NeuralNetwork::GetWeightedInputs(uint64 NeuronID) const
{
	TArray<double> Inputs;
	for (const auto& Connection : Connections)
	{
		if (Connection->Output->ID == NeuronID)
		{
			Inputs.Add(Connection->GetWeightedInput());
		}
	}
	return Inputs;
}

TArray<double> NeuralNetwork::Evaluate(const TArray<double>& Inputs)
{
	if (!Config) return {}; // Invalid configuration

	if (Config->ResetNetworkActivations)
	{
		for (const auto& Neuron : InputNeurons)	{ Neuron->Activation = 0.0;	}
		for (const auto& Neuron : HiddenNeurons) { Neuron->Activation = 0.0; }
		for (const auto& Neuron : OutputNeurons) { Neuron->Activation = 0.0; }
	}

	if (InputNeurons.Num() != (Inputs.Num() + 1)) return {}; // Invalid input size

	for (int Idx = 0, StopIdx = Inputs.Num(); Idx != StopIdx; ++Idx)
	{
		InputNeurons[Idx]->Activation = Inputs[Idx];
	}
	InputNeurons.Last()->Activation = 1.0; // GBX:GVand - Activate bias node

	for (auto Neuron : HiddenNeurons) 
		Neuron->Activate(this);
	for (auto Neuron : OutputNeurons) 
		Neuron->Activate(this);

	TArray<double> Outputs;
	Outputs.SetNum(OutputNeurons.Num());
	for (int Idx = 0, StopIdx = OutputNeurons.Num(); Idx != StopIdx; ++Idx)
	{
		Outputs[Idx] = OutputNeurons[Idx]->Activation;
	}

	return Outputs;
}

NeuralNetwork::NeuronPtr NeuralNetwork::GetNeuronByID(uint64 ID) const
{
	for (const auto& Neuron : InputNeurons) if (Neuron->ID == ID) return Neuron;
	for (const auto& Neuron : HiddenNeurons) if (Neuron->ID == ID) return Neuron;
	for (const auto& Neuron : OutputNeurons) if (Neuron->ID == ID) return Neuron;
	return nullptr;
}

} // namespace NEAT