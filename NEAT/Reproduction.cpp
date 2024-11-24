#include "Reproduction.h"
#include "Genome.h"
#include "Config.h"
#include "Genes.h"
#include "Utils.h"
#include "Math.h"
#include <algorithm>
#include <unordered_map>

namespace NEAT {
namespace InitialTopology {
std::string ToString(EInitialTopology Topology)
{
	switch (Topology)
	{
	case EInitialTopology::None: return "EInitialTopology::None";
	case EInitialTopology::Sparse: return "EInitialTopology::Sparse";
	case EInitialTopology::Full: return "EInitialTopology::Full";
	case EInitialTopology::Tree: return "EInitialTopology::Tree";
	default: return "EInitialTopology::Unknown";
	}
}

GenomePtr InitializeFromParents(const GenomePtr& Parent1, const GenomePtr& Parent2) // Initialize a genome from two parents
{
	const auto Config = Parent1->Config;
	GenomePtr Genome = std::make_shared<NEAT::Genome>(Config);
	Genome->SpeciesID = GetRandomInt(0, 1) ? Parent1->SpeciesID : Parent2->SpeciesID;
	Genome->ID = NEAT::Genome::GenerateNewGenomeID();

	switch (Config->CrossoverType)
	{
	case ECrossoverType::Uniform: return CrossoverType::Uniform(Parent1, Parent2);
	case ECrossoverType::SinglePoint: return CrossoverType::SinglePoint(Parent1, Parent2);
	case ECrossoverType::TwoPoint: return CrossoverType::TwoPoint(Parent1, Parent2);
	case ECrossoverType::Multipoint: return CrossoverType::Multipoint(Parent1, Parent2);
	default: return CrossoverType::Uniform(Parent1, Parent2);
	}
	return nullptr;
}

GenomePtr InitializeFromParent(const GenomePtr& Parent) // Initialize a genome from a single parent
{
	GenomePtr Genome = std::make_shared<NEAT::Genome>(Parent->Config);
	Genome->ID = NEAT::Genome::GenerateNewGenomeID();
	Genome->SpeciesID = Parent->SpeciesID;
	Genome->Genotype = Parent->Genotype;
	return std::move(Genome);
}

GenomePtr InitializeGenome(const ConfigPtr& Config) // Initialize a single genome from Config defaults
{
	GenomePtr Genome = std::make_shared<NEAT::Genome>(Config);
	Genome->ID = NEAT::Genome::GenerateNewGenomeID();

	switch (Config->InitialTopology)
	{
	case EInitialTopology::None: InitialTopology::None(Genome); break;
	case EInitialTopology::Sparse: InitialTopology::Sparse(Genome); break;
	case EInitialTopology::Full: InitialTopology::Full(Genome); break;
	case EInitialTopology::Tree: InitialTopology::Tree(Genome); break;
	default: InitialTopology::None(Genome); break;
	}

	return std::move(Genome);
}

void None(GenomePtr Genome) // Initialize connections for an initially unconnected neural network
{
	auto& Connections = Genome->Genotype.Connections;
	auto& Nodes = Genome->Genotype.Nodes;
	const auto Config = Genome->Config;
	for (uint64 Idx = 0, StopIdx = (Config->NumInputs + 1); Idx != StopIdx; ++Idx) // Initialize Input nodes
	{
		Nodes[Idx] = NodeGene(Idx, ENodeType::Input, EActivation::Linear);
	}
	for (int Idx = 0, StopIdx = Config->NumOutputs; Idx != StopIdx; ++Idx) // Initialize Output nodes
	{
		uint64 NodeID = Idx + Config->NumInputs + 1; // +1 for the bias node
		Nodes[NodeID] = NodeGene(NodeID, ENodeType::Output, Config->DefaultActivationFunction, Config->DefaultAggregationFunction);
	}
	for (int Idx = 0, StopIdx = Config->NumHidden; Idx != StopIdx; ++Idx) // Initialize Hidden nodes
	{
		uint64 NodeID = Idx + Config->NumInputs + Config->NumOutputs + 1; // +1 for the bias node
		Nodes[NodeID] = NodeGene(NodeID, ENodeType::Hidden, Config->DefaultActivationFunction, Config->DefaultAggregationFunction);
	}
}

void Sparse(GenomePtr Genome) // Initialize connections for a sparsely connected neural network
{
	None(Genome); // Start with an unconnected network
	auto& Connections = Genome->Genotype.Connections;
	auto& Nodes = Genome->Genotype.Nodes;
	const auto Config = Genome->Config;
	// Connect each input node to each hidden node 
	for (int Idx = 0, StopIdx = (Config->NumInputs + 1); Idx != StopIdx; ++Idx) // +1 for the bias node
	{
		auto InputNode = Genome->GetNodeByID(Idx); // Get the input node
		for (int Jdx = 0, StopJdx = Config->NumHidden; Jdx != StopJdx; ++Jdx)
		{
			if (GetRandomDouble(0.0, 1.0) >= Config->InitialConnectionProbability) continue; // Connect each input node to each output node with a probability of InitialConnectionProbability
			auto HiddenNode = Genome->GetNodeByID(Jdx + Config->NumInputs + Config->NumOutputs + 1); // Get the hidden node
			auto ConnectionID = Innovations.GetInnovationID(EMutationType::AddConnection, EGeneType::Connection, InputNode->ID, HiddenNode->ID); // Get the new connection ID
			if (Connections.Contains(ConnectionID)) continue; // Connection already exists
			Connections[ConnectionID] = ConnectionGene(ConnectionID, InputNode->ID, HiddenNode->ID, 1.0); // Create a new connection
		}
		for (int Jdx = 0, StopJdx = Config->NumOutputs; Jdx != StopJdx; ++Jdx)
		{
			if (GetRandomDouble(0.0, 1.0) >= Config->InitialConnectionProbability) continue; // Connect each input node to each output node with a probability of InitialConnectionProbability
			auto OutputNode = Genome->GetNodeByID(Jdx + Config->NumInputs + 1); // Get the output node
			auto ConnectionID = Innovations.GetInnovationID(EMutationType::AddConnection, EGeneType::Connection, InputNode->ID, OutputNode->ID); // Get the new connection ID
			if (Connections.Contains(ConnectionID)) continue; // Connection already exists
			Connections[ConnectionID] = ConnectionGene(ConnectionID, InputNode->ID, OutputNode->ID, 1.0); // Create a new connection
		}
	}
	// Connect each hidden node to each output node 
	for (int Idx = 0, StopIdx = Config->NumHidden; Idx != StopIdx; ++Idx)
	{
		auto HiddenNode = Genome->GetNodeByID(Idx + Config->NumInputs + Config->NumOutputs + 1); // Get the hidden node, +1 for the bias node
		for (int Jdx = 0, StopJdx = Config->NumOutputs; Jdx != StopJdx; ++Jdx)
		{
			if (GetRandomDouble(0.0, 1.0) >= Config->InitialConnectionProbability) continue; // Connect each input node to each output node with a probability of InitialConnectionProbability
			auto OutputNode = Genome->GetNodeByID(Jdx + Config->NumInputs + 1); // Get the output node, +1 for the bias node
			auto ConnectionID = Innovations.GetInnovationID(EMutationType::AddConnection, EGeneType::Connection, HiddenNode->ID, OutputNode->ID); // Get the new connection ID
			if (Connections.Contains(ConnectionID)) continue; // Connection already exists
			Connections[ConnectionID] = ConnectionGene(ConnectionID, HiddenNode->ID, OutputNode->ID, 1.0); // Create a new connection
		}
	}
}

void Full(GenomePtr Genome) // Initialize connections from every Input to every Output, from every Input to every Hidden, and from every Hidden to every Output
{
	None(Genome); // Start with an unconnected network
	auto& Connections = Genome->Genotype.Connections;
	auto& Nodes = Genome->Genotype.Nodes;
	const auto Config = Genome->Config;
	// Connect each input node to each hidden node 
	for (int Idx = 0, StopIdx = (Config->NumInputs + 1); Idx != StopIdx; ++Idx) // +1 for the bias node
	{
		auto InputNode = Genome->GetNodeByID(Idx); // Get the input node
		for (int Jdx = 0, StopJdx = Config->NumHidden; Jdx != StopJdx; ++Jdx)
		{
			auto HiddenNode = Genome->GetNodeByID(Jdx + Config->NumInputs + Config->NumOutputs + 1); // Get the hidden node
			auto ConnectionID = Innovations.GetInnovationID(EMutationType::AddConnection, EGeneType::Connection, InputNode->ID, HiddenNode->ID); // Get the new connection ID
			if (Connections.Contains(ConnectionID)) continue; // Connection already exists
			Connections[ConnectionID] = ConnectionGene(ConnectionID, InputNode->ID, HiddenNode->ID, 1.0); // Create a new connection
		}
		for (int Jdx = 0, StopJdx = Config->NumOutputs; Jdx != StopJdx; ++Jdx)
		{
			auto OutputNode = Genome->GetNodeByID(Jdx + Config->NumInputs + 1); // Get the output node
			auto ConnectionID = Innovations.GetInnovationID(EMutationType::AddConnection, EGeneType::Connection, InputNode->ID, OutputNode->ID); // Get the new connection ID
			if (Connections.Contains(ConnectionID)) continue; // Connection already exists
			Connections[ConnectionID] = ConnectionGene(ConnectionID, InputNode->ID, OutputNode->ID, 1.0); // Create a new connection
		}
	}
	// Connect each hidden node to each output node 
	for (int Idx = 0, StopIdx = Config->NumHidden; Idx != StopIdx; ++Idx)
	{
		auto HiddenNode = Genome->GetNodeByID(Idx + Config->NumInputs + Config->NumOutputs + 1); // Get the hidden node, +1 for the bias node
		for (int Jdx = 0, StopJdx = Config->NumOutputs; Jdx != StopJdx; ++Jdx)
		{
			auto OutputNode = Genome->GetNodeByID(Jdx + Config->NumInputs + 1); // Get the output node, +1 for the bias node
			auto ConnectionID = Innovations.GetInnovationID(EMutationType::AddConnection, EGeneType::Connection, HiddenNode->ID, OutputNode->ID); // Get the new connection ID
			if (Connections.Contains(ConnectionID)) continue; // Connection already exists
			Connections[ConnectionID] = ConnectionGene(ConnectionID, HiddenNode->ID, OutputNode->ID, 1.0); // Create a new connection
		}
	}
}

void Tree(GenomePtr Genome) // Initialize connections for a tree-like neural network
{
	None(Genome); // Start with an unconnected network
	auto& Connections = Genome->Genotype.Connections;
	auto& Nodes = Genome->Genotype.Nodes;
	const auto Config = Genome->Config;
	// Connect each input node to each hidden node 
	for (int Idx = 0, StopIdx = (Config->NumInputs + 1); Idx != StopIdx; ++Idx) // +1 for the bias node
	{
		auto InputNode = Genome->GetNodeByID(Idx); // Get the input node
		for (int Jdx = 0, StopJdx = Config->NumHidden; Jdx != StopJdx; ++Jdx)
		{
			auto HiddenNode = Genome->GetNodeByID(Jdx + Config->NumInputs + Config->NumOutputs + 1); // Get the hidden node
			auto ConnectionID = Innovations.GetInnovationID(EMutationType::AddConnection, EGeneType::Connection, InputNode->ID, HiddenNode->ID); // Get the new connection ID
			if (Connections.Contains(ConnectionID)) continue; // Connection already exists
			Connections[ConnectionID] = ConnectionGene(ConnectionID, InputNode->ID, HiddenNode->ID, 1.0); // Create a new connection
		}
	}
	// Connect each hidden node to each output node 
	for (int Idx = 0, StopIdx = Config->NumHidden; Idx != StopIdx; ++Idx)
	{
		for (int Jdx = 0, StopJdx = Config->NumOutputs; Jdx != StopJdx; ++Jdx)
		{
			auto HiddenNode = Genome->GetNodeByID(Idx + Config->NumInputs + Config->NumOutputs + 1); // Get the hidden node, +1 for the bias node
			auto OutputNode = Genome->GetNodeByID(Jdx + Config->NumInputs + 1); // Get the output node, +1 for the bias node
			auto ConnectionID = Innovations.GetInnovationID(EMutationType::AddConnection, EGeneType::Connection, HiddenNode->ID, OutputNode->ID); // Get the new connection ID
			if (Connections.Contains(ConnectionID)) continue; // Connection already exists
			Connections[ConnectionID] = ConnectionGene(ConnectionID, HiddenNode->ID, OutputNode->ID, 1.0); // Create a new connection
		}
	}
}

} // namespace InitialTopology

namespace CrossoverType {
	GenomePtr Uniform(const GenomePtr& Parent1, const GenomePtr& Parent2) // Initialize a genome from two parents using uniform crossover
	{
		const auto Config = Parent1->Config;
		GenomePtr Genome = std::make_shared<NEAT::Genome>(Config);
		Genome->ID = NEAT::Genome::GenerateNewGenomeID();

		// Iterate over the node genes of both parents  
		const auto& NodeKeys1 = Parent1->Genotype.Nodes.GetKeys();
		const auto& NodeKeys2 = Parent2->Genotype.Nodes.GetKeys();
		TArray<uint64> CombinedNodeKeys;
		for (const auto& Key : NodeKeys1) CombinedNodeKeys.AddUnique(Key);
		for (const auto& Key : NodeKeys2) CombinedNodeKeys.AddUnique(Key);

		for (int Idx = 0, StopIdx = CombinedNodeKeys.Num(); Idx != StopIdx; ++Idx)
		{
			uint64 NodeKey = CombinedNodeKeys[Idx];
			if (Parent1->Genotype.Nodes.Contains(NodeKey) && Parent2->Genotype.Nodes.Contains(NodeKey))
			{
				if (GetRandomDouble(0.0, 1.0) < 0.5)
				{
					Genome->Genotype.Nodes[NodeKey] = Parent1->Genotype.Nodes[NodeKey];
				}
				else
				{
					Genome->Genotype.Nodes[NodeKey] = Parent2->Genotype.Nodes[NodeKey];
				}
			}
			else if (Parent1->Genotype.Nodes.Contains(NodeKey))
			{
				Genome->Genotype.Nodes[NodeKey] = Parent1->Genotype.Nodes[NodeKey];
			}
			else if (Parent2->Genotype.Nodes.Contains(NodeKey))
			{
				Genome->Genotype.Nodes[NodeKey] = Parent2->Genotype.Nodes[NodeKey];
			}
		}

		// Iterate over the connection genes of both parents
		const auto& ConnectionKeys1 = Parent1->Genotype.Connections.GetKeys();
		const auto& ConnectionKeys2 = Parent2->Genotype.Connections.GetKeys();
		TArray<uint64> CombinedConnectionKeys;
		for (const auto& Key : ConnectionKeys1) CombinedConnectionKeys.AddUnique(Key);
		for (const auto& Key : ConnectionKeys2) CombinedConnectionKeys.AddUnique(Key);

		for (int Idx = 0, StopIdx = CombinedConnectionKeys.Num(); Idx != StopIdx; ++Idx)
		{
			uint64 ConnectionKey = CombinedConnectionKeys[Idx];
			if (Parent1->Genotype.Connections.Contains(ConnectionKey) && Parent2->Genotype.Connections.Contains(ConnectionKey))
			{
				if (GetRandomDouble(0.0, 1.0) < 0.5)
				{
					Genome->Genotype.Connections[ConnectionKey] = Parent1->Genotype.Connections[ConnectionKey];
				}
				else
				{
					Genome->Genotype.Connections[ConnectionKey] = Parent2->Genotype.Connections[ConnectionKey];
				}
			}
			else if (Parent1->Genotype.Connections.Contains(ConnectionKey))
			{
				Genome->Genotype.Connections[ConnectionKey] = Parent1->Genotype.Connections[ConnectionKey];
			}
			else if (Parent2->Genotype.Connections.Contains(ConnectionKey))
			{
				Genome->Genotype.Connections[ConnectionKey] = Parent2->Genotype.Connections[ConnectionKey];
			}
		}

		return std::move(Genome);
	}

	GenomePtr Multipoint(const GenomePtr& Parent1, const GenomePtr& Parent2) // Initialize a genome from two parents using multipoint crossover with settings from Config->CrossoverPoints  
	{
		const auto Config = Parent1->Config;
		GenomePtr Genome = std::make_shared<NEAT::Genome>(Config);
		Genome->ID = NEAT::Genome::GenerateNewGenomeID();

		int NumCrossoverPoints = Config->CrossoverPoints;
		TArray<int> CrossoverPoints;
		for (int Idx = 0; Idx != NumCrossoverPoints; ++Idx)
		{
			CrossoverPoints.Add(GetRandomInt(0, std::min(Parent1->GetNumNodes(), Parent2->GetNumNodes()) - 1));
		}
		CrossoverPoints.Sort();

		// Iterate over the node genes of both parents
		const auto& NodeKeys1 = Parent1->Genotype.Nodes.GetKeys();
		const auto& NodeKeys2 = Parent2->Genotype.Nodes.GetKeys();
		TArray<uint64> CombinedNodeKeys;
		for (const auto& Key : NodeKeys1) CombinedNodeKeys.AddUnique(Key);
		for (const auto& Key : NodeKeys2) CombinedNodeKeys.AddUnique(Key);

		for (int Idx = 0, StopIdx = CombinedNodeKeys.Num(); Idx != StopIdx; ++Idx)
		{
			uint64 NodeKey = CombinedNodeKeys[Idx];
			if (Parent1->Genotype.Nodes.Contains(NodeKey) && Parent2->Genotype.Nodes.Contains(NodeKey))
			{
				bool bUseParent1 = true;
				for (int Jdx = 0, StopJdx = CrossoverPoints.Num(); Jdx != StopJdx; ++Jdx)
				{
					if (NodeKey == CrossoverPoints[Jdx])
					{
						bUseParent1 = !bUseParent1;
						break;
					}
				}
				if (bUseParent1)
				{
					Genome->Genotype.Nodes[NodeKey] = Parent1->Genotype.Nodes[NodeKey];
				}
				else
				{
					Genome->Genotype.Nodes[NodeKey] = Parent2->Genotype.Nodes[NodeKey];
				}
			}
			else if (Parent1->Genotype.Nodes.Contains(NodeKey))
			{
				Genome->Genotype.Nodes[NodeKey] = Parent1->Genotype.Nodes[NodeKey];
			}
			else if (Parent2->Genotype.Nodes.Contains(NodeKey))
			{
				Genome->Genotype.Nodes[NodeKey] = Parent2->Genotype.Nodes[NodeKey];
			}
		}

		// Iterate over the connection genes of both parents
		const auto& ConnectionKeys1 = Parent1->Genotype.Connections.GetKeys();
		const auto& ConnectionKeys2 = Parent2->Genotype.Connections.GetKeys();
		TArray<uint64> CombinedConnectionKeys;
		for (const auto& Key : ConnectionKeys1) CombinedConnectionKeys.AddUnique(Key);
		for (const auto& Key : ConnectionKeys2) CombinedConnectionKeys.AddUnique(Key);

		// Iterate over the connection genes of both parents, copying genes from the first parent up to the crossover point, and from the second parent after the crossover point
		// If a gene is present in both parents, a random gene is selected from either parent with equal probability
		for (int Idx = 0, StopIdx = CombinedConnectionKeys.Num(); Idx != StopIdx; ++Idx)
		{
			uint64 ConnectionKey = CombinedConnectionKeys[Idx];
			if (Parent1->Genotype.Connections.Contains(ConnectionKey) && Parent2->Genotype.Connections.Contains(ConnectionKey)) // If present in both parents
			{
				bool bUseParent1 = true;
				for (int Jdx = 0, StopJdx = CrossoverPoints.Num(); Jdx != StopJdx; ++Jdx)
				{
					if (ConnectionKey == CrossoverPoints[Jdx])
					{
						bUseParent1 = !bUseParent1;
						break;
					}
				}

				if (bUseParent1) Genome->Genotype.Connections[ConnectionKey] = Parent1->Genotype.Connections[ConnectionKey]; // Copy from Parent1
				else Genome->Genotype.Connections[ConnectionKey] = Parent2->Genotype.Connections[ConnectionKey]; // Copy from Parent2
			}
			else if (Parent1->Genotype.Connections.Contains(ConnectionKey))
			{
				Genome->Genotype.Connections[ConnectionKey] = Parent1->Genotype.Connections[ConnectionKey];
			}
			else if (Parent2->Genotype.Connections.Contains(ConnectionKey))
			{
				Genome->Genotype.Connections[ConnectionKey] = Parent2->Genotype.Connections[ConnectionKey];
			}
		}

		return std::move(Genome);
	}

	GenomePtr SinglePoint(const GenomePtr& Parent1, const GenomePtr& Parent2) // Initialize a genome from two parents using single-point crossover
	{
		const auto Config = Parent1->Config;
		GenomePtr Genome = std::make_shared<NEAT::Genome>(Config);
		Genome->ID = NEAT::Genome::GenerateNewGenomeID();

		int CrossoverPoint = GetRandomInt(0, std::min(Parent1->GetNumNodes(), Parent2->GetNumNodes()) - 1);

		// Iterate over the node genes of both parents
		const auto& NodeKeys1 = Parent1->Genotype.Nodes.GetKeys();
		const auto& NodeKeys2 = Parent2->Genotype.Nodes.GetKeys();
		TArray<uint64> CombinedNodeKeys;
		for (const auto& Key : NodeKeys1) CombinedNodeKeys.AddUnique(Key);
		for (const auto& Key : NodeKeys2) CombinedNodeKeys.AddUnique(Key);

		// Iterate over the node genes of both parents, copying genes from the first parent up to the crossover point, and from the second parent after the crossover point
		// If a gene is present in both parents, a random gene is selected from either parent with equal probability
		for (int Idx = 0, StopIdx = CombinedNodeKeys.Num(); Idx != StopIdx; ++Idx)
		{
			uint64 NodeKey = CombinedNodeKeys[Idx];
			if (Parent1->Genotype.Nodes.Contains(NodeKey) && Parent2->Genotype.Nodes.Contains(NodeKey))
			{
				if (NodeKey < CrossoverPoint)
				{
					Genome->Genotype.Nodes[NodeKey] = Parent1->Genotype.Nodes[NodeKey];
				}
				else
				{
					Genome->Genotype.Nodes[NodeKey] = Parent2->Genotype.Nodes[NodeKey];
				}
			}
			else if (Parent1->Genotype.Nodes.Contains(NodeKey))
			{
				Genome->Genotype.Nodes[NodeKey] = Parent1->Genotype.Nodes[NodeKey];
			}
			else if (Parent2->Genotype.Nodes.Contains(NodeKey))
			{
				Genome->Genotype.Nodes[NodeKey] = Parent2->Genotype.Nodes[NodeKey];
			}
		}

		// Iterate over the connection genes of both parents
		const auto& ConnectionKeys1 = Parent1->Genotype.Connections.GetKeys();
		const auto& ConnectionKeys2 = Parent2->Genotype.Connections.GetKeys();
		TArray<uint64> CombinedConnectionKeys;
		for (const auto& Key : ConnectionKeys1) CombinedConnectionKeys.AddUnique(Key);
		for (const auto& Key : ConnectionKeys2) CombinedConnectionKeys.AddUnique(Key);

		// Iterate over the connection genes of both parents, copying genes from the first parent up to the crossover point, and from the second parent after the crossover point
		// If a gene is present in both parents, a random gene is selected from either parent with equal probability
		for (int Idx = 0, StopIdx = CombinedConnectionKeys.Num(); Idx != StopIdx; ++Idx)
		{
			uint64 ConnectionKey = CombinedConnectionKeys[Idx];
			if (Parent1->Genotype.Connections.Contains(ConnectionKey) && Parent2->Genotype.Connections.Contains(ConnectionKey))
			{
				if (ConnectionKey < CrossoverPoint)
				{
					Genome->Genotype.Connections[ConnectionKey] = Parent1->Genotype.Connections[ConnectionKey];
				}
				else
				{
					Genome->Genotype.Connections[ConnectionKey] = Parent2->Genotype.Connections[ConnectionKey];
				}
			}
			else if (Parent1->Genotype.Connections.Contains(ConnectionKey))
			{
				Genome->Genotype.Connections[ConnectionKey] = Parent1->Genotype.Connections[ConnectionKey];
			}
			else if (Parent2->Genotype.Connections.Contains(ConnectionKey))
			{
				Genome->Genotype.Connections[ConnectionKey] = Parent2->Genotype.Connections[ConnectionKey];
			}
		}


		return std::move(Genome);
	}

	GenomePtr TwoPoint(const GenomePtr& Parent1, const GenomePtr& Parent2) // Initialize a genome from two parents using two-point crossover  
	{
		const auto Config = Parent1->Config;
		GenomePtr Genome = std::make_shared<NEAT::Genome>(Config);
		Genome->ID = NEAT::Genome::GenerateNewGenomeID();

		int CrossoverPoint1 = GetRandomInt(0, std::min(Parent1->GetNumNodes(), Parent2->GetNumNodes()) - 1);
		int CrossoverPoint2 = GetRandomInt(0, std::min(Parent1->GetNumNodes(), Parent2->GetNumNodes()) - 1);

		// Iterate over the node genes of both parents
		const auto& NodeKeys1 = Parent1->Genotype.Nodes.GetKeys();
		const auto& NodeKeys2 = Parent2->Genotype.Nodes.GetKeys();
		TArray<uint64> CombinedNodeKeys;
		for (const auto& Key : NodeKeys1) CombinedNodeKeys.AddUnique(Key);
		for (const auto& Key : NodeKeys2) CombinedNodeKeys.AddUnique(Key);

		// Iterate over the node genes of both parents, copying genes from the first parent up to the first crossover point, 
		// from the second parent between the two crossover points, and from the first parent after the second crossover point
		// If a gene is present in both parents, a random gene is selected from either parent with equal probability
		for (int Idx = 0, StopIdx = CombinedNodeKeys.Num(); Idx != StopIdx; ++Idx)
		{
			uint64 NodeKey = CombinedNodeKeys[Idx];
			if (Parent1->Genotype.Nodes.Contains(NodeKey) && Parent2->Genotype.Nodes.Contains(NodeKey))
			{
				if (NodeKey < CrossoverPoint1)
				{
					Genome->Genotype.Nodes[NodeKey] = Parent1->Genotype.Nodes[NodeKey];
				}
				else if (NodeKey < CrossoverPoint2)
				{
					Genome->Genotype.Nodes[NodeKey] = Parent2->Genotype.Nodes[NodeKey];
				}
				else
				{
					Genome->Genotype.Nodes[NodeKey] = Parent1->Genotype.Nodes[NodeKey];
				}
			}
			else if (Parent1->Genotype.Nodes.Contains(NodeKey))
			{
				Genome->Genotype.Nodes[NodeKey] = Parent1->Genotype.Nodes[NodeKey];
			}
			else if (Parent2->Genotype.Nodes.Contains(NodeKey))
			{
				Genome->Genotype.Nodes[NodeKey] = Parent2->Genotype.Nodes[NodeKey];
			}
		}

		// Iterate over the connection genes of both parents
		const auto& ConnectionKeys1 = Parent1->Genotype.Connections.GetKeys();
		const auto& ConnectionKeys2 = Parent2->Genotype.Connections.GetKeys();
		TArray<uint64> CombinedConnectionKeys;
		for (const auto& Key : ConnectionKeys1) CombinedConnectionKeys.AddUnique(Key);
		for (const auto& Key : ConnectionKeys2) CombinedConnectionKeys.AddUnique(Key);

		// Iterate over the connection genes of both parents, copying genes from the first parent up to the first crossover point, 
		// from the second parent between the two crossover points, and from the first parent after the second crossover point
		// If a gene is present in both parents, a random gene is selected from either parent with equal probability
		for (int Idx = 0, StopIdx = CombinedConnectionKeys.Num(); Idx != StopIdx; ++Idx)
		{
			uint64 ConnectionKey = CombinedConnectionKeys[Idx];
			if (Parent1->Genotype.Connections.Contains(ConnectionKey) && Parent2->Genotype.Connections.Contains(ConnectionKey))
			{
				if (ConnectionKey < CrossoverPoint1)
				{
					Genome->Genotype.Connections[ConnectionKey] = Parent1->Genotype.Connections[ConnectionKey];
				}
				else if (ConnectionKey < CrossoverPoint2)
				{
					Genome->Genotype.Connections[ConnectionKey] = Parent2->Genotype.Connections[ConnectionKey];
				}
				else
				{
					Genome->Genotype.Connections[ConnectionKey] = Parent1->Genotype.Connections[ConnectionKey];
				}
			}
			else if (Parent1->Genotype.Connections.Contains(ConnectionKey))
			{
				Genome->Genotype.Connections[ConnectionKey] = Parent1->Genotype.Connections[ConnectionKey];
			}
			else if (Parent2->Genotype.Connections.Contains(ConnectionKey))
			{
				Genome->Genotype.Connections[ConnectionKey] = Parent2->Genotype.Connections[ConnectionKey];
			}
		}

		return std::move(Genome);
	}
} // namespace CrossoverType

namespace GenomePairing {
GenomePtr Offspring::GetChild() const
{
	if (Parent1 && Parent2) return InitialTopology::InitializeFromParents(Parent1, Parent2);
	else if (Parent1) return InitialTopology::InitializeFromParent(Parent1);
	else return InitialTopology::InitializeGenome(Config);
}

TArray<Offspring> Random(const TArray<GenomePtr>& Population, int ReproductionCount, const ConfigPtr& Config)
{
	TArray<Offspring> OffspringList;
	int NumOffspring = int(Config->PopulationSize - Population.Num());
	for (int Idx = 0; Idx != ReproductionCount; ++Idx)
	{
		bool bCrossover = (Config->CrossoverRate > 0.0 && (double)rand() / RAND_MAX < Config->CrossoverRate);
		if (bCrossover) OffspringList.Add(Offspring(Config, Population[rand() % Population.Num()], Population[rand() % Population.Num()]));
		else if ((double)rand() / RAND_MAX > 0.5) OffspringList.Add(Offspring(Config, Population[rand() % Population.Num()])); // 50/50 chance of asexual reproduction with mutation 
		else OffspringList.Add(Offspring(Config)); // 50% chance of random initialization
	}

	return OffspringList;
}

TArray<Offspring> Fittest(const TArray<GenomePtr>& Population, int ReproductionCount, const ConfigPtr& Config)
{
	GenomePtr FittestGenome = nullptr;
	for (const auto& Genome : Population)
	{
		if (!FittestGenome || Genome->Fitness > FittestGenome->Fitness)
		{
			FittestGenome = Genome;
		}
	}

	TArray<Offspring> OffspringList;
	for (int Idx = 0; Idx != ReproductionCount; ++Idx)
	{
		bool bCrossover = (Config->CrossoverRate > 0.0 && (double)rand() / RAND_MAX < Config->CrossoverRate);
		if (bCrossover) OffspringList.Add(Offspring(Config, FittestGenome, Population[rand() % Population.Num()]));
		else if ((double)rand() / RAND_MAX > 0.5) OffspringList.Add(Offspring(Config, FittestGenome)); // 50/50 chance of asexual reproduction with mutation 
		else OffspringList.Add(Offspring(Config)); // 50% chance of random initialization
	}

	return OffspringList;
}

TArray<Offspring> Weakest(const TArray<GenomePtr>& Population, int ReproductionCount, const ConfigPtr& Config)
{
	GenomePtr WeakestGenome = nullptr;
	for (const auto& Genome : Population)
	{
		if (!WeakestGenome || Genome->Fitness < WeakestGenome->Fitness)
		{
			WeakestGenome = Genome;
		}
	}

	TArray<Offspring> OffspringList;
	for (int Idx = 0; Idx != ReproductionCount; ++Idx)
	{
		bool bCrossover = (Config->CrossoverRate > 0.0 && (double)rand() / RAND_MAX < Config->CrossoverRate);
		if (bCrossover) OffspringList.Add(Offspring(Config, WeakestGenome, Population[rand() % Population.Num()]));
		else if ((double)rand() / RAND_MAX > 0.5) OffspringList.Add(Offspring(Config, WeakestGenome)); // 50/50 chance of asexual reproduction with mutation 
		else OffspringList.Add(Offspring(Config)); // 50% chance of random initialization
	}

	return OffspringList;

}

TArray<Offspring> Alternating(const TArray<GenomePtr>& Population, int ReproductionCount, const ConfigPtr& Config)
{
	GenomePtr FittestGenome = nullptr;
	GenomePtr WeakestGenome = nullptr;
	for (const auto& Genome : Population)
	{
		if (!FittestGenome || Genome->Fitness > FittestGenome->Fitness)
		{
			FittestGenome = Genome;
		}
		if (!WeakestGenome || Genome->Fitness < WeakestGenome->Fitness)
		{
			WeakestGenome = Genome;
		}
	}

	TArray<Offspring> OffspringList;
	for (int Idx = 0; Idx != ReproductionCount; ++Idx)
	{
		bool bCrossover = (Config->CrossoverRate > 0.0 && (double)rand() / RAND_MAX < Config->CrossoverRate);
		if (Idx % 2 == 0) // Even index  
		{
			if (bCrossover) OffspringList.Add(Offspring(Config, FittestGenome, Population[rand() % Population.Num()]));
			else if ((double)rand() / RAND_MAX > 0.5) OffspringList.Add(Offspring(Config, FittestGenome)); // 50/50 chance of asexual reproduction with mutation 
			else OffspringList.Add(Offspring(Config)); // 50% chance of random initialization
		}
		else // Odd index  
		{
			if (bCrossover) OffspringList.Add(Offspring(Config, WeakestGenome, Population[rand() % Population.Num()]));
			else if ((double)rand() / RAND_MAX > 0.5) OffspringList.Add(Offspring(Config, WeakestGenome)); // 50/50 chance of asexual reproduction with mutation 
			else OffspringList.Add(Offspring(Config)); // 50% chance of random initialization
		}
	}

	return OffspringList;
}

TArray<Offspring> SimilarFitness(const TArray<GenomePtr>& Population, int ReproductionCount, const ConfigPtr& Config)
{
	TArray<Offspring> OffspringList;
	for (int Idx = 0; Idx != ReproductionCount; ++Idx)
	{
		int Parent1Idx = rand() % Population.Num(); // Select Parent1 at random  
		GenomePtr Parent1 = Population[Parent1Idx];

		int Parent2Idx = -1; // Select Parent2 from a nearby neighbor (if crossover)
		if (Config->CrossoverRate > 0.0 && (double)rand() / RAND_MAX < Config->CrossoverRate)
		{
			// Find a nearby neighbor with similar fitness  
			double MinDistance = std::numeric_limits<double>::max();
			for (int NeighborIdx = 0; NeighborIdx < Population.Num(); ++NeighborIdx)
			{
				if (NeighborIdx == Parent1Idx) continue;
				double Distance = std::abs(Parent1->Fitness - Population[NeighborIdx]->Fitness);
				if (Distance < MinDistance)
				{
					MinDistance = Distance;
					Parent2Idx = NeighborIdx;
				}
			}
		}

		if (Parent2Idx != INDEX_NONE) OffspringList.Add(Offspring(Config, Parent1, Population[Parent2Idx])); // Crossover 
		else if ((double)rand() / RAND_MAX > 0.5) OffspringList.Add(Offspring(Config, Parent1)); // 50/50 chance of asexual reproduction with mutation 
		else OffspringList.Add(Offspring(Config)); // 50% chance of random initialization
	}

	return OffspringList;
}

TArray<Offspring> DissimilarFitness(const TArray<GenomePtr>& Population, int ReproductionCount, const ConfigPtr& Config)
{
	TArray<Offspring> OffspringList;
	for (int Idx = 0; Idx != ReproductionCount; ++Idx)
	{
		int Parent1Idx = rand() % Population.Num(); // Select Parent1 at random  
		GenomePtr Parent1 = Population[Parent1Idx];

		int Parent2Idx = -1; // Select Parent2 from a distant neighbor (if crossover)
		if (Config->CrossoverRate > 0.0 && (double)rand() / RAND_MAX < Config->CrossoverRate)
		{
			// Find a distant neighbor with dissimilar fitness  
			double MaxDistance = 0.0;
			for (int NeighborIdx = 0; NeighborIdx < Population.Num(); ++NeighborIdx)
			{
				if (NeighborIdx == Parent1Idx) continue;
				double Distance = std::abs(Parent1->Fitness - Population[NeighborIdx]->Fitness);
				if (Distance > MaxDistance)
				{
					MaxDistance = Distance;
					Parent2Idx = NeighborIdx;
				}
			}
		}

		if (Parent2Idx != INDEX_NONE) OffspringList.Add(Offspring(Config, Parent1, Population[Parent2Idx])); // Crossover 
		else if ((double)rand() / RAND_MAX > 0.5) OffspringList.Add(Offspring(Config, Parent1)); // 50/50 chance of asexual reproduction with mutation 
		else OffspringList.Add(Offspring(Config)); // 50% chance of random initialization
	}

	return OffspringList;
}

// This should use a distance metric to select parents with similar genomes
TArray<Offspring> Proximity(const TArray<GenomePtr>& Population, int ReproductionCount, const ConfigPtr& Config)
{
	TArray<Offspring> OffspringList;
	for (int Idx = 0; Idx != ReproductionCount; ++Idx)
	{
		int Parent1Idx = rand() % Population.Num(); // Select Parent1 at random  
		GenomePtr Parent1 = Population[Parent1Idx];

		int Parent2Idx = -1; // Select Parent2 from a nearby neighbor (if crossover)
		if (Config->CrossoverRate > 0.0 && (double)rand() / RAND_MAX < Config->CrossoverRate)
		{
			// Find a nearby neighbor with similar genome  
			double MinDistance = std::numeric_limits<double>::max();
			for (int NeighborIdx = 0; NeighborIdx < Population.Num(); ++NeighborIdx)
			{
				if (NeighborIdx == Parent1Idx) continue;
				double Distance = Distance::Calculate(Parent1, Population[NeighborIdx], Config);
				if (Distance < MinDistance)
				{
					MinDistance = Distance;
					Parent2Idx = NeighborIdx;
				}
			}
		}

		if (Parent2Idx != INDEX_NONE) OffspringList.Add(Offspring(Config, Parent1, Population[Parent2Idx])); // Crossover 
		else if ((double)rand() / RAND_MAX > 0.5) OffspringList.Add(Offspring(Config, Parent1)); // 50/50 chance of asexual reproduction with mutation 
		else OffspringList.Add(Offspring(Config)); // 50% chance of random initialization
	}

	return OffspringList;
}

TArray<Offspring> Diversity(const TArray<GenomePtr>& Population, int ReproductionCount, const ConfigPtr& Config)
{
	TArray<Offspring> OffspringList;
	for (int Idx = 0; Idx != ReproductionCount; ++Idx)
	{
		int Parent1Idx = rand() % Population.Num(); // Select Parent1 at random  
		GenomePtr Parent1 = Population[Parent1Idx];

		int Parent2Idx = -1; // Select Parent2 from a distant neighbor (if crossover)
		if (Config->CrossoverRate > 0.0 && (double)rand() / RAND_MAX < Config->CrossoverRate)
		{
			// Find a distant neighbor with dissimilar genome  
			double MaxDistance = 0.0;
			for (int NeighborIdx = 0; NeighborIdx < Population.Num(); ++NeighborIdx)
			{
				if (NeighborIdx == Parent1Idx) continue;
				double Distance = Distance::Calculate(Parent1, Population[NeighborIdx], Config);
				if (Distance > MaxDistance)
				{
					MaxDistance = Distance;
					Parent2Idx = NeighborIdx;
				}
			}
		}

		if (Parent2Idx != INDEX_NONE) OffspringList.Add(Offspring(Config, Parent1, Population[Parent2Idx])); // Crossover 
		else if ((double)rand() / RAND_MAX > 0.5) OffspringList.Add(Offspring(Config, Parent1)); // 50/50 chance of asexual reproduction with mutation 
		else OffspringList.Add(Offspring(Config)); // 50% chance of random initialization
	}

	return OffspringList;
}

TArray<Offspring> Reproduce(const TArray<GenomePtr>& Population, int ReproductionCount, const ConfigPtr& Config)
{
	if (Config->PairingMethod == EGenomePairing::Random) return Random(Population, ReproductionCount, Config);
	if (Config->PairingMethod == EGenomePairing::Fittest) return Fittest(Population, ReproductionCount, Config);
	if (Config->PairingMethod == EGenomePairing::Weakest) return Weakest(Population, ReproductionCount, Config);
	if (Config->PairingMethod == EGenomePairing::Alternating) return Alternating(Population, ReproductionCount, Config);
	if (Config->PairingMethod == EGenomePairing::SimilarFitness) return SimilarFitness(Population, ReproductionCount, Config);
	if (Config->PairingMethod == EGenomePairing::DissimilarFitness) return DissimilarFitness(Population, ReproductionCount, Config);
	if (Config->PairingMethod == EGenomePairing::Proximity) return Proximity(Population, ReproductionCount, Config);
	if (Config->PairingMethod == EGenomePairing::Diversity) return Diversity(Population, ReproductionCount, Config);
	return Random(Population, ReproductionCount, Config);
}

} // namespace GenomePairing
namespace Distance {

double Euclidean(const GenomePtr& Genome1, const GenomePtr& Genome2, const ConfigPtr& Config)
{
	// Build a list of connections that only exist for one genome or the other and apply the excess gene coefficient for each
	// Then take the remaining list of shared connections and apply the difference in weights for each.

	bool bDebugDistance = false;

	TArray<uint64> MatchingConnectionKeys;
	TArray<uint64> DisjointConnectionKeys;
	for (const uint64& ConnectionID1 : Genome1->Genotype.GetConnectionKeys())
	{
		if (Genome2->GetConnectionByID(ConnectionID1)) MatchingConnectionKeys.AddUnique(ConnectionID1);
		else DisjointConnectionKeys.Add(ConnectionID1);
	}

	for (const uint64& ConnectionID2 : Genome2->Genotype.GetConnectionKeys())
	{
		if (!Genome1->GetConnectionByID(ConnectionID2)) DisjointConnectionKeys.Add(ConnectionID2);
	}

	if (bDebugDistance)
	{
		LogMessage(LogLevel::Debug, "Matching Connection Keys:");
		for (const uint64& ConnectionID : MatchingConnectionKeys)
		{
			LogMessage(LogLevel::Debug, "ConnectionID: %llu", ConnectionID);
		}

		LogMessage(LogLevel::Debug, "Disjoint Connection Keys:");
		for (const uint64& ConnectionID : DisjointConnectionKeys)
		{
			LogMessage(LogLevel::Debug, "ConnectionID: %llu", ConnectionID);
		}
	}

	double ConnectionDistance = 0.0;
	for (const uint64& ConnectionID : MatchingConnectionKeys)
	{
		const auto Connection1 = Genome1->GetConnectionByID(ConnectionID);
		const auto Connection2 = Genome2->GetConnectionByID(ConnectionID);
		ConnectionDistance += Config->MatchingGeneCoefficient * std::pow(std::abs(Connection1->Weight - Connection2->Weight), Config->DistanceExponent);
	}

	TArray<uint64> MatchingNodeKeys;
	TArray<uint64> DisjointNodeKeys;
	TArray<uint64> ExcessNodeKeys;

	for (const uint64& NodeID1 : Genome1->Genotype.GetNodeKeys())
	{
		auto* ParentNode1 = Genome1->GetNodeByID(NodeID1);
		if (!ParentNode1) { BREAKPOINT(); continue; }
		if (auto* ParentNode2 = Genome2->GetNodeByID(NodeID1))
		{
			bool bMatchingNode = (ParentNode1->Activation == ParentNode2->Activation && ParentNode1->Aggregation == ParentNode2->Aggregation);
			if (bMatchingNode) MatchingNodeKeys.AddUnique(NodeID1);
			else DisjointNodeKeys.Add(NodeID1);
		}
		else
		{
			ExcessNodeKeys.Add(NodeID1);
		}
	}

	for (const uint64& NodeID2 : Genome2->Genotype.GetNodeKeys())
	{
		if (!Genome1->GetNodeByID(NodeID2)) ExcessNodeKeys.Add(NodeID2);
	}

	int NumNodes1 = Genome1->Genotype.Nodes.Num();
	int NumConnections1 = Genome1->Genotype.Connections.Num();
	int NumNodes2 = Genome2->Genotype.Nodes.Num();
	int NumConnections2 = Genome2->Genotype.Connections.Num();

	int MaxGenomeSize = Math::Max(NumNodes1 + NumConnections1, NumNodes2 + NumConnections2);

	double MatchingNodeDistance = 0.0;
	for (const uint64& NodeID : MatchingNodeKeys)
	{
		const auto Node1 = Genome1->GetNodeByID(NodeID);
		const auto Node2 = Genome2->GetNodeByID(NodeID);
		MatchingNodeDistance += Config->MatchingGeneCoefficient * std::pow(std::abs(Node1->Bias - Node2->Bias), Config->DistanceExponent);
	}
	
	double ExcessDistance = Config->ExcessGeneCoefficient * ((ExcessNodeKeys.Num() + DisjointConnectionKeys.Num()) / double(MaxGenomeSize));
	double DisjointDistance = Config->ExcessGeneCoefficient * (DisjointNodeKeys.Num() / double(MaxGenomeSize));

	if (bDebugDistance)
	{
		LogMessage(LogLevel::Debug, "Matching Node Keys:");
		for (const uint64& NodeID : MatchingNodeKeys)
		{
			LogMessage(LogLevel::Debug, "NodeID: %llu", NodeID);
		}

		LogMessage(LogLevel::Debug, "Disjoint Node Keys:");
		for (const uint64& NodeID : DisjointNodeKeys)
		{
			LogMessage(LogLevel::Debug, "NodeID: %llu", NodeID);
		}

		LogMessage(LogLevel::Debug, "Excess Node Keys:");
		for (const uint64& NodeID : ExcessNodeKeys)
		{
			LogMessage(LogLevel::Debug, "NodeID: %llu", NodeID);
		}
	}

	//bDebugDistance = (DisjointDistance > 0.0) || (ExcessDistance > 0.0);
	double Distance = ConnectionDistance + MatchingNodeDistance + DisjointDistance + ExcessDistance;
	if (bDebugDistance)
	{
		LogMessage(LogLevel::Debug, "Connection Distance: %f", ConnectionDistance);
		LogMessage(LogLevel::Debug, "Node Distance: %f", MatchingNodeDistance);
		LogMessage(LogLevel::Debug, "Disjoint Distance: %f", DisjointDistance);
		LogMessage(LogLevel::Debug, "Excess Distance: %f", ExcessDistance);
		LogMessage(LogLevel::Debug, "Total Distance: %f", Distance);
	}
	
	return Distance;
}

// "Calculate the Manhattan distance between two genomes by summing the absolute differences between corresponding gene values. For each gene, calculate |gene1 - gene2| and sum these values to get the total distance."
double Manhattan(const GenomePtr& Genome1, const GenomePtr& Genome2, const ConfigPtr& Config)
{
	double Distance = 0.0;
	// Build a list of connections that only exist for one genome or the other and apply the excess gene coefficient for each
	// Then take the remaining list of shared connections and apply the difference in weights for each.

	return Distance;
}

double Chebyshev(const GenomePtr& Genome1, const GenomePtr& Genome2, const ConfigPtr& Config)
{
	double Distance = 0.0;
	return Distance;
}

double Calculate(const GenomePtr& Genome1, const GenomePtr& Genome2, const ConfigPtr& Config)
{
	switch (Config->DistanceMethod)
	{
	case EDistance::Euclidean: return Euclidean(Genome1, Genome2, Config);
	case EDistance::Manhattan: return Manhattan(Genome1, Genome2, Config);
	case EDistance::Chebyshev: return Chebyshev(Genome1, Genome2, Config);
	default: throw std::invalid_argument("Invalid distance method");
	}
}

} // namespace Distance

namespace CullingMethod {
NEAT::ECullingMethod FromString(const std::string& Method)
{
	if (Method == "ECullingMethod::Elitism") return ECullingMethod::Elitism;
	else if (Method == "ECullingMethod::Random") return ECullingMethod::Random;
	else if (Method == "ECullingMethod::RouletteWheel") return ECullingMethod::RouletteWheel;
	else if (Method == "ECullingMethod::Rank") return ECullingMethod::Rank;
	else if (Method == "ECullingMethod::Boltzmann") return ECullingMethod::Boltzmann;
	else return ECullingMethod::Elitism;
}

std::string ToString(ECullingMethod Method)
{
	switch (Method)
	{
	case ECullingMethod::RouletteWheel: return "ECullingMethod::RouletteWheel";
	case ECullingMethod::Random: return "ECullingMethod::Random";
	case ECullingMethod::Boltzmann: return "ECullingMethod::Boltzmann";
	case ECullingMethod::Elitism: return "ECullingMethod::Elitism";
	case ECullingMethod::Rank: return "ECullingMethod::Rank";
	default: return "ECullingMethod::Unknown";
	}
}

// Select the top N individuals based on their fitness
TArray<NEAT::GenomePtr> NEAT::CullingMethod::Elitism(const TArray<NEAT::GenomePtr>& Population, size_t N)
{
	if (Population.Num() <= N) return Population; // Not enough individuals to cull
	auto SortedPopulation = Population;
	SortedPopulation.Sort([](const GenomePtr& LHS, const GenomePtr& RHS) { return LHS->Fitness > RHS->Fitness; });
	return SortedPopulation.First(N);
}

// Fast, simple, and efficient. Encourages diversity. May not always select the best individuals if Population size too small.
TArray<GenomePtr> Random(const TArray<GenomePtr>& Population, size_t N)
{
	TArray<GenomePtr> SelectedGenomes;
	if (Population.Num() <= N) return Population; // Not enough individuals to cull
	for (const auto& Genome : Population) if (Genome->bElite) SelectedGenomes.AddUnique(Genome);

	while (SelectedGenomes.Num() < N)
	{
		int RandomIndex = GetRandomInt(0, Population.Num() - 1);
		SelectedGenomes.AddUnique(Population[RandomIndex]);
	}
	return SelectedGenomes;
}

// Selects individuals based on their fitness proportion, may not work well with very large fitness differences.
TArray<GenomePtr> RouletteWheel(const TArray<GenomePtr>& Population, size_t N)
{
	TArray<GenomePtr> SelectedGenomes;
	if (Population.Num() <= N) return Population; // Not enough individuals to cull
	auto PopulationCopy = Population;
	for (const auto& Genome : Population)
	{
		if (Genome->bElite)
		{
			SelectedGenomes.AddUnique(Genome);
			PopulationCopy.Remove(Genome);
		}
	}

	double TotalFitness = 0.0;
	for (const auto& Genome : PopulationCopy)
	{
		TotalFitness += Genome->Fitness;
	}

	while (SelectedGenomes.Num() < N)
	{
		double RandomFitness = (double)rand() / RAND_MAX * TotalFitness;
		double CurrentFitness = 0.0;
		while (CurrentFitness < RandomFitness)
		{
			int RandomIndex = GetRandomInt(0, PopulationCopy.Num() - 1);
			auto Genome = PopulationCopy[RandomIndex];
			CurrentFitness += Genome->Fitness;
			if (CurrentFitness >= RandomFitness)
			{
				SelectedGenomes.AddUnique(Genome);
				PopulationCopy.Remove(Genome);
				break;
			}
		}
	}
	return SelectedGenomes;
}

// Similar to roulette wheel selection, but uses a ranking system to reduce the impact of very large fitness differences.
TArray<GenomePtr> Rank(const TArray<GenomePtr>& Population, size_t N)
{
	TArray<GenomePtr> SelectedGenomes;
	if (Population.Num() <= N) return Population; // Not enough individuals to cull
	for (const auto& Genome : Population) if (Genome->bElite) SelectedGenomes.AddUnique(Genome);

	auto SortedPopulation = Population;
	SortedPopulation.Sort([](const GenomePtr& A, const GenomePtr& B) { return A->Fitness > B->Fitness; });
	double TotalRank = 0.0;
	for (auto i = 0; i < Population.Num(); ++i)
	{
		TotalRank += i + 1;
	}

	while (SelectedGenomes.Num() < N)
	{
		double RandomRank = (double)rand() / RAND_MAX * TotalRank;
		double CurrentRank = 0.0;
		for (size_t j = 0; j < Population.Num(); ++j)
		{
			CurrentRank += j + 1;
			if (CurrentRank >= RandomRank)
			{
				SelectedGenomes.AddUnique(SortedPopulation[j]);
				break;
			}
		}
	}

	return SelectedGenomes;
}

// Similar to roulette wheel selection, but uses a temperature parameter to control the selection pressure.
TArray<GenomePtr> Boltzmann(const TArray<GenomePtr>& Population, size_t N)
{
	TArray<GenomePtr> SelectedGenomes;
	if (Population.Num() <= N) return Population; // Not enough individuals to cull
	for (const auto& Genome : Population) if (Genome->bElite) SelectedGenomes.AddUnique(Genome);

	double TotalFitness = 0.0;
	for (const auto& Genome : Population)
	{
		TotalFitness += Genome->Fitness;
	}

	double Temperature = 1.0;
	while (SelectedGenomes.Num() < N)
	{
		double RandomFitness = (double)rand() / RAND_MAX * TotalFitness;
		double CurrentFitness = 0.0;
		for (const auto& Genome : Population)
		{
			CurrentFitness += exp(Genome->Fitness / Temperature);
			if (CurrentFitness >= RandomFitness)
			{
				SelectedGenomes.AddUnique(Genome);
				break;
			}
		}
	}

	return SelectedGenomes;
}

TArray<GenomePtr> CullPopulation(const TArray<GenomePtr>& Population, size_t N, ECullingMethod Method)
{
	if (Method == ECullingMethod::Elitism) return Elitism(Population, N);
	else if (Method == ECullingMethod::Random) return Random(Population, N);
	else if (Method == ECullingMethod::RouletteWheel) return RouletteWheel(Population, N);
	else if (Method == ECullingMethod::Rank) return Rank(Population, N);
	else if (Method == ECullingMethod::Boltzmann) return Boltzmann(Population, N);
	else return Elitism(Population, N);
}

} // namespace CullingMethod
} // namespace NEAT