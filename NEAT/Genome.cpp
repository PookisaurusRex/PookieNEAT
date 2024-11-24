#include "Genome.h"
#include "Reproduction.h"
#include "Network.h"
#include "Genes.h"
#include "Math.h"

NEAT::NeuralNetworkPtr NEAT::Genome::CreateNeuralNetwork() const
{
	if (!Config) return nullptr;
	return std::make_shared<NeuralNetwork>(std::make_shared<Genome>(*this));
}

const NEAT::ConnectionGene* NEAT::Genome::GetConnectionByID(uint64 InID) const
{
	return Genotype.Connections.Find(InID);
}

const NEAT::NodeGene* NEAT::Genome::GetNodeByID(uint64 InID) const
{
	return Genotype.Nodes.Find(InID);
}

NEAT::ConnectionGene* NEAT::Genome::GetConnectionByID(uint64 InID)
{
	return Genotype.Connections.Find(InID);
}

NEAT::NodeGene* NEAT::Genome::GetNodeByID(uint64 InID)
{
	return Genotype.Nodes.Find(InID);
}