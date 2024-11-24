#pragma once

#include <functional>
#include "Config.h"
#include "Genes.h"
#include "Array.h"
#include "Map.h"

namespace NEAT
{
	extern InnovationTracker Innovations;

	struct Genotype
	{
		using ConnectionFilter = std::function<bool(const std::pair<uint64, NEAT::ConnectionGene>&)>;
		using NodeFilter = std::function<bool(const std::pair<uint64, NEAT::NodeGene>&)>;

		TMap<uint64, NEAT::NodeGene> Nodes;
		TMap<uint64, NEAT::ConnectionGene> Connections;

		Genotype() = default;
		virtual ~Genotype() = default;

		void Prune(); // Removes connections that have invalid input or output nodes
		void ReduceGeneKeys(); // Reduces the gene keys to the smallest possible values
		void PrintGenotype() const;
		uint64 GetNewestGeneKey() const;
		ConnectionFilter ValidConnectionFilter() const;
		TArray<uint64> GetConnectionKeys() const;
		TArray<uint64> GetNodeKeys() const;
		std::string ToPrettyString() const;
		bool Deserialize(const std::string& Data);
		std::string Serialize();

		void Mutate(const ConfigPtr& Config);
		bool MutateAddNode(const ConfigPtr& Config);
		bool MutateAddConnection(const ConfigPtr& Config);
		bool MutateRemoveNode(const ConfigPtr& Config);
		bool MutateRemoveConnection(const ConfigPtr& Config);
		bool MutateModifyWeight(const ConfigPtr& Config);
		bool MutateModifyBias(const ConfigPtr& Config);
		bool MutateModifyActivation(const ConfigPtr& Config);
		bool MutateModifyAggregation(const ConfigPtr& Config);
		bool MutateToggleConnection(const ConfigPtr& Config);

		template<typename Predicate>
		TArray<uint64> GetFilteredConnectionKeys(Predicate Filter) const
		{
			TArray<uint64> FilteredKeys;
			for (const auto& ConnectionPair : Connections)
			{
				if (Filter(ConnectionPair)) FilteredKeys.Add(ConnectionPair.first);
			}
			return FilteredKeys;
		}

		template<typename Predicate>
		TArray<uint64> GetFilteredNodeKeys(Predicate Filter) const
		{
			TArray<uint64> FilteredKeys;
			for (const auto& NodePair : Nodes)
			{
				if (Filter(NodePair)) FilteredKeys.Add(NodePair.first);
			}
			return FilteredKeys;
		}
	};
} // namespace NEAT