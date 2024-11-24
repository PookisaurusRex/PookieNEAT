#pragma once  

#include <atomic>
#include <memory>
#include <string>
#include "Aggregations.h"
#include "Activations.h"
#include "Mutations.h"
#include "Config.h"
#include "Types.h"
#include "Array.h"
#include "Map.h"

namespace NEAT 
{
	enum class EGeneType { Node, Connection };
	enum class ENodeType { Input, Hidden, Output };

	struct Innovation
	{
		uint64 ID = 0;
		EMutationType MutationType = EMutationType::AddConnection;
		EGeneType GeneType = EGeneType::Connection;
		uint64 Input = 0;
		uint64 Output = 0;

		bool Matches(EMutationType Mutation, EGeneType Gene, uint64 In, uint64 Out) const
		{
			return MutationType == Mutation && GeneType == Gene && Input == In && Output == Out;
		}
	};

	struct InnovationTracker
	{
		std::atomic<uint64> NextInnovationID;
		TMap<uint64, Innovation> Innovations;

		InnovationTracker() : NextInnovationID(0) {}
		InnovationTracker(const InnovationTracker& Other) : NextInnovationID(Other.NextInnovationID.load()), Innovations(Other.Innovations) {}
		InnovationTracker(InnovationTracker&& Other) noexcept : NextInnovationID(Other.NextInnovationID.load()), Innovations(std::move(Other.Innovations)) {}

		InnovationTracker& operator=(const InnovationTracker& Other) { NextInnovationID = Other.NextInnovationID.load(); Innovations = Other.Innovations; return *this; }
		InnovationTracker& operator=(InnovationTracker&& Other) noexcept { NextInnovationID = Other.NextInnovationID.load(); Innovations = std::move(Other.Innovations); return *this; }

		uint64 GetInnovationID(EMutationType MutationType, EGeneType GeneType, uint64 Input, uint64 Output)
		{
			for (const auto& InnovationPair : Innovations)
			{
				const auto& Innovation = InnovationPair.second;
				if (Innovation.Matches(MutationType, GeneType, Input, Output)) return Innovation.ID;
			}

			auto NextID = NextInnovationID.fetch_add(1);
			Innovations[NextID] = Innovation{ NextID, MutationType, GeneType, Input, Output };
			return NextID;
		}

		void Reset(uint64 StartingInnovation)
		{
			NextInnovationID = StartingInnovation;
			Innovations.Reset();
		}
	};

	struct BaseGene
	{
		uint64 ID = 0;
		bool Enabled = true;

		BaseGene(BaseGene&& Other) noexcept : ID(Other.ID), Enabled(Other.Enabled) {}
		BaseGene(const BaseGene& Other) : ID(Other.ID), Enabled(Other.Enabled) {}
		BaseGene(uint64 InID, bool InEnabled) : ID(InID), Enabled(InEnabled) {}
		BaseGene(uint64 InID) : ID(InID), Enabled(true) {}
		BaseGene() : ID(0), Enabled(true) {}
		virtual ~BaseGene() {}

		BaseGene& operator=(BaseGene&& Other) noexcept { ID = std::move(Other.ID); Enabled = std::move(Other.Enabled); return *this; }
		BaseGene& operator=(const BaseGene& Other) { ID = Other.ID; Enabled = Other.Enabled; return *this; }
	};
	
	namespace NodeType
	{
		static std::string ToString(ENodeType NodeType)
		{
			switch (NodeType)
			{
			case ENodeType::Input: return "ENodeType::Input";
			case ENodeType::Hidden: return "ENodeType::Hidden";
			case ENodeType::Output: return "ENodeType::Output";
			default: return "Unknown";
			}
		}

		static ENodeType FromString(const std::string& String)
		{
			if (String == "ENodeType::Input") return ENodeType::Input;
			else if (String == "ENodeType::Hidden") return ENodeType::Hidden;
			else if (String == "ENodeType::Output") return ENodeType::Output;
			return ENodeType::Hidden;
		}
	}

	struct NodeGene : public BaseGene
	{
		EActivation Activation = EActivation::Sigmoid;
		EAggregation Aggregation = EAggregation::Mean;
		ENodeType Type = ENodeType::Hidden;
		double Bias = 0.0;

		explicit NodeGene(uint64 InID, ENodeType InType, EActivation InActivation, EAggregation InAggregation, double InBias, bool InEnabled) : BaseGene(InID, InEnabled), Type(InType), Activation(InActivation), Aggregation(InAggregation), Bias(InBias) {}
		explicit NodeGene(uint64 InID, ENodeType InType, EActivation InActivation, EAggregation InAggregation, double InBias) : BaseGene(InID), Type(InType), Activation(InActivation), Aggregation(InAggregation), Bias(InBias) {}
		explicit NodeGene(uint64 InID, ENodeType InType, EActivation InActivation, EAggregation InAggregation) : BaseGene(InID), Type(InType), Activation(InActivation), Aggregation(InAggregation) {}
		explicit NodeGene(uint64 InID, ENodeType InType, EActivation InActivation, double InBias) : BaseGene(InID), Type(InType), Activation(InActivation), Bias(InBias) {}
		explicit NodeGene(uint64 InID, ENodeType InType, EActivation InActivation) : BaseGene(InID), Type(InType), Activation(InActivation), Bias(0.0) {}
		NodeGene(NodeGene&& Other) noexcept : BaseGene(std::move(Other.ID), std::move(Other.Enabled)), Type(std::move(Other.Type)), Activation(std::move(Other.Activation)), Bias(std::move(Other.Bias)) {}
		NodeGene(const NodeGene& Other) : BaseGene(Other.ID, Other.Enabled), Type(Other.Type), Activation(Other.Activation), Bias(Other.Bias) {}
		NodeGene() = default;
		virtual ~NodeGene() = default;

		NodeGene& operator=(NodeGene&& Other) noexcept { ID = std::move(Other.ID); Enabled = std::move(Other.Enabled); Type = std::move(Other.Type); Activation = std::move(Other.Activation); Bias = std::move(Other.Bias); return *this; }
		NodeGene& operator=(const NodeGene& Other) { ID = Other.ID; Enabled = Other.Enabled; Type = Other.Type; Activation = Other.Activation; Bias = Other.Bias; return *this; }
	};

	using NodeGenePtr = std::shared_ptr<NodeGene>;

	struct ConnectionGene : public BaseGene
	{
		explicit ConnectionGene(uint64 InID, uint64 InSourceNode, uint64 InTargetNode, double InWeight, bool InEnabled) : BaseGene(InID, InEnabled), Input(InSourceNode), Output(InTargetNode), Weight(InWeight) {}
		explicit ConnectionGene(uint64 InID, uint64 InSourceNode, uint64 InTargetNode, double InWeight) : BaseGene(InID), Input(InSourceNode), Output(InTargetNode), Weight(InWeight) {}
		ConnectionGene(ConnectionGene&& Other) noexcept : BaseGene(std::move(Other.ID), std::move(Other.Enabled)), Input(std::move(Other.Input)), Output(std::move(Other.Output)), Weight(std::move(Other.Weight)) {}
		ConnectionGene(const ConnectionGene& Other) : BaseGene(Other.ID, Other.Enabled), Input(Other.Input), Output(Other.Output), Weight(Other.Weight) {}
		ConnectionGene() = default;

		virtual ~ConnectionGene() = default;

		ConnectionGene& operator=(ConnectionGene&& Other) noexcept { ID = std::move(Other.ID); Enabled = std::move(Other.Enabled); Input = std::move(Other.Input); Output = std::move(Other.Output); Weight = std::move(Other.Weight); return *this; }
		ConnectionGene& operator=(const ConnectionGene& Other) { ID = Other.ID; Enabled = Other.Enabled; Input = Other.Input; Output = Other.Output; Weight = Other.Weight; return *this; }

		uint64 Input = 0;
		uint64 Output = 0;
		double Weight = 1.0;
	};

	using ConnectionGenePtr = std::shared_ptr<ConnectionGene>;
}