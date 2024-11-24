#include "Mutations.h"
#include "Genome.h"
#include "Config.h"
#include "Utils.h"
#include "Genes.h"
#include "Math.h"

std::string NEAT::Mutations::ToString(NEAT::EMutationType MutationType)
{
	switch (MutationType)
	{
	case EMutationType::AddNode: return "EMutationType::AddNode";
	case EMutationType::AddConnection: return "EMutationType::AddConnection";
	case EMutationType::RemoveNode: return "EMutationType::RemoveNode";
	case EMutationType::RemoveConnection: return "EMutationType::RemoveConnection";
	case EMutationType::ModifyWeight: return "EMutationType::ModifyWeight";
	case EMutationType::ModifyBias: return "EMutationType::ModifyBias";
	case EMutationType::ModifyActivation: return "EMutationType::ModifyActivation";
	case EMutationType::ModifyAggregation: return "EMutationType::ModifyAggregation";
	case EMutationType::ToggleConnection: return "EMutationType::ToggleConnection";
	default: return "Unknown";
	}
}

NEAT::EMutationType NEAT::Mutations::FromString(const std::string& MutationType)
{
	if (MutationType == "EMutationType::AddNode") return EMutationType::AddNode;
	else if (MutationType == "EMutationType::AddConnection") return EMutationType::AddConnection;
	else if (MutationType == "EMutationType::RemoveNode") return EMutationType::RemoveNode;
	else if (MutationType == "EMutationType::RemoveConnection") return EMutationType::RemoveConnection;
	else if (MutationType == "EMutationType::ModifyWeight") return EMutationType::ModifyWeight;
	else if (MutationType == "EMutationType::ModifyBias") return EMutationType::ModifyBias;
	else if (MutationType == "EMutationType::ModifyActivation") return EMutationType::ModifyActivation;
	else if (MutationType == "EMutationType::ModifyAggregation") return EMutationType::ModifyAggregation;
	else if (MutationType == "EMutationType::ToggleConnection") return EMutationType::ToggleConnection;
	return EMutationType::AddNode;
}