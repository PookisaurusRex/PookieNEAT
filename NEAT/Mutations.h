#pragma once

#include <memory>
#include <string>

namespace NEAT 
{
	class Genome;
	using GenomePtr = std::shared_ptr<NEAT::Genome>;

	class Config;
	//using ConfigPtr = std::shared_ptr<const NEAT::Config>;
	using ConfigPtr = std::shared_ptr<NEAT::Config>;

	enum class EMutationType
	{
		AddNode,
		AddConnection,
		RemoveNode,
		RemoveConnection,
		ModifyWeight,
		ModifyBias,
		ModifyActivation,
		ModifyAggregation,
		ToggleConnection,
		MAX
	};

	namespace Mutations
	{
		std::string ToString(EMutationType MutationType);
		EMutationType FromString(const std::string& MutationType);
	};

} // namespace NEAT 