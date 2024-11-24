#pragma once

#include <string>

namespace GVand
{
	struct Def
	{
		std::string Serialize() const;
		bool Deserialize(const std::string& Data);
		bool SaveToFile(const std::string& FilePath) const;
		bool LoadFromFile(const std::string& FilePath);
	};
}