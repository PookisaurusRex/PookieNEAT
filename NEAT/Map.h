#pragma once

#include <map>
#include "Array.h"

template<typename Key, typename Value>
class TMap
{
public:
	TMap() = default;
	~TMap() = default;

	TMap(TMap&& Other) noexcept : Map(std::move(Other.Map)) {}
	TMap(const TMap& Other) : Map(Other.Map) {}
	TMap(std::map<Key, Value>&& InMap) noexcept : Map(std::move(InMap)) {}
	TMap(const std::map<Key, Value>& InMap) : Map(InMap) {}

	TMap& operator=(const TMap& Other)
	{
		if (this != &Other)
		{
			Map = Other.Map;
		}
		return *this;
	}

	TMap& operator=(TMap&& Other) noexcept
	{
		if (this != &Other)
		{
			Map = std::move(Other.Map);
		}
		return *this;
	}

	TMap& operator=(const std::map<Key, Value>& InMap)
	{
		Map = InMap;
		return *this;
	}

	TMap& operator=(std::map<Key, Value>&& InMap) noexcept
	{
		Map = std::move(InMap);
		return *this;
	}

	void Add(const Key& InKey, const Value& InValue)
	{
		Map[InKey] = InValue;
	}

	bool Contains(const Key& InKey) const
	{
		return Map.find(InKey) != Map.end();
	}

	template<typename Predicate>
	bool ContainsByPredicate(Predicate Pred) const
	{
		for (const auto& Pair : Map)
		{
			if (Pred(Pair))
			{
				return true;
			}
		}
		return false;
	}

	Value* Find(const Key& InKey)
	{
		auto It = Map.find(InKey);
		if (It != Map.end())
		{
			return &It->second;
		}
		return nullptr;
	}

	const Value* Find(const Key& InKey) const
	{
		auto It = Map.find(InKey);
		if (It != Map.end())
		{
			return &It->second;
		}
		return nullptr;
	}

	template<typename Predicate>
	Value* FindByPredicate(Predicate Pred)
	{
		for (auto& Pair : Map)
		{
			if (Pred(Pair))
			{
				return &Pair.second;
			}
		}
		return nullptr;
	}

	template<typename Predicate>
	const Value* FindByPredicate(Predicate Pred) const
	{
		for (const auto& Pair : Map)
		{
			if (Pred(Pair))
			{
				return &Pair.second;
			}
		}
		return nullptr;
	}

	int Remove(const Key& InKey)
	{
		return int(Map.erase(InKey));
	}

	template<typename Predicate>
	int RemoveByPredicate(Predicate Pred)
	{
		int NumRemoved = 0;
		for (auto It = Map.begin(); It != Map.end();)
		{
			if (Pred(*It))
			{
				It = Map.erase(It);
				++NumRemoved;
			}
			else
			{
				++It;
			}
		}
		return NumRemoved;
	}

	void ShrinkToFit()
	{
		Map.shrink_to_fit();
	}

	template<template<typename> typename Container = TArray>
	TArray<Key> GetKeys() const
	{
		Container<Key> Keys;
		for (const auto& Pair : Map)
		{
			Keys.Add(Pair.first);
		}
		return Keys;
	}

	template<template<typename> typename Container = TArray>
	TArray<Value> GetValues() const
	{
		Container<Value> Values;
		for (const auto& Pair : Map)
		{
			Values.Add(Pair.second);
		}
		return Values;
	}

	template<typename Predicate>
	TArray<Key> FilterKeysByPredicate(Predicate Pred) const
	{
		TArray<Key> Keys;
		for (const auto& Pair : Map)
		{
			if (Pred(Pair))
			{
				Keys.Add(Pair.first);
			}
		}
		return Keys;
	}

	template<typename Predicate>
	TArray<Value> FilterValuesByPredicate(Predicate Pred) const
	{
		TArray<Value> Values;
		for (const auto& Pair : Map)
		{
			if (Pred(Pair))
			{
				Values.Add(Pair.second);
			}
		}
		return Values;
	}

	template<typename Predicate>
	TMap<Key, Value> FilterByPredicate(Predicate Pred) const
	{
		TMap<Key, Value> FilteredMap;
		for (const auto& Pair : Map)
		{
			if (Pred(Pair))
			{
				FilteredMap.Add(Pair.first, Pair.second);
			}
		}
		return FilteredMap;
	}

	Value& FindOrAdd(const Key& InKey)
	{
		auto It = Map.find(InKey);
		if (It != Map.end())
		{
			return It->second;
		}
		return Map[InKey];
	}

	Value& FindOrAdd(const Key& InKey, const Value& InValue)
	{
		auto It = Map.find(InKey);
		if (It != Map.end())
		{
			return It->second;
		}
		return Map[InKey] = InValue;
	}

	void Clear()
	{
		Map.clear();
	}

	void Reset()
	{
		Map.clear();
	}

	bool IsEmpty() const
	{
		return Map.empty();
	}

	TArray<Key> GetKeys() const
	{
		TArray<Key> Keys;
		for (const auto& Pair : Map)
		{
			Keys.Add(Pair.first);
		}
		return Keys;
	}

	TArray<Value> GetValues() const
	{
		TArray<Value> Values;
		for (const auto& Pair : Map)
		{
			Values.Add(Pair.second);
		}
		return Values;
	}

	Value& operator[](const Key& InKey)
	{
		return FindOrAdd(InKey);
	}

	const Value& operator[](const Key& InKey) const
	{
		return Map.at(InKey);
	}

	int Num() const
	{
		return int(Map.size());
	}

	void Sort()
	{
		Map.sort();
	}

	template<typename Predicate>
	void Sort(Predicate Pred)
	{
		Map.sort(Pred);
	}

	const auto begin() const { return Map.begin(); }
	const auto end() const { return Map.end(); }
	auto begin() { return Map.begin(); }
	auto end() { return Map.end(); }

private:
	std::map<Key, Value> Map;
};
