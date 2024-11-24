#pragma once

#include <vector>
#include <algorithm>
#include <initializer_list>

template<typename T>
class TArray
{
protected:
	std::vector<T> Data;

public:
	TArray() = default;
	TArray(const TArray& Other) : Data(Other.Data) {}
	TArray(TArray&& Other) noexcept : Data(std::move(Other.Data)) {}
	TArray(std::initializer_list<T> InitList) : Data(std::move(InitList)) {}

	TArray& operator=(const TArray& Other)
	{
		if (this != &Other)
		{
			Data = Other.Data;
		}
		return *this;
	}

	TArray& operator=(TArray&& Other) noexcept
	{
		if (this != &Other)
		{
			Data = std::move(Other.Data);
		}
		return *this;
	}

	TArray(const std::vector<T>& InData) : Data(InData) {}
	TArray(std::vector<T>&& InData) noexcept : Data(std::move(InData)) {}

	TArray& operator=(const std::vector<T>& InData)
	{
		Data = InData;
		return *this;
	}

	TArray& operator=(std::vector<T>&& InData) noexcept
	{
		Data = std::move(InData);
		return *this;
	}

	TArray(const T& Value, size_t Count) : Data(Count, Value) {}

	TArray& operator=(const T& Value)
	{
		Data.assign(Data.size(), Value);
		return *this;
	}

	TArray& operator=(T&& Value)
	{
		Data.assign(Data.size(), std::move(Value));
		return *this;
	}

	TArray& operator+=(const TArray& Other)
	{
		Data.insert(Data.end(), Other.Data.begin(), Other.Data.end());
		return *this;
	}

	TArray& operator+=(TArray&& Other)
	{
		Data.insert(Data.end(), std::make_move_iterator(Other.Data.begin()), std::make_move_iterator(Other.Data.end()));
		return *this;
	}

	TArray& operator+=(const std::vector<T>& InData)
	{
		Data.insert(Data.end(), InData.begin(), InData.end());
		return *this;
	}

	TArray& operator+=(std::vector<T>&& InData)
	{
		Data.insert(Data.end(), std::make_move_iterator(InData.begin()), std::make_move_iterator(InData.end()));
		return *this;
	}

	TArray& operator+=(const T& Value)
	{
		Data.push_back(Value);
		return *this;
	}

	TArray& operator+=(T&& Value)
	{
		Data.push_back(std::move(Value));
		return *this;
	}

	TArray& operator<<(const TArray& Other)
	{
		Data.insert(Data.end(), Other.Data.begin(), Other.Data.end());
		return *this;
	}

	TArray& operator<<(TArray&& Other)
	{
		Data.insert(Data.end(), std::make_move_iterator(Other.Data.begin()), std::make_move_iterator(Other.Data.end()));
		return *this;
	}

	TArray& operator<<(const std::vector<T>& InData)
	{
		Data.insert(Data.end(), InData.begin(), InData.end());
		return *this;
	}

	TArray& operator<<(std::vector<T>&& InData)
	{
		Data.insert(Data.end(), std::make_move_iterator(InData.begin()), std::make_move_iterator(InData.end()));
		return *this;
	}

	TArray& operator<<(const T& Value)
	{
		Data.push_back(Value);
		return *this;
	}

	TArray& operator<<(T&& Value)
	{
		Data.push_back(std::move(Value));
		return *this;
	}

	TArray& Append(const TArray& Other)
	{
		Data.insert(Data.end(), Other.Data.begin(), Other.Data.end());
		return *this;
	}

	TArray& Append(TArray&& Other)
	{
		Data.insert(Data.end(), std::make_move_iterator(Other.Data.begin()), std::make_move_iterator(Other.Data.end()));
		return *this;
	}

	TArray& Append(const std::vector<T>& InData)
	{
		Data.insert(Data.end(), InData.begin(), InData.end());
		return *this;
	}

	TArray& Append(std::vector<T>&& InData)
	{
		Data.insert(Data.end(), std::make_move_iterator(InData.begin()), std::make_move_iterator(InData.end()));
		return *this;
	}

	TArray& Append(const T& Value)
	{
		Data.push_back(Value);
		return *this;
	}

	TArray& Append(T&& Value)
	{
		Data.push_back(std::move(Value));
		return *this;
	}

	TArray& Append(std::initializer_list<T> InitList)
	{
		Data.insert(Data.end(), InitList.begin(), InitList.end());
		return *this;
	}

	int Num() const
	{
		return int(Data.size());
	}

	bool IsValidIndex(int Index) const
	{
		return Index >= 0 && Index < int(Data.size());
	}

    template<typename Predicate>
    int CountByPredicate(Predicate Pred) const
    {
		int Count = 0;
		for (const T& Value : Data)
		{
			if (Pred(Value)) Count++;
		}
		return Count;
    }

	template<typename IntType = int>
	const T& operator[](IntType Index) const
	{
		return Data[Index];
	}

	template<typename IntType = int>
	T& operator[](IntType Index)
	{
		return Data[Index];
	}

	const T& operator[](int Index) const
	{
		return Data[Index];
	}

	T& operator[](int Index)
	{
		return Data[Index];
	}

	const T& Last() const
	{
		return Data.back();
	}

	T& Last()
	{
		return Data.back();
	}

	TArray<T> Last(int Num) const
	{
		TArray<T> Result;
		Result.Reserve(Num);
		for (int i = int(Data.size()) - 1; i >= 0 && Num > 0; i--, Num--)
		{
			Result.Add(Data[i]);
		}
		return Result;
	}

	const T& First() const
	{
		return Data.front();
	}

	T& First()
	{
		return Data.front();
	}

	TArray<T> First(int Num) const
	{
		TArray<T> Result;
		Result.Reserve(Num);
		for (int i = 0; i < int(Data.size()) && Num > 0; i++, Num--)
		{
			Result.Add(Data[i]);
		}
		return Result;
	}

	const T* GetData() const
	{
		return Data.data();
	}

	T* GetData()
	{
		return Data.data();
	}

	const std::vector<T>& GetArray() const
	{
		return Data;
	}

	std::vector<T>& GetArray()
	{
		return Data;
	}

	void Reserve(int Num)
	{
		Data.reserve(Num);
	}

	int Add(const T& Value)
	{
		Data.push_back(Value);
		return int(Data.size()) - 1;
	}

	int Add(T&& Value)
	{
		Data.push_back(std::move(Value));
		return int(Data.size()) - 1;
	}

	int AddUnique(const T& Value)
	{
		if (std::find(Data.begin(), Data.end(), Value) == Data.end())
		{
			Data.push_back(Value);
			return int(Data.size()) - 1;
		}
		return -1;
	}

	int AddUnique(T&& Value)
	{
		if (std::find(Data.begin(), Data.end(), std::move(Value)) == Data.end())
		{
			Data.push_back(std::move(Value));
			return int(Data.size()) - 1;
		}
		return -1;
	}

	T& AddGetRef(const T& Value)
	{
		Data.push_back(Value);
		return Data.back();
	}

	T& AddGetRef(T&& Value)
	{
		Data.push_back(std::move(Value));
		return Data.back();
	}

	T& AddGetRef()
	{
		Data.emplace_back();
		return Data.back();
	}

	void Insert(int Index, const T& Value)
	{
		Data.insert(Data.begin() + Index, Value);
	}

	void Insert(int Index, T&& Value)
	{
		Data.insert(Data.begin() + Index, std::move(Value));
	}

	bool RemoveAt(int Index)
	{
		if (Index >= 0 && Index < int(Data.size()))
		{
			Data.erase(Data.begin() + Index);
			return true;
		}
		return false;
	}

	int Remove(const T& Value)
	{
		auto NewEnd = std::remove(Data.begin(), Data.end(), Value);
		int NumRemoved = int(std::distance(NewEnd, Data.end()));
		Data.erase(NewEnd, Data.end());
		return NumRemoved;
	}

	int Remove(T&& Value)
	{
		auto NewEnd = std::remove(Data.begin(), Data.end(), std::move(Value));
		int NumRemoved = int(std::distance(NewEnd, Data.end()));
		Data.erase(NewEnd, Data.end());
		return NumRemoved;
	}

	template<typename Predicate>
	int RemoveByPredicate(Predicate Pred)
	{
		auto NewEnd = std::remove_if(Data.begin(), Data.end(), Pred);
		int NumRemoved = int(std::distance(NewEnd, Data.end()));
		Data.erase(NewEnd, Data.end());
		return NumRemoved;
	}

	bool Contains(const T& Value) const
	{
		return std::find(Data.begin(), Data.end(), Value) != Data.end();
	}

	bool Contains(T&& Value) const
	{
		return std::find(Data.begin(), Data.end(), std::move(Value)) != Data.end();
	}

	template<typename Predicate>
	bool ContainsByPredicate(Predicate Pred) const
	{
		return std::find_if(Data.begin(), Data.end(), Pred) != Data.end();
	}

	bool IsEmpty() const
	{
		return Data.empty();
	}

	void Reset(size_t Reserve = 0)
	{
		Data.clear();
		Data.reserve(Reserve);
	}

	void SetNum(int NewSize)
	{
		Data.resize(NewSize);
	}

	void SetNum(int NewSize, const T& Value)
	{
		Data.resize(NewSize, Value);
	}

	int FindIndex(const T& ToFind) const
	{
		auto It = std::find(Data.begin(), Data.end(), ToFind);
		if (It != Data.end())
		{
			return int(std::distance(Data.begin(), It));
		}
		return -1;
	}

	template<typename Predicate>
	int FindIndexByPredicate(Predicate Pred) const
	{
		auto It = std::find_if(Data.begin(), Data.end(), Pred);
		if (It != Data.end())
		{
			return int(std::distance(Data.begin(), It));
		}
		return -1;
	}

	template<typename Predicate>
	T* FindByPredicate(Predicate Pred)
	{
		auto It = std::find_if(Data.begin(), Data.end(), Pred);
		if (It != Data.end())
		{
			return &(*It);
		}
		return nullptr;
	}

	template<typename Predicate>
	const T* FindByPredicate(Predicate Pred) const
	{
		auto It = std::find_if(Data.begin(), Data.end(), Pred);
		if (It != Data.end())
		{
			return &(*It);
		}
		return nullptr;
	}

	template<typename Predicate>
	TArray<T> FilterByPredicate(Predicate Pred) const
	{
		TArray<T> FilteredArray;
		for (const T& Value : Data)
		{
			if (Pred(Value))
			{
				FilteredArray.Add(Value);
			}
		}
		return FilteredArray;
	}

	template<typename Predicate>
	TArray<T> FilterByPredicate(Predicate Pred)
	{
		TArray<T> FilteredArray;
		for (T& Value : Data)
		{
			if (Pred(Value))
			{
				FilteredArray.Add(Value);
			}
		}
		return FilteredArray;
	}

	void Sort()
	{
		std::sort(Data.begin(), Data.end());
	}

	template<typename Predicate>
	void Sort(Predicate Pred)
	{
		std::sort(Data.begin(), Data.end(), Pred);
	}

	const auto begin() const { return Data.begin(); }
	const auto end() const { return Data.end(); }
	auto begin() { return Data.begin(); }
	auto end() { return Data.end(); }
};