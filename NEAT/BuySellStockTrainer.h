#include "NEAT.h"
#include <fstream>
#include <string>

struct FStockData
{
	double Open = 0.0;
	double High = 0.0;
	double Low = 0.0;
	double Close = 0.0;
	double Volume = 0.0;

	FStockData() = default;
	~FStockData() = default;

	FStockData(double InOpen, double InHigh, double InLow, double InClose, double InVolume) : Open(InOpen), High(InHigh), Low(InLow), Close(InClose), Volume(InVolume) { }
};

struct FExtendedStockData : public FStockData
{
	double SMA5 = 0.0;		// Simple Moving Average over the last five days.
	double SMA10 = 0.0;		// Simple Moving Average over the last ten days.
	double SMA20 = 0.0;		// Simple Moving Average over the last twenty days.
	double EMA5 = 0.0;		// Exponential Moving Average over the last five days.
	double EMA10 = 0.0;		// Exponential Moving Average over the last ten days.
	double EMA20 = 0.0;		// Exponential Moving Average over the last twenty days.
	double RSI = 0.0;		// Relative Strength Index.
	double BB_Middle = 0.0;	// Bollinger Bands Middle Line.
	double BB_Upper = 0.0;	// Bollinger Bands Upper Line.
	double BB_Lower = 0.0;	// Bollinger Bands Lower Line.
	double Conversion = 0.0;// Ichimoku Cloud Conversion Line.
	double BaseLine = 0.0;	// Ichimoku Cloud Base Line.
	double LeadingA = 0.0;	// Ichimoku Cloud Leading Span A.
	double LeadingB = 0.0;	// Ichimoku Cloud Leading Span B.

	FExtendedStockData() = default;
	~FExtendedStockData() = default;

	FExtendedStockData(double InOpen, double InHigh, double InLow, double InClose, double InVolume, double InSMA5, double InSMA10, double InSMA20, double InEMA5, double InEMA10, double InEMA20,
		double InRSI, double InBB_Middle, double InBB_Upper, double InBB_Lower, double InConversion, double InBaseLine, double InLeadingA, double InLeadingB)
		: FStockData(InOpen, InHigh, InLow, InClose, InVolume), SMA5(InSMA5), SMA10(InSMA10), SMA20(InSMA20), EMA5(InEMA5), EMA10(InEMA10), EMA20(InEMA20),
		RSI(InRSI), BB_Middle(InBB_Middle), BB_Upper(InBB_Upper), BB_Lower(InBB_Lower), Conversion(InConversion), BaseLine(InBaseLine), LeadingA(InLeadingA), LeadingB(InLeadingB) { }
};

enum class EStockAction
{
	StrongBuy,
	Buy,
	Hold,
	Sell,
	StrongSell
};

namespace StockAction
{
	static std::string ToString(EStockAction Action)
	{
		switch (Action)
		{
		case EStockAction::StrongBuy: return "EStockAction::StrongBuy";
		case EStockAction::Buy: return "EStockAction::Buy";
		case EStockAction::Hold: return "EStockAction::Hold";
		case EStockAction::Sell: return "EStockAction::Sell";
		case EStockAction::StrongSell: return "EStockAction::StrongSell";
		default: return "EStockAction::Unknown";
		}
	}

	static EStockAction FromString(const std::string& String)
	{
		if (String == "EStockAction::StrongBuy") return EStockAction::StrongBuy;
		if (String == "EStockAction::Buy") return EStockAction::Buy;
		if (String == "EStockAction::Hold") return EStockAction::Hold;
		if (String == "EStockAction::Sell") return EStockAction::Sell;
		if (String == "EStockAction::StrongSell") return EStockAction::StrongSell;
		return EStockAction::Hold;
	}

	static EStockAction FromDouble(double Value)
	{
		Value = NEAT::Math::Clamp(Value, -1.0, 1.0);
		if (Value <= -0.8) return EStockAction::StrongSell;
		if (Value >= 0.8) return EStockAction::StrongBuy;
		if (Value <= -0.2) return EStockAction::Sell;
		if (Value >= 0.2) return EStockAction::Buy;
		return EStockAction::Hold; // If the value is between -0.2 and 0.2, then hold.
	}
}

class BuySellStockTrainer final : public NEAT::Trainer
{
	using ThisClass = BuySellStockTrainer;
	using Super = NEAT::Trainer;

public:
	constexpr static uint8 LookbackDays = 30; // The number of days to look back when making a prediction.
	constexpr static uint8 NumFeatures = 5; // The number of features per day (Open, High, Low, Close, Volume)
	constexpr static uint8 NumInputs = NumFeatures * LookbackDays; // 5 features per day * LookbackDays
	constexpr static uint8 NumOutputs = 1; // The output is a single value representing the stock action.
	constexpr static uint8 PredictionWindow = 5; // The number of days to predict into the future.
	constexpr static double StrongIndicatorThreshold = 0.10; // The percent change in price to indicate a strong buy/sell
	constexpr static double MinActionThreshold = 0.02; // The percent change in price to indicate a buy/sell
	constexpr static uint8 RunForwardDays = 100; // The number of days to run the network forward to make predictions.

	int GetNumInputs() const override { return NumInputs; } // Returns the number of inputs for the neural network
	int GetNumOutputs() const override { return NumOutputs; } // Returns the number of outputs for the neural network

	struct FInputData
	{
		FStockData StockData[LookbackDays]; // The last LookbackDays of stock data.

		TArray<double> ToArray() const
		{
			TArray<double> Array;
			Array.Reserve(NumInputs); // Total number of inputs (factoring for the extended stock data) = 19*LookbackDays + LookbackDays = 20*LookbackDays
			
			for (int Idx = 0, StopIdx = LookbackDays; Idx != StopIdx; ++Idx)
			{
				const auto& Daily = StockData[Idx];
				Array.Append({ Daily.Open, Daily.High, Daily.Low, Daily.Close, Daily.Volume });
			}

			return Array;
		}
	};

	enum class EDataType
	{
		Training,
		Validation,
		Testing
	};

	EDataType ExtendedDataType = EDataType::Training;
	TMap<std::string, FStockData> InputStockData;
	TMap<std::string, FStockData> RawPriceData;
	TMap<std::string, FInputData> InputData;
	TMap<std::string, double> OutputPercentChanges;

	TMap<std::string, TArray<double>> ParseCSV(const std::string& Filepath) const
	{
		TMap<std::string, TArray<double>> Data;
		std::ifstream File(Filepath);
		if (!File.is_open()) return Data;

		std::string Line;
		std::getline(File, Line); // Skip the header.

		while (std::getline(File, Line))
		{
			std::stringstream Stream(Line);
			std::string Symbol;
			std::getline(Stream, Symbol, ',');
			double Value;
			TArray<double> Values;
			while (Stream >> Value)
			{
				Values.Add(Value);
				if (Stream.peek() == ',') Stream.ignore();
			}
			Data.Add(Symbol, Values);
		}

		return Data;
	}

	const FStockData* FindInputDataByDate(const std::string& Date, int DayOffset = 0) const
	{
		const auto& Dates = RawPriceData.GetKeys();
		auto FoundIdx = Dates.FindIndex(Date);
		if (FoundIdx == INDEX_NONE) return nullptr;
		if (!Dates.IsValidIndex(FoundIdx + DayOffset)) return nullptr;
		return &InputStockData[Dates[FoundIdx + DayOffset]];
	}

	FStockData* FindInputDataByDate(const std::string& Date, int DayOffset = 0)
	{
		const auto& Dates = InputStockData.GetKeys();
		auto FoundIdx = Dates.FindIndex(Date);
		if (FoundIdx == INDEX_NONE) return nullptr;
		if (!Dates.IsValidIndex(FoundIdx + DayOffset)) return nullptr;
		return &InputStockData[Dates[FoundIdx + DayOffset]];
	}

	const FStockData* FindRawDataByDate(const std::string& Date, int DayOffset = 0) const
	{
		const auto& Dates = RawPriceData.GetKeys();
		auto FoundIdx = Dates.FindIndex(Date);
		if (FoundIdx == INDEX_NONE) return nullptr;
		if (!Dates.IsValidIndex(FoundIdx + DayOffset)) return nullptr;
		return &RawPriceData[Dates[FoundIdx + DayOffset]];
	}	

	FStockData* FindRawDataByDate(const std::string& Date, int DayOffset = 0)
	{
		const auto& Dates = RawPriceData.GetKeys();
		auto FoundIdx = Dates.FindIndex(Date);
		if (FoundIdx == INDEX_NONE) return nullptr;
		if (!Dates.IsValidIndex(FoundIdx + DayOffset)) return nullptr;
		return &RawPriceData[Dates[FoundIdx + DayOffset]];
	}

	void InitializeRawPriceData()
	{
		if (!RawPriceDataFilepath.empty())
		{
			const auto& ParsedRawData = ParseCSV(RawPriceDataFilepath);
			for (const auto& ParsedDaily : ParsedRawData)
			{
				const auto& Date = ParsedDaily.first;
				const auto& DailyData = ParsedDaily.second;
				RawPriceData[Date] = FStockData{ DailyData[0], DailyData[1], DailyData[2], DailyData[3], DailyData[4] };
			}
		}
	}

	void InitializeInputStockPriceData()
	{
		std::string InputDataFilepath;
		switch (ExtendedDataType)
		{
		case EDataType::Training: InputDataFilepath = TrainingDataFilepath; break;
		case EDataType::Validation: InputDataFilepath = ValidationDataFilepath; break;
		case EDataType::Testing: InputDataFilepath = TestingDataFilepath; break;
		}

		if (!InputDataFilepath.empty())
		{
			const auto& ParsedExtendedData = ParseCSV(InputDataFilepath);
			for (const auto& ParsedDaily : ParsedExtendedData)
			{
				const auto& Date = ParsedDaily.first;
				const auto& DailyData = ParsedDaily.second;
				InputStockData[Date] = FStockData{ DailyData[0], DailyData[1], DailyData[2], DailyData[3], DailyData[4] };
			}
		}
	}

	std::string GetFirstValidInputDate() const
	{
		// This needs to be the first date in the extended data that has LookbackDays many days preceding it (to fully populate the input).
		const auto& Dates = InputStockData.GetKeys();
		if (Dates.Num() < LookbackDays) return "";
		return Dates[LookbackDays];
	}

	std::string GetLastValidInputDate() const
	{
		// This needs to be the last date in the raw data that has PredictionWindow many days following it (to fully populate the output).
		auto Dates = InputStockData.GetKeys();
		if (Dates.Num() < PredictionWindow + RunForwardDays) return ""; // Ensure that there are enough dates to predict into the future.
		auto FirstDayIdx = Dates.FindIndex(GetFirstValidInputDate()); // Grab the index of the first day that has LookbackDays many days preceding it.
		auto LastDayIdx = FirstDayIdx + RunForwardDays; // Calculate the last date's index from the first date's index + the RunForwardDays window.
		if (!Dates.IsValidIndex(LastDayIdx)) return ""; // Ensure that index is valid 
		return Dates[LastDayIdx]; // Return the last date of the run.
	}

	void PopulateInputData()
	{
		const auto& Dates = InputStockData.GetKeys();
		auto StartIdx = Dates.FindIndex(GetFirstValidInputDate());
		auto StopIdx = Dates.FindIndex(GetLastValidInputDate());
		if (!Dates.IsValidIndex(StartIdx) || !Dates.IsValidIndex(StopIdx)) return;
		if (StartIdx >= StopIdx) return;

		for (int Idx = StartIdx; Idx != StopIdx; ++Idx)
		{
			const auto& Date = Dates[Idx];
			FInputData& CurrentInput = InputData[Date];
			for (int StockDataIdx = LookbackDays - 1; StockDataIdx != INDEX_NONE; --StockDataIdx) // Iterate backwards so that the most recent data is first
			{
				if (const auto* StockData = FindInputDataByDate(Date, -StockDataIdx)) // Find the extended data for the offset date
				{
					CurrentInput.StockData[StockDataIdx] = *StockData;
				}
			}
		}
	}

	void PopulateOutputData()
	{
		const auto& Dates = RawPriceData.GetKeys();
		auto StartIdx = Dates.FindIndex(GetFirstValidInputDate());
		auto StopIdx = Dates.FindIndex(GetLastValidInputDate());
		if (!Dates.IsValidIndex(StartIdx) || !Dates.IsValidIndex(StopIdx)) return;
		if (StartIdx >= StopIdx) return;

		for (int Idx = StartIdx; Idx != StopIdx; ++Idx)
		{
			const auto& CurrentDate = Dates[Idx];
			const auto& CurrentData = RawPriceData[CurrentDate];
			const auto& FutureData = RawPriceData[Dates[Idx + PredictionWindow]];
			const double PercentChange = (FutureData.Close - CurrentData.Close) / CurrentData.Close;
			OutputPercentChanges.Add(CurrentDate, PercentChange);
		}
	}

	void Initialize() override
	{
		Super::Initialize();

		InitializeRawPriceData();
		InitializeInputStockPriceData();
		
		PopulateInputData();
		PopulateOutputData();

		// Double check that the number of inputs matches the number of outputs
		if (InputData.Num() != OutputPercentChanges.Num()) NEAT::LogMessage(NEAT::LogLevel::Error, "The number of inputs does not match the number of outputs."); return;

		// Ensure that every date in the input data has a corresponding output
		for (const auto& InputPair : InputData)
		{
			if (!OutputPercentChanges.Contains(InputPair.first)) NEAT::LogMessage(NEAT::LogLevel::Error, "The input data does not have a corresponding output for date %s.", InputPair.first.c_str()); return;
		}
	}

	double Evaluate(const NEAT::GenomePtr& Genome) override
	{
		NEAT::NeuralNetworkPtr Network = Genome->CreateNeuralNetwork();
		if (!Network) return 0.0;

		const auto& Dates = InputData.GetKeys();

		TArray<double> Predictions;
		Predictions.Reserve(Dates.Num());

		FInputData CurrentInput;
		for (auto CurrentDateIdx = 0, StopIdx = Dates.Num(); CurrentDateIdx != StopIdx; ++CurrentDateIdx)
		{
			const auto& CurrentDate = Dates[CurrentDateIdx];
			const auto& Inputs = InputData[CurrentDate];
			const auto Outputs = Network->Evaluate(Inputs.ToArray());
			double Prediction = Outputs.IsValidIndex(0) ? Outputs[0] : 0.0;
			Predictions.Add(Prediction);
		}

		double Fitness = 0.0;
		for (int PredictionIdx = 0, StopIdx = Predictions.Num(); PredictionIdx != StopIdx; ++PredictionIdx)
		{
			const auto& CurrentDate = Dates[PredictionIdx];
			const auto& CurrentPrediction = Predictions[PredictionIdx];
			const auto& CurrentPercentChange = OutputPercentChanges[CurrentDate];
			const auto& CurrentAction = StockAction::FromDouble(CurrentPrediction);

			if (CurrentPercentChange > 0.0) // If the stock price increased
			{
				if (CurrentAction == EStockAction::Buy || CurrentAction == EStockAction::StrongBuy) // If the prediction was to buy
				{
					bool bCorrectStrength = (CurrentAction == EStockAction::StrongBuy) && (CurrentPercentChange > StrongIndicatorThreshold);
					double StrengthMultiplier = bCorrectStrength ? 2.0 : 1.0;
					Fitness += (StrengthMultiplier * CurrentPercentChange);
				}
				else if (CurrentAction == EStockAction::Sell || CurrentAction == EStockAction::StrongSell) // If the prediction was to sell
				{
					double StrengthMultiplier = (CurrentAction == EStockAction::StrongSell) ? 2.0 : 1.0;
					Fitness -= (StrengthMultiplier * CurrentPercentChange);
				}
			}
			else if (CurrentPercentChange < 0.0)
			{
				if (CurrentAction == EStockAction::Sell || CurrentAction == EStockAction::StrongSell) // If the prediction was to sell
				{
					bool bCorrectStrength = (CurrentAction == EStockAction::StrongSell) && (CurrentPercentChange < -StrongIndicatorThreshold);
					double StrengthMultiplier = bCorrectStrength ? 2.0 : 1.0;
					Fitness += (StrengthMultiplier * CurrentPercentChange);
				}
				else if (CurrentAction == EStockAction::Buy || CurrentAction == EStockAction::StrongBuy) // If the prediction was to buy
				{
					double StrengthMultiplier = (CurrentAction == EStockAction::StrongBuy) ? 2.0 : 1.0;
					Fitness -= (StrengthMultiplier * CurrentPercentChange);
				}
			}
		}

		return Fitness;
	}

public:
	BuySellStockTrainer(const NEAT::ConfigPtr& Config) : NEAT::Trainer(Config) { }

	std::string RawPriceDataFilepath = "";
	std::string TrainingDataFilepath = "";
	std::string ValidationDataFilepath = "";
	std::string TestingDataFilepath = "";

	void Report()
	{
		using namespace NEAT;
		LogMessage(LogLevel::Info, "Raw Price Data Filepath: %s", RawPriceDataFilepath.c_str());
		LogMessage(LogLevel::Info, "Training Data Filepath: %s", TrainingDataFilepath.c_str());
		LogMessage(LogLevel::Info, "Validation Data Filepath: %s", ValidationDataFilepath.c_str());
		LogMessage(LogLevel::Info, "Testing Data Filepath: %s", TestingDataFilepath.c_str());
	}
};