#include "NEAT.h"

// Create a new NEAT trainer that produces a genome that performs XOR, running for 1000 generations or until a solution is found
class XORTrainer : public NEAT::Trainer
{
	double Evaluate(const NEAT::GenomePtr& Genome) override final
	{
		// Create a neural network from the genome  
		NEAT::NeuralNetworkPtr Network = Genome->CreateNeuralNetwork();

		// Define the XOR inputs and expected outputs  
		TArray<TArray<double>> Inputs = 
		{
			{0.0, 0.0},
			{0.0, 1.0},
			{1.0, 0.0},
			{1.0, 1.0}
		};
		std::vector<double> ExpectedOutputs = { 0.0, 1.0, 1.0, 0.0 };

		// Evaluate the neural network on the XOR inputs  
		TArray<double> Outputs;
		for (const auto& Input : Inputs)
		{
			Outputs.Add(Network->Evaluate(Input)[0]);
		}

		return NEAT::Fitness::Regression::MeanAbsoluteError(Outputs, ExpectedOutputs);
	}

public:
	XORTrainer(const NEAT::ConfigPtr& Config) : NEAT::Trainer(Config) { }

	int GetNumInputs() const override { return 2; } // Returns the number of inputs for the neural network
	int GetNumOutputs() const override { return 1; } // Returns the number of outputs for the neural network

	void Report()
	{
		if (bHasBestGenome)
		{
			NEAT::LogMessage(NEAT::LogLevel::Info, "Potential solution found! XOR operation successfully evolved.");

			// Test the solution with a set of inputs  
			TArray<TArray<double>> inputs = {
			  {0.0, 0.0},
			  {0.0, 1.0},
			  {1.0, 0.0},
			  {1.0, 1.0}
			};
			TArray<double> expectedOutputs = { 0.0, 1.0, 1.0, 0.0 };

			NEAT::NeuralNetworkPtr Network = BestGenome.CreateNeuralNetwork();

			NEAT::LogMessage(NEAT::LogLevel::Info, "Testing solution with inputs:");
			for (int i = 0; i < inputs.Num(); i++)
			{
				TArray<double> output = Network->Evaluate(inputs[i]);
				NEAT::LogMessage(NEAT::LogLevel::Info, "Input: [" + std::to_string(inputs[i][0]) + ", " + std::to_string(inputs[i][1]) + "] Expected Output: " + std::to_string(expectedOutputs[i]) + " Actual Output: " + std::to_string(output[0]));
			}
		}
		else
		{
			NEAT::LogMessage(NEAT::LogLevel::Info, "No solution found after 1000 generations.");
		}
	}
};

class XANDTrainer : public NEAT::Trainer
{
	double Evaluate(const NEAT::GenomePtr& Genome) override final
	{
		// Create a neural network from the genome  
		NEAT::NeuralNetworkPtr Network = Genome->CreateNeuralNetwork();

		// Define the XOR inputs and expected outputs  
		TArray<TArray<double>> Inputs = {
			{0.0, 0.0},
			{0.0, 1.0},
			{1.0, 0.0},
			{1.0, 1.0}
		};
		std::vector<double> ExpectedOutputs = { 1.0, 0.0, 0.0, 1.0 };

		TArray<double> Outputs;
		for (const auto& Input : Inputs)
		{
			Outputs.Add(Network->Evaluate(Input)[0]);
		}

		return NEAT::Fitness::Regression::MeanAbsoluteError(Outputs, ExpectedOutputs);
	}

public:
	XANDTrainer(const NEAT::ConfigPtr& Config) : NEAT::Trainer(Config) { }

	int GetNumInputs() const override { return 2; } // Returns the number of inputs for the neural network
	int GetNumOutputs() const override { return 1; } // Returns the number of outputs for the neural network

	void Report()
	{
		if (bHasBestGenome)
		{
			NEAT::LogMessage(NEAT::LogLevel::Info, "Potential solution found! XAND operation successfully evolved.");

			// Test the solution with a set of inputs  
			TArray<TArray<double>> inputs = {
			  {0.0, 0.0},
			  {0.0, 1.0},
			  {1.0, 0.0},
			  {1.0, 1.0}
			};
			TArray<double> expectedOutputs = { 1.0, 0.0, 0.0, 1.0 };

			NEAT::NeuralNetworkPtr Network = BestGenome.CreateNeuralNetwork();

			NEAT::LogMessage(NEAT::LogLevel::Info, "Testing solution with inputs:");
			for (int i = 0; i < inputs.Num(); i++)
			{
				TArray<double> output = Network->Evaluate(inputs[i]);
				NEAT::LogMessage(NEAT::LogLevel::Info, "Input: [" + std::to_string(inputs[i][0]) + ", " + std::to_string(inputs[i][1]) + "] Expected Output: " + std::to_string(expectedOutputs[i]) + " Actual Output: " + std::to_string(output[0]));
			}
		}
		else
		{
			NEAT::LogMessage(NEAT::LogLevel::Info, "No solution found after 1000 generations.");
		}
	}
};

class DotProductTrainer : public NEAT::Trainer
{
	struct InputPair
	{
		TArray<double> Vector1;
		TArray<double> Vector2;

		TArray<double> GetFlattenedInputs()
		{
			TArray<double> FlattenedInputs;
			FlattenedInputs.Append(Vector1);
			FlattenedInputs.Append(Vector2);
			return FlattenedInputs;
		}

		double GetExpectedOutput()
		{
			double DotProduct = 0.0;
			for (int i = 0; i < Vector1.Num(); i++)
			{
				DotProduct += Vector1[i] * Vector2[i];
			}
			return DotProduct;
		}
	};

	double Evaluate(const NEAT::GenomePtr& Genome) override final
	{
		// Create a neural network from the genome  
		NEAT::NeuralNetworkPtr Network = Genome->CreateNeuralNetwork();
		TArray<InputPair> Inputs;
		TArray<double> ExpectedOutputs;
		for (int Idx = 0; Idx != GetNumInputs(); ++Idx)
		{
			TArray<double> Vector1 = { NEAT::GetRandomDouble(-1.0, 1.0), NEAT::GetRandomDouble(-1.0, 1.0), NEAT::GetRandomDouble(-1.0, 1.0) };
			TArray<double> Vector2 = { NEAT::GetRandomDouble(-1.0, 1.0), NEAT::GetRandomDouble(-1.0, 1.0), NEAT::GetRandomDouble(-1.0, 1.0) };
			InputPair Pair = { Vector1, Vector2 };
			ExpectedOutputs.Add(Pair.GetExpectedOutput());
			Inputs.Add(std::move(Pair));
		}

		TArray<double> Outputs;
		for (int i = 0; i < Inputs.Num(); i++)
		{
			auto Output = Network->Evaluate(Inputs[i].GetFlattenedInputs());
			Outputs.Add(Output[0]);
		}

		return NEAT::Fitness::Regression::MeanAbsoluteError(Outputs, ExpectedOutputs);
	}

public:
	DotProductTrainer(const NEAT::ConfigPtr& Config) : NEAT::Trainer(Config) { }
	int GetNumInputs() const override { return 4; } // Returns the number of inputs for the neural network
	int GetNumOutputs() const override { return 1; } // Returns the number of outputs for the neural network

	void Report()
	{
		if (bHasBestGenome)
		{
			NEAT::LogMessage(NEAT::LogLevel::Info, "Potential solution found! Dot product operation successfully evolved.");

			NEAT::NeuralNetworkPtr Network = BestGenome.CreateNeuralNetwork();
			NEAT::LogMessage(NEAT::LogLevel::Info, "Testing solution with inputs:");
			for (int Idx = 0; Idx < GetNumInputs(); ++Idx)
			{
				TArray<double> Vector1 = { NEAT::GetRandomDouble(-1.0, 1.0), NEAT::GetRandomDouble(-1.0, 1.0), NEAT::GetRandomDouble(-1.0, 1.0) };
				TArray<double> Vector2 = { NEAT::GetRandomDouble(-1.0, 1.0), NEAT::GetRandomDouble(-1.0, 1.0), NEAT::GetRandomDouble(-1.0, 1.0) };
				InputPair Pair = { Vector1, Vector2 };
				TArray<double> output = Network->Evaluate(Pair.GetFlattenedInputs());
				NEAT::LogMessage(NEAT::LogLevel::Info, "Input: [" + std::to_string(Vector1[0]) + ", " + std::to_string(Vector1[1]) + ", " + std::to_string(Vector1[2]) + "] [" + std::to_string(Vector2[0]) + ", " + std::to_string(Vector2[1]) + ", " + std::to_string(Vector2[2]) + "] Expected Output: " + std::to_string(Pair.GetExpectedOutput()) + " Actual Output: " + std::to_string(output[0]));
			}
		}
		else
		{
			NEAT::LogMessage(NEAT::LogLevel::Info, "No solution found after 1000 generations.");
		}
	}
};