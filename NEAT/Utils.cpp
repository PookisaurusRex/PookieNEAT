#include "Utils.h"  
#include <iostream>  
#include <fstream>  
#include <string>  
#include <cstdlib>

namespace NEAT
{
	// Initialize random seed  
	void InitializeRandomSeed(unsigned Seed /*= 0*/)
	{
		srand(static_cast<unsigned>(Seed));
	}

	// Generate random integer  
	int GetRandomInt(int Min, int Max)
	{
		return (rand() % (Max - Min + 1)) + Min;
	}

	// Generate random float  
	double GetRandomDouble(double Min, double Max)
	{
		return (double)rand() / RAND_MAX * (Max - Min) + Min;
	}

	// Log message  
	void LogMessage(LogLevel Level, const std::string& Message, bool bSaveToLog /*= false*/)
	{
		std::time_t TimeNow = std::time(nullptr);
		std::tm LocalTime;
		localtime_s(&LocalTime, &TimeNow);
		char Timestamp[20];
		std::strftime(Timestamp, 20, "%Y-%m-%d %H:%M:%S", &LocalTime);

		if (bSaveToLog)
		{
			std::ofstream LogFile("Logs/NEAT.log", std::ios_base::app);
			if (!LogFile.is_open()) return;
			LogFile << Timestamp << " [" << to_string(Level) << "] " << Message << std::endl;
			LogFile.close();
		}
		std::cout << Timestamp << " [" << to_string(Level) << "] " << Message << std::endl;
	}

	void LogMessage(LogLevel Level, const std::string& Message, const std::string& Filename)
	{
		std::ofstream LogFile(Filename, std::ios_base::app);
		if (!LogFile.is_open()) return;
		std::time_t TimeNow = std::time(nullptr);
		std::tm LocalTime;
		localtime_s(&LocalTime, &TimeNow);
		char Timestamp[20];
		std::strftime(Timestamp, 20, "%Y-%m-%d %H:%M:%S", &LocalTime);
		LogFile << Timestamp << " [" << to_string(Level) << "] " << Message << std::endl;
		LogFile.close();
		std::cout << Timestamp << " [" << to_string(Level) << "] " << Message << std::endl;
	}

	// Get log level string  
	std::string GetLogLevelString(LogLevel Level) 
	{
		switch (Level)
		{
			case Debug: return "DEBUG";
			case Info: return "INFO";
			case Warning: return "WARNING";
			case Error: return "ERROR";
			case Fatal: return "FATAL";
			default: return "UNKNOWN";
		}
	}

	// Print genome information  
	/*void PrintGenomeInfo(const Genome& Genome) 
	{
		LogMessage(Debug, "Genome Information:");
		LogMessage(Debug, "  ID: " + std::to_string(genome.GetId()));
		LogMessage(Debug, "  Fitness: " + std::to_string(genome.Fitness));
		LogMessage(Debug, "  Number of Inputs: " + std::to_string(genome.GetNumInputs()));
		LogMessage(Debug, "  Number of Outputs: " + std::to_string(genome.GetNumOutputs()));
		LogMessage(Debug, "  Number of Hidden Layers: " + std::to_string(genome.GetNumHiddenLayers()));
	}*/

	// Print neural network information  
	/*void PrintNeuralNetworkInfo(const NeuralNetwork& Network) 
	{
		LogMessage(Debug, "Neural Network Information:");
		LogMessage(Debug, "  Number of Inputs: " + std::to_string(Network.GetNumInputs()));
		LogMessage(Debug, "  Number of Outputs: " + std::to_string(Network.GetNumOutputs()));
		LogMessage(Debug, "  Number of Hidden Layers: " + std::to_string(Network.GetNumHiddenLayers()));
	}*/

	// Print population information  
	/*void PrintPopulationInfo(const Population& Agents)
	{
		LogMessage(Debug, "Population Information:");
		LogMessage(Debug, "  Size: " + std::to_string(Agents.GetSize()));
		LogMessage(Debug, "  Generation: " + std::to_string(Agents.GetGeneration()));
	}*/
}