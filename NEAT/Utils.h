#pragma once

#include <iostream>  
#include <fstream>  
#include <string>  
#include <vector>  
#include <random>
#include <cstdio>

#define BREAKPOINT() __debugbreak()

namespace NEAT 
{
	enum LogLevel 
	{
		Debug,
		Info,
		Warning,
		Error,
		Fatal
	};

	static std::string to_string(LogLevel Level) 
	{
		switch (Level) 
		{
		case Debug: return "Debug";
		case Info: return "Info";
		case Warning: return "Warning";
		case Error: return "Error";
		case Fatal: return "Fatal";
		default: return "Unknown";
		}
	}

	// Function declarations  
	void InitializeRandomSeed(unsigned Seed = 0); // Initialize random seed  
	int GetRandomInt(int Min, int Max); // Generate random integer  
	double GetRandomDouble(double Min, double Max); // Generate random double  

	void LogMessage(LogLevel Level, const std::string& Message, bool bSaveToLog = false); // Log message to file  
	void LogMessage(LogLevel Level, const std::string& Message, const std::string& Filename); // Log message to file with filename  

	template<typename... Args>
	inline void LogMessage(LogLevel Level, const std::string& Message, const Args&... args)
	{
		// Implements printf-like logic on the Message string and the args... to resolve a complex message with its component parameters
		char Buffer[512];
		sprintf_s(Buffer, 512, Message.data(), args...);
		LogMessage(Level, std::string(Buffer), false);
	}


	//void PrintGenome(const Genome& Genome); // Print genome to console  
	//void PrintNeuralNetwork(const NeuralNetwork& Network); // Print neural network to console  
	//void PrintPopulation(const Population& Agents); // Print population to console

} // namespace NEAT  
