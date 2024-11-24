#include "NEAT.h"
#include "NEAT/ExampleTrainers.h"
#include "NEAT/BuySellStockTrainer.h"

int main()
{
	NEAT::ConfigPtr Config = NEAT::Config::CreateDefaultConfig();
	Config->MaxGenerations = 10000;
	Config->StoppingFitness = 0.999995;

	//auto Trainer = XORTrainer(Config);
	//auto Trainer = XANDTrainer(Config);
	auto Trainer = DotProductTrainer(Config);

	//auto Trainer = BuySellStockTrainer(Config);
	//Trainer.RawPriceDataFilepath = "C:/Users/gmv00/source/repos/ForexBot/stock_daily/HAL_daily_json.csv";
	//Trainer.TrainingDataFilepath = "C:/Users/gmv00/source/repos/ForexBot/stock_daily/HAL_train.csv";
	//Trainer.ValidationDataFilepath = "C:/Users/gmv00/source/repos/ForexBot/stock_daily/HAL_val.csv";
	//Trainer.TestingDataFilepath = "C:/Users/gmv00/source/repos/ForexBot/stock_daily/HAL_test.csv";

	Config->NumInputs = Trainer.GetNumInputs();
	Config->NumOutputs = Trainer.GetNumOutputs();

	Trainer.Train();
	Trainer.Report();

	return 0;
}