#pragma once

#include <cmath>
#include <string>
#include <stdexcept>
#include "Math.h"

namespace NEAT
{
	// Enum class for activation functions  
	enum class EActivation
	{
		Sigmoid, // Sigmoid function: maps input to a value between 0 and 1, often used in binary classification problems. It has an S-shaped curve and is continuously differentiable.  
		Tanh, // Hyperbolic tangent function: similar to sigmoid, but maps input to a value between -1 and 1. It is often used in hidden layers of neural networks.  
		Relu, // Rectified linear unit function: maps all negative values to 0 and all positive values to the same value. It is computationally efficient and widely used in deep neural networks.  
		LeakyRelu, // Leaky rectified linear unit function: similar to ReLU, but allows a small fraction of the input value to pass through, even if it's negative. This helps to avoid dying neurons.  
		Softplus, // Softplus function: similar to ReLU, but has a smooth, continuous derivative. It is often used in place of ReLU to avoid dying neurons.  
		Swish, // Swish function: a recently introduced activation function that has been shown to outperform ReLU and other activation functions in some cases. It has a smooth, continuous derivative.  
		Gelu, // Gaussian error linear unit function: similar to ReLU, but has a Gaussian distribution. It is often used in transformer models.  
		Elu, // Exponential linear unit function: similar to ReLU, but has an exponential distribution. It is often used in place of ReLU to avoid dying neurons.  
		Selu, // Scaled exponential linear unit function: similar to ELU, but has a scaled exponential distribution. It is often used in place of ELU to improve performance.  
		Softsign, // Softsign function: maps input to a value between -1 and 1, similar to tanh. It is often used in place of tanh to avoid saturation.  
		BentIdentity, // Bent identity function: a non-linear activation function that maps input to a value between -1 and 1. It is often used in place of tanh to avoid saturation.  
		BipolarSigmoid, // Bipolar sigmoid function: similar to sigmoid, but maps input to a value between -1 and 1. It is often used in binary classification problems.  
		BipolarTanh, // Bipolar hyperbolic tangent function: similar to tanh, but maps input to a value between -1 and 1. It is often used in hidden layers of neural networks.  
		Gaussian, // Gaussian function: maps input to a Gaussian distribution. It is often used in place of ReLU to avoid dying neurons.  
		Inverse, // Inverse function: maps input to its inverse. It is often used in place of ReLU to avoid dying neurons.  
		Absolute, // Absolute value function: maps input to its absolute value. It is often used in place of ReLU to avoid dying neurons.  
		Step, // Step function: maps input to 0 if it's negative, and 1 if it's positive. It is often used in binary classification problems.  
		Linear, // Linear function: maps input to its original value. It is often used in output layers of neural networks.
		Arctangent, // Arctangent function: maps input to a value between -π/2 and π/2, similar to tanh. It is often used in place of tanh to avoid saturation.,
		MAX
	};

	namespace Activation
	{
		static std::string ToString(EActivation FromEnum)
		{
			switch (FromEnum)
			{
			case EActivation::Sigmoid: return "EActivation::Sigmoid";
			case EActivation::Tanh: return "EActivation::Tanh";
			case EActivation::Relu: return "EActivation::Relu";
			case EActivation::LeakyRelu: return "EActivation::LeakyRelu";
			case EActivation::Softplus: return "EActivation::Softplus";
			case EActivation::Swish: return "EActivation::Swish";
			case EActivation::Gelu: return "EActivation::Gelu";
			case EActivation::Elu: return "EActivation::Elu";
			case EActivation::Selu: return "EActivation::Selu";
			case EActivation::Softsign: return "EActivation::Softsign";
			case EActivation::BentIdentity: return "EActivation::BentIdentity";
			case EActivation::BipolarSigmoid: return "EActivation::BipolarSigmoid";
			case EActivation::BipolarTanh: return "EActivation::BipolarTanh";
			case EActivation::Gaussian: return "EActivation::Gaussian";
			case EActivation::Inverse: return "EActivation::Inverse";
			case EActivation::Absolute: return "EActivation::Absolute";
			case EActivation::Step: return "EActivation::Step";
			case EActivation::Linear: return "EActivation::Linear";
			case EActivation::Arctangent: return "EActivation::Arctangent";
			default: return "EActivation::Unknown";
			}
		}

		static EActivation FromString(const std::string& String)
		{
			if (String == "EActivation::Sigmoid") return EActivation::Sigmoid;
			if (String == "EActivation::Tanh") return EActivation::Tanh;
			if (String == "EActivation::Relu") return EActivation::Relu;
			if (String == "EActivation::LeakyRelu") return EActivation::LeakyRelu;
			if (String == "EActivation::Softplus") return EActivation::Softplus;
			if (String == "EActivation::Swish") return EActivation::Swish;
			if (String == "EActivation::Gelu") return EActivation::Gelu;
			if (String == "EActivation::Elu") return EActivation::Elu;
			if (String == "EActivation::Selu") return EActivation::Selu;
			if (String == "EActivation::Softsign") return EActivation::Softsign;
			if (String == "EActivation::BentIdentity") return EActivation::BentIdentity;
			if (String == "EActivation::BipolarSigmoid") return EActivation::BipolarSigmoid;
			if (String == "EActivation::BipolarTanh") return EActivation::BipolarTanh;
			if (String == "EActivation::Gaussian") return EActivation::Gaussian;
			if (String == "EActivation::Inverse") return EActivation::Inverse;
			if (String == "EActivation::Absolute") return EActivation::Absolute;
			if (String == "EActivation::Step") return EActivation::Step;
			if (String == "EActivation::Linear") return EActivation::Linear;
			if (String == "EActivation::Arctangent") return EActivation::Arctangent;
			throw std::invalid_argument("Invalid activation function string");
		}

		template <typename T>
		T Sigmoid(T X) { return 1 / (1 + exp(-X)); }

		template <typename T>
		T Tanh(T X) { return 2 / (1 + exp(-2 * X)) - 1; }

		template <typename T>
		T Relu(T X) { return X > 0 ? X : 0; }

		template <typename T>
		T LeakyRelu(T X, T Alpha) { return X > 0 ? X : Alpha * X; }

		template <typename T>
		T Softplus(T X) { return log(1 + exp(X)); }

		template <typename T>
		T Swish(T X) { return X * Sigmoid(X); }

		template <typename T>
		T Gelu(T X) { return 0.5 * X * (1 + Tanh(sqrt(2 / Math::Pi) * (X + 0.044715 * pow(X, 3)))); }

		template <typename T>
		T Elu(T X, T Alpha) { return X > 0 ? X : Alpha * (exp(X) - 1); }

		template <typename T>
		T Selu(T X) { return X > 0 ? X : 1.0507 * (exp(X) - 1); }

		template <typename T>
		T Softsign(T X) { return X == 0 ? 0 : X / (1 + abs(X)); }

		template <typename T>
		T BentIdentity(T X) { return (sqrt(X * X + 1) - 1) / 2 + X; }

		template <typename T>
		T BipolarSigmoid(T X) { return 2 / (1 + exp(-X)) - 1; }

		template <typename T>
		T BipolarTanh(T X) { return 2 / (1 + exp(-2 * X)) - 1; }

		template <typename T>
		T Gaussian(T X) { return exp(-pow(X, 2)); }

		template <typename T>
		T Inverse(T X) { return X == 0 ? 0 : 1 / X; }

		template <typename T>
		T Absolute(T X) { return abs(X); }

		template <typename T>
		T Step(T X) { return X > 0 ? 1 : 0; }

		template <typename T>
		T Linear(T X) { return X; }

		template <typename T>
		T Activate(T X, EActivation Method)
		{
			switch (Method)
			{
			case EActivation::Sigmoid: return Sigmoid(X);
			case EActivation::Tanh: return Tanh(X);
			case EActivation::Relu: return Relu(X);
			case EActivation::LeakyRelu: return LeakyRelu(X, 0.01); // default alpha value  
			case EActivation::Softplus: return Softplus(X);
			case EActivation::Swish: return Swish(X);
			case EActivation::Gelu: return Gelu(X);
			case EActivation::Elu: return Elu(X, 1.0); // default alpha value  
			case EActivation::Selu: return Selu(X);
			case EActivation::Softsign: return Softsign(X);
			case EActivation::BentIdentity: return BentIdentity(X);
			case EActivation::BipolarSigmoid: return BipolarSigmoid(X);
			case EActivation::BipolarTanh: return BipolarTanh(X);
			case EActivation::Gaussian: return Gaussian(X);
			case EActivation::Inverse: return Inverse(X);
			case EActivation::Absolute: return Absolute(X);
			case EActivation::Step: return Step(X);
			case EActivation::Linear: return Linear(X);
			case EActivation::Arctangent: return atan(X); // arctangent function  
			default: throw std::invalid_argument("Invalid activation function");
			}
		}
	}	
} // namespace NEAT