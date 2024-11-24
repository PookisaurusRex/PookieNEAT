#pragma once

#include <limits>
#include <cmath>

namespace NEAT {
namespace Math {
	constexpr double Pi = 3.14159265358979323846;
	constexpr double Tau = 6.28318530717958647692;
	constexpr double E = 2.71828182845904523536;
	constexpr double Phi = 1.61803398874989484820;
	constexpr double Sqrt2 = 1.41421356237309504880;
	constexpr double Sqrt3 = 1.73205080756887729352;
	constexpr double Sqrt5 = 2.23606797749978969640;
	constexpr double Sqrt7 = 2.64575131106459059050;
	constexpr double Sqrt11 = 3.31662479035539984911;
	constexpr double Sqrt13 = 3.60555127546398929312;
	constexpr double Sqrt17 = 4.12310562561766054982;
	constexpr double Sqrt19 = 4.35889894354067436848;
	constexpr double Log2E = 1.44269504088896340736;
	constexpr double Log10E = 0.434294481903251827651;

	constexpr double KindaSmallNumber = 1.0e-4;
	constexpr double VerySmallNumber = 1.0e-8;
	constexpr double KindaBigNumber = 3.4e+8;
	constexpr double BigNumber = 3.4e+16;

	constexpr double NaN = std::numeric_limits<double>::quiet_NaN();
	constexpr double Infinity = std::numeric_limits<double>::infinity();

	template<typename T>
	inline T Abs(const T& Value)
	{
		return Value < 0 ? -Value : Value;
	}

	inline bool IsNaN(double Value)
	{
		return std::isnan(Value);
	}

	inline bool IsFinite(double Value)
	{
		return std::isfinite(Value);
	}

	inline bool IsInfinite(double Value)
	{
		return std::isinf(Value);
	}

	inline bool AlmostZero(double Value, double Tolerance = KindaSmallNumber)
	{
		return Abs(Value) <= Tolerance;
	}

	inline bool NearlyEqual(double A, double B, double Tolerance = KindaSmallNumber)
	{
		return Abs(A - B) <= Tolerance;
	}

	template<typename T>
	inline T Lerp(const T& A, const T& B, double Alpha)
	{
		return A + Alpha * (B - A);
	}

	template<typename T>
	T InverseLerp(const T& A, const T& B, const T& Value)
	{
		return (Value - A) / (B - A);
	}

	template<typename T>
	inline T Remap(const T& Value, const T& A1, const T& B1, const T& A2, const T& B2)
	{
		return Lerp(A2, B2, InverseLerp(A1, B1, Value));
	}

	template<typename T>
	inline T Clamp(const T& Value, const T& Min, const T& Max)
	{
		return Value < Min ? Min : (Value > Max ? Max : Value);
	}

	template<typename T>
	inline T Sign(const T& Value)
	{
		return Value < 0 ? -1 : (Value > 0 ? 1 : 0);
	}

	template<typename T>
	inline T Min(const T& A, const T& B)
	{
		return A < B ? A : B;
	}

	template<typename T, typename... Args>
	inline T Min(const T& A, const T& B, const Args&... args)
	{
		return Min(Min(A, B), args...);
	}

	template<typename T>
	inline T Max(const T& A, const T& B)
	{
		return (A > B) ? A : B;
	}

	template<typename T, typename... Args>
	inline T Max(const T& A, const T& B, const Args&... args)
	{
		return Max(Max(A, B), args...);
	}

	template<typename T>
	inline T Sum(const T& A, const T& B)
	{
		return A + B;
	}

	template<typename T, typename... Args>
	inline T Sum(const T& A, const T& B, const Args&... args)
	{
		return Sum(Sum(A, B), args...);
	}

	template<typename T>
	inline T Average(const T& A, const T& B)
	{
		return (A + B) / 2;
	}

	template<typename T, typename... Args>
	inline T Average(const T& A, const T& B, const Args&... args)
	{
		return Average(Average(A, B), args...);
	}

	template<typename T>
	inline T Median(const T& A, const T& B)
	{
		return (A + B) / 2;
	}

	template<typename T, typename... Args>
	inline T Median(const T& A, const T& B, const Args&... args)
	{
		return Median(Median(A, B), args...);
	}

	template<typename T>
	inline T Variance(const T& A, const T& B)
	{
		T Mean = Average(A, B);
		return Average((A - Mean) * (A - Mean), (B - Mean) * (B - Mean));
	}

	template<typename T, typename... Args>
	inline T Variance(const T& A, const T& B, const Args&... args)
	{
		return Variance(Variance(A, B), args...);
	}

	template<typename T>
	inline T Random(const T& Min, const T& Max)
	{
		return Min + (Max - Min) * double(rand()) / RAND_MAX;
	}

	template<typename T>
	inline T Random(const T& Max)
	{
		return Random(T(0), Max);
	}

	template<typename T>
	inline T Random()
	{
		return Random(std::numeric_limits<T>::max());
	}

	template<typename T>
	inline T RandomSign()
	{
		return rand() % 2 == 0 ? T(1) : T(-1);
	}

	template<typename T>
	inline T RandomBool()
	{
		return rand() % 2 == 0;
	}

	template<typename T>
	inline T RandomNormal(const T& Mean, const T& StdDev)
	{
		double U1 = Random(0.0, 1.0);
		double U2 = Random(0.0, 1.0);
		double Z0 = sqrt(-2.0 * log(U1)) * cos(Tau * U2);
		return Mean + Z0 * StdDev;
	}

	template<typename T>
	inline T Pow(const T& Base, const T& Exponent)
	{
		return pow(Base, Exponent);
	}

	template<typename T>
	inline T Log(const T& Value, const T& Base)
	{
		return log(Value) / log(Base);
	}

	template<typename T>
	inline T Floor(const T& Value)
	{
		return floor(Value);
	}

	template<typename T>
	inline T Ceiling(const T& Value)
	{
		return ceil(Value);
	}

	template<typename T>
	inline T Round(const T& Value)
	{
		return round(Value);
	}

} // namespace Math
} // namespace NEAT