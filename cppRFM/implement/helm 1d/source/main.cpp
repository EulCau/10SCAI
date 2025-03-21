#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include "torch/torch.h"
#include "helm-1d.h"
#include <windows.h>

using namespace std;

constexpr double
A = 1.0,
a = 2.0 * M_PI,
b = 3.0 * M_PI,
c = 0.05,
d = 2.0,
lambda = 4.0;

constexpr double
intervalStart = 0.0,
intervalLength = 8.0;

static double uReal(double x)
{
	return
		A * cos(a * (x + c)) * sin(b * (x + c)) + d;
}

static double d2uReal(double x)
{
	return
		-A * (a * a + b * b) * cos(a * (x + c)) * sin(b * (x + c))
		- 2 * A * a * b * sin(a * (x + c)) * cos(b * (x + c));
}

static torch::Tensor fReal(const torch::Tensor& points)
{
	torch::Tensor f = torch::empty_like(points);
	size_t n = points.size(0);
	for (size_t i = 0; i < n; i++)
	{
		double x = points[i].item<double>();
		f[i] = d2uReal(x) - lambda * uReal(x);
	}
	return f;
}

static torch::Tensor fCalculate(
	const torch::Tensor& out, const torch::Tensor& grad2Out)
{
	return grad2Out - lambda * out;
}

int main()
{
	for (int Q : {60, 80, 100, 120, 140, 160})
	{
		int Jn = 50, Mp = 6;
		double Rm = 3.0;
		int opt = 1;
		vector<shared_ptr<rfmModel>> models = preDefine(
			Mp, Jn, Rm, opt, intervalStart, intervalLength);

		torch::Tensor A, f;
		tie(A, f) = calculateMatrixA(
			models, Q, intervalStart, intervalLength,
			uReal(intervalStart), uReal(intervalStart + intervalLength),
			&fReal, &fCalculate);

		torch::Tensor w = std::get<0>(torch::linalg::lstsq(A, f, nullopt, "gelsd"));

		torch::Tensor finalL2 = sqrt(at::mean(at::pow(mm(A, w) - f, 2)));

		double errorLinf = 0.0, errorL2 = 0.0;
		tie(errorLinf, errorL2) = test(
			models, Q, opt, intervalStart, intervalLength, w, &uReal);

		std::cout << "******************** ERROR ********************" << std::endl;
		cout<< "PoU = psi_a, " << " Rm = " << Rm << ", Mp = " << Mp
			<< ", Jn = " << Jn << ", Q = " << Q << ";" << endl
			<< "Final loss: " << finalL2.item<double>() << ";" << endl
			<< "L_inf = " << errorLinf << ", L_2 = " << errorL2 << "." << endl;
	}

	for (int Mp : {6, 8, 10, 12, 14, 50})
	{
		int Jn = 50, Q = 50;
		double Rm = 1.0;
		int opt = 2;

		DWORD startTime = GetTickCount();

		vector<shared_ptr<rfmModel>> models = preDefine(
			Mp, Jn, Rm, opt, intervalStart, intervalLength);

		torch::Tensor A, f;
		tie(A, f) = calculateMatrixB(
			models, Q, intervalStart, intervalLength,
			uReal(intervalStart), uReal(intervalStart + intervalLength),
			&fReal, &fCalculate);

		torch::Tensor w = std::get<0>(torch::linalg::lstsq(A, f, nullopt, "gelsd"));

		torch::Tensor finalL2 = sqrt(at::mean(at::pow(mm(A, w) - f, 2)));

		double errorLinf = 0.0, errorL2 = 0.0;
		tie(errorLinf, errorL2) = test(
			models, Q, opt, intervalStart, intervalLength, w, &uReal);

		DWORD endTime = GetTickCount();

		std::cout << "******************** ERROR ********************" << std::endl;
		cout << "PoU = psi_b, " << " Rm = " << Rm << ", Mp = " << Mp;
		cout << ", Jn = " << Jn << ", Q = " << Q << ";" << endl;
		cout << "Final loss: " << finalL2.item<double>() << ";" << endl;
		cout << "L_inf = " << errorLinf << ", L_2 = " << errorL2 << "." << endl;
		cout << (endTime - startTime) / 1000. << endl;
	}

	return 0;
}
