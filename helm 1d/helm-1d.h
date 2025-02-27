#pragma once

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include "torch/torch.h"

class rfmModel : public torch::nn::Module
{
public:
	rfmModel(int inFeatures, int hiddenFeatures, double xMin, double xMax);
	virtual torch::Tensor forward(torch::Tensor x) = 0;
	int inFeatures() const;
	int hiddenFeatures() const;
	virtual ~rfmModel();

protected:
	int _inFeatures;
	int _hiddenFeatures;
	double _zoom, _xCenter;
	torch::nn::Sequential _hiddenLayer;
};

std::vector<std::shared_ptr<rfmModel>> preDefine(
	int Mp, int Jn, double Rm, int option, double intervalStart, double totalLength);

std::pair<double, double> test(
	const std::vector<std::shared_ptr<rfmModel>>& models,
	int Q, int opt, double intervalStart, double totalLength,
	const torch::Tensor& w, double (*uReal)(double));

class psiaModel : public rfmModel
{
public:
	psiaModel(int inFeatures, int hiddenFeatures, double xMin, double xMax);
	torch::Tensor forward(torch::Tensor x);
	~psiaModel();
};

class psibModel : public rfmModel
{
public:
	psibModel(int inFeatures, int hiddenFeatures,
		double xMin, double xMax,
		double intervalStart, double intervalLength);
	torch::Tensor forward(torch::Tensor x);
	~psibModel();
private:
	double _xMin, _xMax;
	double _intervalStart, _intervalLength;
};

std::pair<torch::Tensor, torch::Tensor> calculateMatrixA(
	const std::vector<std::shared_ptr<rfmModel>>& models,
	int Q, double intervalStart, double totalLength,
	double bound0, double bound1,
	torch::Tensor(*fReal)(const torch::Tensor& points),
	torch::Tensor(*fCalculate)(const torch::Tensor&, const torch::Tensor&));

std::pair<torch::Tensor, torch::Tensor> calculateMatrixB(
	const std::vector<std::shared_ptr<rfmModel>>& models,
	int Q, double intervalStart, double totalLength,
	double bound0, double bound1,
	torch::Tensor(*fReal)(const torch::Tensor& points),
	torch::Tensor(*fCalculate)(const torch::Tensor&, const torch::Tensor&));
