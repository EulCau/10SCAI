#include "helm-1d.h"

namespace aid = at::indexing;

rfmModel::rfmModel(int inFeatures, int hiddenFeatures, double xMin, double xMax) :
	_inFeatures(inFeatures), _hiddenFeatures(hiddenFeatures),
	_zoom(2.0 / (xMax - xMin)), _xCenter((xMax + xMin) / 2),
	Module(), _hiddenLayer(torch::nn::Sequential(
		torch::nn::Linear(_inFeatures, _hiddenFeatures),
		torch::nn::Tanh()))
{
	_hiddenLayer->to(torch::kFloat64);
	register_module("hidden_layer", _hiddenLayer);
	return;
}

int rfmModel::inFeatures() const
{
	return _inFeatures;
}

int rfmModel::hiddenFeatures() const
{
	return _hiddenFeatures;
}

rfmModel::~rfmModel()
{
	return;
}

static void weights_init(torch::nn::Module& model, double range)
{
	if (auto* linear = dynamic_cast<torch::nn::LinearImpl*>(&model))
	{
		linear->weight.data().uniform_(-range, range);
		linear->bias.data().uniform_(-range, range);
	}
	return;
}

std::vector<std::shared_ptr<rfmModel>> preDefine(
	int Mp, int Jn, double Rm, int option,
	double intervalStart, double totalLength)
{
	std::vector<std::shared_ptr<rfmModel>> models;

	for (size_t i = 0; i < Mp; i++)
	{
		double xMin = totalLength / Mp * i + intervalStart;
		double xMax = totalLength / Mp * (i + 1) + intervalStart;
		std::shared_ptr<rfmModel> model;

		if (option & 1)
		{
			model = std::make_shared<psiaModel>(1, Jn, xMin, xMax);
		}
		else
		{
			model = std::make_shared<psibModel>(
				1, Jn, xMin, xMax, intervalStart, totalLength);
		}
		weights_init(*model, Rm);
		for (torch::Tensor& param : model->parameters())
		{
			param.requires_grad_(false);
		}

		models.push_back(model);
	}
	return models;
}

std::pair<double, double> test(
	const std::vector<std::shared_ptr<rfmModel>>& models,
	int Q, int opt, double intervalStart, double totalLength,
	const torch::Tensor& w, double (*uReal)(double))
{
	int Mp = models.size(), Jn = models[0]->hiddenFeatures();
	double errorLinf = 0.0;
	double errorL2 = 0.0;

	int testQ = int(1000 / Mp) + 1;

	for (size_t i = 0; i < Mp; i++)
	{
		double xMin = totalLength / Mp * i + intervalStart;
		double xMax = totalLength / Mp * (i + 1) + intervalStart;
		torch::Tensor points = torch::linspace(
			xMin, xMax, testQ).reshape({ -1, 1 }).to(torch::kFloat64);

		torch::Tensor out;
		torch::Tensor values;
		torch::Tensor uNumerical;
		if (opt & 1)
		{
			out = models[i]->forward(points);
			values = out.detach();
			uNumerical = torch::mm(
				values,
				w.index({ aid::Slice::Slice(i * Jn, (i + 1) * Jn),
					aid::Slice::Slice() }));
		}
		else
		{
			for (size_t j = 0; j < Mp; j++)
			{
				out = models[j]->forward(points);
				if (values.defined())
				{
					values = torch::cat({ values, out.detach()}, 1);
				}
				else
				{
					values = out.detach().clone();
				}
			}
			uNumerical = torch::mm(values, w);
		}

		std::vector<double> uRealVector(testQ);
		for (size_t j = 0; j < testQ; j++)
		{
			uRealVector[j] = uReal(points[j].item<double>());
		}
		torch::Tensor uReal = torch::tensor(
			uRealVector, torch::kFloat64).reshape({ -1, 1 });

		auto epsilon = (uReal - uNumerical).abs();
		errorLinf = std::max(errorLinf, epsilon.max().item<double>());
		errorL2 += epsilon.pow(2).sum().item<double>();
	}
	errorL2 = std::sqrt(totalLength * errorL2 / (Mp * testQ));

	return std::pair<double, double>(errorLinf, errorL2);
}
