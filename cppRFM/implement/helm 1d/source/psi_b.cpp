#include "helm-1d.h"

namespace aid = at::indexing;

psibModel::psibModel(int inFeatures, int hiddenFeatures,
    double xMin, double xMax,
    double intervalStart, double intervalLength) :
	rfmModel(inFeatures, hiddenFeatures, xMin, xMax),
    _xMin(xMin), _xMax(xMax), _intervalStart(intervalStart),
    _intervalLength(intervalLength)
{
	return;
}

torch::Tensor psibModel::forward(torch::Tensor x)
{
    torch::Tensor d = (x - _xMin) / (_xMax - _xMin);

    torch::Tensor d0 = d <= -0.25;
    torch::Tensor d1 = (d <= 0.25) & (d > -0.25);
    torch::Tensor d2 = (d <= 0.75) & (d > 0.25);
    torch::Tensor d3 = (d <= 1.25) & (d > 0.75);
    torch::Tensor d4 = d > 1.25;

    torch::Tensor y = _hiddenLayer->forward(_zoom * (x - _xCenter));

    torch::Tensor y0 = torch::zeros_like(y);
    torch::Tensor y1 = y * (1 + torch::sin(2 * M_PI * d)) / 2;
    torch::Tensor y2 = y.clone();
    torch::Tensor y3 = y * (1 - torch::sin(2 * M_PI * (d - 1))) / 2;
    torch::Tensor y4 = torch::zeros_like(y);

    if (abs(_xMin - _intervalStart) < (0.1 / _zoom))
    {
        return d0 * y0 + (d1 + d2) * y2 + d3 * y3 + d4 * y4;
    }
	else if (abs(_xMax - _intervalStart - _intervalLength) < (0.1 / _zoom))
    {
        return d0 * y0 + d1 * y1 + (d2 + d3) * y2 + d4 * y4;
    }
    else
    {
        return d0 * y0 + d1 * y1 + d2 * y2 + d3 * y3 + d4 * y4;
    }
}

psibModel::~psibModel()
{
	return;
}

std::pair<torch::Tensor, torch::Tensor> calculateMatrixB(
	const std::vector<std::shared_ptr<rfmModel>>& models,
	int Q, double intervalStart, double totalLength,
	double bound0, double bound1,
	torch::Tensor(*fReal)(const torch::Tensor& points),
	torch::Tensor(*fCalculate)(const torch::Tensor&, const torch::Tensor&))
{
	int Mp = models.size(), Jn = models[0]->hiddenFeatures();

	torch::Tensor f = torch::zeros({ Q * Mp + 2, 1 }, torch::kFloat64);

	torch::Tensor A1 = torch::zeros({ Q * Mp, Jn * Mp }, torch::kFloat64);
	torch::Tensor A2 = torch::zeros({ 2, Jn * Mp }, torch::kFloat64);

	torch::Tensor A;

	for (size_t i = 0; i < Mp; i++)
	{
		torch::Tensor points = torch::linspace(
			totalLength / Mp * i + intervalStart
			, totalLength / Mp * (i + 1) + intervalStart,
			size_t(Q) + 1,
			torch::TensorOptions().dtype(torch::kFloat64).requires_grad(true)
		).reshape({ -1, 1 });

		for (size_t j = 0; j < Mp; j++)
		{
			torch::Tensor out = models[j]->forward(points);

			std::vector<torch::Tensor> gradsOutVector;
			std::vector<torch::Tensor> grad2sOutVector;
			for (size_t k = 0; k < Jn; k++)
			{
				torch::Tensor gradOut = torch::autograd::grad(
					{ out.index({aid::Slice::Slice(), int(k)}) }, { points },
					{ torch::ones_like(out.index({aid::Slice::Slice(), int(k)})) },
					true, true)[0];
				gradsOutVector.push_back(gradOut);

				torch::Tensor grad2Out = torch::autograd::grad(
					{ gradOut.index({ aid::Slice::Slice(), 0 }) }, { points },
					{ torch::ones_like(gradOut.index({ aid::Slice::Slice(), 0 })) },
					true, false)[0];
				grad2sOutVector.push_back(grad2Out);
			}

			for (auto& tensor : gradsOutVector)
			{
				tensor = tensor.squeeze(1);
			}
			for (auto& tensor : grad2sOutVector)
			{
				tensor = tensor.squeeze(1);
			}
			torch::Tensor gradsOut = torch::stack(gradsOutVector, 1);
			torch::Tensor grad2sOut = torch::stack(grad2sOutVector, 1);

			torch::Tensor Lout = fCalculate(out, grad2sOut);

			A1.index_put_(
				{ aid::Slice::Slice(i * Q, i * Q + Q),
				aid::Slice::Slice(j * Jn, j * Jn + Jn) },
				Lout.index({ aid::Slice::Slice(0, Q), aid::Slice::Slice() }));

			auto value_l = out.index({ 0, aid::Slice::Slice() });
			auto value_r = out.index({ -1, aid::Slice::Slice() });

			if (i == 0 && j == i)
			{
				A2.index_put_({ 0, aid::Slice::Slice(0, Jn) }, value_l);
			}
			else if (i == size_t(Mp)-1 && i == j)
			{
				A2.index_put_(
					{ 1, aid::Slice::Slice(Mp * Jn - Jn, Mp * Jn) }, value_r);
			}
		}

		torch::Tensor _fReal = fReal(points.detach()).reshape({ -1, 1 });

		f.index_put_(
			{ aid::Slice::Slice(i * Q, (i + 1) * Q), 0 },
			_fReal.index({ aid::Slice::Slice(0, Q), 0 }));
	}

	A = torch::cat({ A1 , A2 }, 0);
	f.index_put_({ Mp * Q, 0 }, bound0);
	f.index_put_({ Mp * Q + 1, 0 }, bound1);

	return std::pair<torch::Tensor, torch::Tensor>(A, f);
}
