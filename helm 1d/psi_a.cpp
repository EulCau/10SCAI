#include "helm-1d.h"

namespace aid = at::indexing;

psiaModel::psiaModel(int inFeatures, int hiddenFeatures, double xMin, double xMax):
	rfmModel(inFeatures, hiddenFeatures, xMin, xMax)
{
	return;
}

torch::Tensor psiaModel::forward(torch::Tensor x)
{
	return _hiddenLayer->forward(_zoom * (x - _xCenter));
}

psiaModel::~psiaModel()
{
	return;
}

std::pair<torch::Tensor, torch::Tensor> calculateMatrixA(
	const std::vector<std::shared_ptr<rfmModel>>& models,
	int Q, double intervalStart, double totalLength,
	double bound0, double bound1,
	torch::Tensor(*fReal)(const torch::Tensor& points),
	torch::Tensor(*fCalculate)(const torch::Tensor&, const torch::Tensor&))
{
	int Mp = models.size(), Jn = models[0]->hiddenFeatures();

	torch::Tensor f = torch::zeros({ (Q + 2) * Mp, 1 }, torch::kFloat64);

	torch::Tensor A1 = torch::zeros({ Q * Mp, Jn * Mp }, torch::kFloat64);
	torch::Tensor A2 = torch::zeros({ 2, Jn * Mp }, torch::kFloat64);
	torch::Tensor A3 = torch::zeros({ Mp - 1, Jn * Mp }, torch::kFloat64);
	torch::Tensor A4 = torch::zeros({ Mp - 1, Jn * Mp }, torch::kFloat64);

	torch::Tensor A;

	for (size_t i = 0; i < Mp; i++)
	{
		torch::Tensor points = torch::linspace(
			totalLength / Mp * i + intervalStart
			, totalLength / Mp * (i + 1) + intervalStart,
			size_t(Q) + 1,
			torch::TensorOptions().dtype(torch::kFloat64).requires_grad(true)
		).reshape({ -1, 1 });

		torch::Tensor out = models[i]->forward(points);

		std::vector<torch::Tensor> gradsOutVector;
		std::vector<torch::Tensor> grad2sOutVector;
		for (size_t j = 0; j < Jn; j++)
		{
			torch::Tensor gradOut = torch::autograd::grad(
				{ out.index({aid::Slice::Slice(), int(j)}) }, { points },
				{ torch::ones_like(out.index({aid::Slice::Slice(), int(j)})) },
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
			aid::Slice::Slice(i * Jn, i * Jn + Jn) },
			Lout.index({ aid::Slice::Slice(0, Q), aid::Slice::Slice() }));
		f.index_put_(
			{ aid::Slice::Slice(i * Q, (i + 1) * Q), 0 },
			fReal(points.index({ aid::Slice::Slice(0, Q), 0 }).detach()));
		auto value_l = out.index({ 0, aid::Slice::Slice() });
		auto value_r = out.index({ -1, aid::Slice::Slice() });
		auto grad_l = gradsOut.index({ 0, aid::Slice::Slice() });
		auto grad_r = gradsOut.index({ -1, aid::Slice::Slice() });

		if (Mp > 1)
		{
			if (i == 0)
			{
				A2.index_put_({ 0, aid::Slice::Slice(0, Jn) }, value_l);
				A3.index_put_({ 0, aid::Slice::Slice(0, Jn) }, -value_r);
				A4.index_put_({ 0, aid::Slice::Slice(0, Jn) }, -grad_r);
			}
			else if (i == size_t(Mp) - 1)
			{
				A2.index_put_(
					{ 1, aid::Slice::Slice(Mp * Jn - Jn, Mp * Jn) }, value_r);
				A3.index_put_(
					{ Mp - 2, aid::Slice::Slice(Mp * Jn - Jn, Mp * Jn) }, value_l);
				A4.index_put_(
					{ Mp - 2, aid::Slice::Slice(Mp * Jn - Jn, Mp * Jn) }, grad_l);
			}
			else
			{
				A3.index_put_(
					{ int(i) - 1, aid::Slice::Slice(i * Jn, i * Jn + Jn) }, value_l);
				A4.index_put_(
					{ int(i) - 1, aid::Slice::Slice(i * Jn, i * Jn + Jn) }, grad_l);
				A3.index_put_(
					{ int(i), aid::Slice::Slice(i * Jn, i * Jn + Jn) }, -value_r);
				A4.index_put_(
					{ int(i), aid::Slice::Slice(i * Jn, i * Jn + Jn) }, -grad_r);
			}
		}
		else
		{
			A2.index_put_({ 0, aid::Slice::Slice(0, Jn) }, value_l);
			A2.index_put_({ 1, aid::Slice::Slice(-Jn, 0) }, value_r);
		}
	}

	A = torch::cat({ A1 , A2, A3, A4 }, 0);
	f.index_put_({ Mp * Q, 0 }, bound0);
	f.index_put_({ Mp * Q + 1, 0 }, bound1);

	return std::pair<torch::Tensor, torch::Tensor>(A, f);
}
