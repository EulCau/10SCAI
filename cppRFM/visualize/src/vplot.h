#pragma once

#include <vector>
#include "torch/torch.h"

namespace visualizePlot
{
	class PlotBC
	{
	public:
		PlotBC();
		unsigned dimInVar() const;
		unsigned dimDeVar() const;
		at::Tensor lowerInVar() const;
		at::Tensor upperInVar() const;
		at::Tensor stepInVar() const;
		virtual void show() = 0;
		virtual ~PlotBC();

	protected:
		unsigned _dimInVar;
		unsigned _dimDeVar;
		at::Tensor _lowerInVar, _upperInVar, _stepInVar;
		at::Tensor(*_map)(at::Tensor);

	};

	void show1to1(at::Tensor lowerInVar, at::Tensor upperInVar, at::Tensor(*_map)(at::Tensor));

	void show1to2(at::Tensor lowerInVar, at::Tensor upperInVar, at::Tensor(*_map)(at::Tensor));

	void show2to1(at::Tensor lowerInVar, at::Tensor upperInVar, at::Tensor(*_map)(at::Tensor));
}
