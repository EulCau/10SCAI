#include "vplot.h"

using namespace visualizePlot;

PlotBC::PlotBC():
	_dimDeVar(0), _dimInVar(0),
	_lowerInVar(at::zeros({})), _upperInVar(at::zeros({})), _stepInVar(at::zeros({})),
	_map(nullptr)
{
	return;
}

unsigned visualizePlot::PlotBC::dimInVar() const
{
	return _dimInVar;
}

unsigned visualizePlot::PlotBC::dimDeVar() const
{
	return _dimDeVar;
}

at::Tensor visualizePlot::PlotBC::lowerInVar() const
{
	return _lowerInVar;
}

at::Tensor visualizePlot::PlotBC::upperInVar() const
{
	return _upperInVar;
}

at::Tensor visualizePlot::PlotBC::stepInVar() const
{
	return _stepInVar;
}

PlotBC::~PlotBC()
{
	return;
}

void visualizePlot::show1to1(at::Tensor lowerInVar, at::Tensor upperInVar, at::Tensor(*_map)(at::Tensor))
{
	return;
}

void visualizePlot::show1to2(at::Tensor lowerInVar, at::Tensor upperInVar, at::Tensor(*_map)(at::Tensor))
{
	return;
}

void visualizePlot::show2to1(at::Tensor lowerInVar, at::Tensor upperInVar, at::Tensor(*_map)(at::Tensor))
{
	return;
}
