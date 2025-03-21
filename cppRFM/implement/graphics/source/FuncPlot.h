#pragma once

#include <graphics.h>
#include "torch/torch.h"

class graphic
{
public:
	graphic();
	~graphic();

protected:
	int _dim;
	bool _needXAxis;
	bool _needYAxis;
	bool _needZAxis;

	at::Tensor _xPoints;
	at::Tensor _yPoints;
	at::Tensor _zPoints;

};
