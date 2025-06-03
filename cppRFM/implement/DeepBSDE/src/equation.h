#pragma once

#include "config.h"
#include <cmath>
#include <memory>
#include <torch/torch.h>
#include <vector>

class Equation
{
public:
	Equation(const EqnConfig& config)
		: dim_(config.dim),
		  total_time_(config.total_time),
		  num_time_interval_(config.num_time_interval),
		  delta_t_(config.total_time / config.num_time_interval),
		  sqrt_delta_t_(std::sqrt(delta_t_)) {}

	virtual ~Equation() = default;

	virtual std::pair<torch::Tensor, torch::Tensor> sample(int64_t num_sample) const = 0;

	virtual torch::Tensor f(const torch::Tensor& t, const torch::Tensor& x, const torch::Tensor& y, const torch::Tensor& z) const = 0;

	virtual torch::Tensor g(const torch::Tensor& t, const torch::Tensor& x) const = 0;

	int dim() const { return dim_; }
	double total_time() const { return total_time_; }
	int num_time_interval() const { return num_time_interval_; }
	double delta_t() const { return delta_t_; }
	double sqrt_delta_t() const { return sqrt_delta_t_; }

protected:
	int dim_;
	double total_time_;
	int num_time_interval_;
	double delta_t_;
	double sqrt_delta_t_;
};
