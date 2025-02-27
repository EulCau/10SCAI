#pragma once

#include <optional>
#include <iostream>
#include <vector>
#include <unordered_map>
#include "torch/torch.h"

namespace rfm
{
	enum class State
	{
		isIn = 0,
		isOut = 1,
		isOn = 2,
		isUnknown = 3
	};

	class GeometryBase
	{
	public:
		GeometryBase(std::optional<int>, std::optional<int>);
		virtual ~GeometryBase();
		bool operator==(const GeometryBase&);
		virtual torch::Tensor sdf(const torch::Tensor& p) const = 0;
		virtual std::vector<float> getBoundingBox() const = 0;
		virtual torch::Tensor inSample(int numSamples, bool withBoundary = false) const = 0;
		virtual torch::Tensor onSample(int numSamples, bool withNormal = false) const = 0;

	public:
		int dim() const;
		torch::Dtype dtype() const;
		torch::Device device() const;
		int intrinsicDim() const;
		std::vector<std::shared_ptr<GeometryBase>> boundary() const;

	protected:
		int _dim;
		torch::Dtype _dtype;
		torch::Device _device;
		int _intrinsicDim;
		std::vector<std::shared_ptr<GeometryBase>> _boundary;

	private:
		bool iseqbd(
			std::vector<std::shared_ptr<GeometryBase>>,
			std::vector<std::shared_ptr<GeometryBase>>);
	};

	class UnionGeometry : public GeometryBase
	{
	public:
		UnionGeometry(GeometryBase&, GeometryBase&);
		virtual ~UnionGeometry();
		torch::Tensor sdf(const torch::Tensor&);
		std::vector<float> getBoundingBox();
		torch::Tensor inSample(int numSamples, bool withBoundary = false);
		torch::Tensor onSample(int numSamples, bool withNormal = false);

	protected:
		GeometryBase& _geomA;
		GeometryBase& _geomB;

	private:

	};
}
