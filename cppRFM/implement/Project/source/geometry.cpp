#include "geometry.h"

using namespace rfm;

GeometryBase::GeometryBase(
	std::optional<int> dim = std::nullopt,
	std::optional<int> intrinsicDim = std::nullopt):
	_dim(dim.has_value() ? dim.value() : 0),
	_dtype(torch::tensor(0.0).scalar_type()),
	_device(torch::tensor(0.0).device()),
	_intrinsicDim(intrinsicDim.has_value() ? intrinsicDim.value() : 0)
{
	return;
}

GeometryBase::~GeometryBase()
{
	std::cout << "A geometry is deleted." << std::endl;
	return;
}

bool rfm::GeometryBase::operator==(const GeometryBase& other)
{
	return
		typeid(*this) == typeid(other) &&
		_dim == other._dim &&
		iseqbd(_boundary, other._boundary);
}

int GeometryBase::dim() const
{
	return _dim;
}

torch::Dtype GeometryBase::dtype() const
{
	return _dtype;
}

torch::Device rfm::GeometryBase::device() const
{
	return _device;
}

int rfm::GeometryBase::intrinsicDim() const
{
	return _intrinsicDim;
}

std::vector<std::shared_ptr<GeometryBase>> rfm::GeometryBase::boundary() const
{
	return _boundary;
}

bool rfm::GeometryBase::iseqbd
(
	std::vector<std::shared_ptr<GeometryBase>> bd1,
	std::vector<std::shared_ptr<GeometryBase>> bd2
)
{
	int n = bd1.size();
	if (n != bd2.size())
	{
		return false;
	}

	std::unordered_map<std::shared_ptr<GeometryBase>, int> count1, count2;
	for (size_t i = 0; i < n; i++)
	{
		count1[bd1[i]]++;
		count2[bd2[i]]++;
	}
	for (const auto& [key, value] : count1)
	{
		if (count2[key] != value)
		{
			return false;
		}
	}

	return true;
}

UnionGeometry::UnionGeometry(GeometryBase& geomA, GeometryBase& geomB) :
	_geomA(geomA), _geomB(geomB),
	GeometryBase(geomA.dim(), geomA.intrinsicDim())
{
	if (geomA.dim() != geomB.dim() || geomA.intrinsicDim() != geomB.intrinsicDim())
	{
		std::cout << "\033[31mUnion: different dimensions.\033[0m" << std::endl;
	}
	std::vector<std::shared_ptr<GeometryBase>>
		bdA = geomA.boundary(), bdB = geomB.boundary();
	_boundary.insert(_boundary.end(), bdA.begin(), bdA.end());
	_boundary.insert(_boundary.end(), bdB.begin(), bdB.end());
	return;
}

torch::Tensor UnionGeometry::sdf(const torch::Tensor& p)
{
	return torch::min(_geomA.sdf(p), _geomB.sdf(p));
}

std::vector<float> UnionGeometry::getBoundingBox()
{
	std::vector<float>
		boxA = _geomA.getBoundingBox(), boxB = _geomB.getBoundingBox(), box;
	for (size_t i = 0; i < size_t(2) * _dim; i++)
	{
		box.push_back(
			i % 2 == 0 ? std::min(boxA[i], boxB[i]) : std::max(boxA[i], boxB[i]));
	}
}

torch::Tensor rfm::UnionGeometry::inSample(int numSamples, bool withBoundary)
{
	return torch::Tensor();
}

torch::Tensor rfm::UnionGeometry::onSample(int numSamples, bool withNormal)
{
	return torch::Tensor();
}

UnionGeometry::~UnionGeometry()
{
	std::cout << "A union of geometry is deleted" << std::endl;
}
