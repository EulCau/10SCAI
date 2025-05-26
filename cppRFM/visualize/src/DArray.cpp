#include "DArray.h"

Array::Array(double* initArray, size_t N):
_n(N)
{
	_p = new double[_n];
	for (size_t i = 0; i < _n; i++)
	{
		_p[i] = initArray[i];
	}
	
}

Array::Array():
_n(0)
{
	_p = nullptr;
}

Array::Array(const Array& other) : _n(other.size())
{
	_p = new double[_n];
	for (size_t i = 0; i < _n; i++)
	{
		_p[i] = other[i];
	}
	return;
}

Array::Array(int N):
_n(N)
{
	_p = new double[_n];
	return;
}

double &Array::operator[](size_t i) const
{
	return _p[i];
}

size_t Array::size() const
{
	return _n;
}

Array& Array::operator=(Array& other)
{
	if (this != &other)
	{
		delete[] _p;
		_n = other.size();
		_p = new double[_n];
		for (size_t i = 0; i < _n; i++)
		{
			_p[i] = other[i];
		}
	}
	return *this;
}

Array::~Array()
{
	delete[] _p;
	return;
}

std::ostream &operator<<(std::ostream &os, const Array &arr)
{
    os << "Array: ";
    for (size_t i = 0; i < arr.size(); i++)
	{
        os << arr[i] << " ";
    }
	os << "size: " << arr.size();
    return os;
}
