#pragma once

#include <cstddef>
#include <ostream>

class Array
{
protected:
	size_t _n;
	double* _p;
public:
	Array();
	Array(double* initArray, size_t N);
	Array(const Array& other);
	Array(int N);
	
	double& operator[] (size_t i) const;
	size_t size() const;

	Array& operator= (Array& other);

	friend std::ostream& operator<<(std::ostream& os, const Array& arr);

	virtual ~Array();
};
