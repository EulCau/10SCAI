#include <torch/torch.h>
#include <iostream>
#include <vector>
#include "rfm.h"

using namespace std;

int main()
{
	if (torch::cuda::is_available())
	{
	   cout << 1 << endl;
	}
	else
	{
		cout << 0 << endl;
	}

	vector<double> alpha(0.0, 3);

	alpha.push_back(1.0);

	torch::Dtype a = torch::tensor(0.0).scalar_type();
	auto b = torch::tensor(0.0).device();

	cout << a << b << endl;

	return 0;
}
