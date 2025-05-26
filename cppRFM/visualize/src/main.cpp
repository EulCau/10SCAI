#include <iostream>
#include "DArray.h"

using namespace std;

int main()
{
	double a[5] = {0., 1., 2., 3., 4.};
	Array test0(a, 5), test1, test2 = test0;
	test2 = test1;
	test1 = test0;

	test0[1] = 0.;

	cout << test0 << endl;
	cout << test1 << endl;
	cout << test2 << endl;

	for (size_t i = 0; i < test2.size(); i++)
	{
		test1[i] = i * (i + 1.);
	}

	cout << test1 << endl;
	
	getchar();
	
	return 0;
}
