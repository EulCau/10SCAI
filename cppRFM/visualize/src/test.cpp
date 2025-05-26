#include <torch/torch.h>
#include <iostream>
#include <vector>

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

    return 0;
}
