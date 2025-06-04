#include "equation_factory.h"
#include "config.h"
#include <iostream>

int main()
{
    Config cfg = load_config("hjb_lq_d100.json");

    try
    {
        auto eqn = EquationFactory::instance().create(cfg.eqn_config.eqn_name, cfg.eqn_config);
        auto [dw, x] = eqn->sample(2);
        std::cout << "Sample success. Shape of x: " << x.sizes() << std::endl;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to create equation: " << e.what() << std::endl;
    }

    return 0;
}
