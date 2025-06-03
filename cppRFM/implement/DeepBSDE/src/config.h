#pragma once

#include "json.hpp"
#include <string>
#include <vector>

struct EqnConfig
{
    std::string eqn_name;
    double total_time;
    int dim;
    int num_time_interval;
};

struct NetConfig
{
    std::vector<double> y_init_range;
    std::vector<int> num_hiddens;
    std::vector<double> lr_values;
    std::vector<int> lr_boundaries;
    int num_iterations;
    int batch_size;
    int valid_size;
    std::string dtype;
    bool verbose;
    int logging_frequency;
};

struct Config
{
    EqnConfig eqn_config;
    NetConfig net_config;
};

Config load_config(const std::string& json_path);
