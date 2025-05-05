#pragma once
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <cstdint>
#include "Tensor.hpp"


struct Pack {
    std::vector<Tensor*> train_label;
    std::vector<Tensor*> train_data;
};

class MnistCsvReader {
public:
    static Pack* readCsv(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file: " + filename);
        }

        std::vector<Tensor*> dataset;
        std::string line;
        Pack* result = new Pack();
        std::getline(file, line);
        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string value;


            std::getline(ss, value, ',');
            Tensor* train_label = new Tensor(vector<int>{10});

            train_label->at(static_cast<float>(std::stoi(value))) = 1;

            Tensor* train_data = new Tensor(vector<int>{784});
            int i = 0;
            while (std::getline(ss, value, ',')) {
                train_data->at(i) = static_cast<float>(std::stoi(value)) / 255.0;
                i++;
            }

            result->train_label.push_back(train_label);
            result->train_data.push_back(train_data);
        }

        file.close();
        return result;
    }
};