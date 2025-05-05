#include <iostream>
#include "thread_pool.hpp"
#include "Tensor.hpp"
#include "model.hpp"
#include "serialize.hpp"
#include "mnist_csv_reader_csv.hpp"
int main() {

    /*
     Architecture 784 DENSE RELU 128 DENSE RELU 64 DENSE RELU 10 SOFTMAX 10
     */
    MnistCsvReader reader = MnistCsvReader();
    Pack* data = reader.readCsv("mnist_train.csv");
    Pack* data_test = reader.readCsv("mnist_test.csv");

    /*Shared resources*/
    Tensor layer1_weight(vector<int>{784, 128});
    layer1_weight.HeInitilize();
    Tensor layer1_bias(vector<int>{128});
    layer1_bias.ZeroInitilize();

    Tensor layer2_weight(vector<int>{128, 64});
    layer2_weight.HeInitilize();
    Tensor layer2_bias(vector<int>{64});
    layer2_bias.ZeroInitilize();

    Tensor layer3_weight(vector<int>{64, 10});
    layer3_weight.HeInitilize();
    Tensor layer3_bias(vector<int>{10});
    layer3_bias.ZeroInitilize();

    vector<Tensor*> weight_bias;
    weight_bias.push_back(&layer1_weight);
    weight_bias.push_back(&layer1_bias);
    weight_bias.push_back(&layer2_weight);
    weight_bias.push_back(&layer2_bias);
    weight_bias.push_back(&layer3_weight);
    weight_bias.push_back(&layer3_bias);

    // example of serializing weights and biases
    // save(weight_bias, "weight_bias.bin");
    // vector<Tensor*>* weight_bias_loaded = load("weight_bias.bin");


    //Example with single thread
    // int number_of_epochs = 1;
    //Model::evaluate(weight_bias, data_test);
    // auto start = std::chrono::high_resolution_clock::now();
    // for (int i = 0; i < number_of_epochs; i++) {
    //     Model::training(weight_bias, data);
    // }
    // auto end = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> duration = end - start;
    // std::cout << "Execution time: " << duration.count() << " seconds" << std::endl;
    // Model::evaluate(weight_bias, data_test);


    // Example with multiple threads and classic SGD
     Model::learning_rate = 0.01;
     int number_of_epochs = 3;
     auto start = std::chrono::high_resolution_clock::now();
     Model::evaluate(weight_bias, data_test);
     Model::training_multiple_threads(weight_bias, data, false, 56, 10, number_of_epochs);
     auto end = std::chrono::high_resolution_clock::now();
     std::chrono::duration<double> duration = end - start;
     std::cout << "Execution time: " << duration.count() << " seconds" << std::endl;
     Model::evaluate(weight_bias, data_test);


    //Example with multiple threads and adam
    // Model::learning_rate = 0.007;
    // Model::epsilon = 1e-8f;
    // Model::first_moment_decay = 0.9;
    // Model::second_moment_decay = 0.999;
    // Model::t_index = 1;
    //
    // int number_of_epochs = 10;
    // auto start = std::chrono::high_resolution_clock::now();
    // Model::evaluate(weight_bias, data_test);
    // Model::training_multiple_threads(weight_bias, data, true, 56, 10, number_of_epochs);
    // auto end = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> duration = end - start;
    // std::cout << "Execution time: " << duration.count() << " seconds" << std::endl;
    // Model::evaluate(weight_bias, data_test);



    return 0;
}
