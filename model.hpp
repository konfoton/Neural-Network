#pragma once
#ifndef MODEL_HPP
#define MODEL_HPP
#include "Tensor.hpp"
#include "mnist_csv_reader_csv.hpp"
#include "thread_pool.hpp"
#include <cmath>
class Model {
public:
  static float learning_rate;
  static float first_moment_decay;
  static float second_moment_decay;
  static float epsilon;
  static int t_index;
  static void matmul(Tensor *input, Tensor *output, Tensor *result);
  static void matvec_transposed(Tensor *matrix, Tensor *vector, Tensor *result);
  static void matvec(Tensor *matrix, Tensor *vector, Tensor *result);
  static float ReLU(float x);
  static float dReLU(float x);
  static void softmax(Tensor *x, Tensor *y);
  static void cross_entropy(Tensor *logits, Tensor *labels);
  static void add_bias(Tensor *input, Tensor *bias, Tensor *output);
  static void activate(Tensor *input, Tensor *output);
  static int forward_pass(Tensor *input, vector<Tensor *> &stored_result,
                          vector<Tensor *> &weight_bias);
  static void backward_pass(Tensor *input, vector<Tensor *> &stored_result,
                            vector<Tensor *> &weight_bias,
                            vector<Tensor *> &weight_bias_d, Tensor *labels);
  static void training(vector<Tensor *> &weight_bias, Pack *data,
                       int batch_size);
  static void add_weight_bias(vector<Tensor *> &weight_bias_cumulative_d,
                              vector<Tensor *> &weight_bias_d);
  static float evaluate(vector<Tensor *> &weight_bias, Pack *data);
  static void update_gradient(vector<Tensor *> &weight_bias,
                              vector<Tensor *> &weight_bias_mini_batch_d,
                              int batch_size);
  static void update_gradient_adam(vector<Tensor *> &weight_bias,
                                   vector<Tensor *> &weight_bias_mini_batch_d,
                                   vector<vector<Tensor *> *> *moments,
                                   int batch_size);
  static void training_multiple_threads(vector<Tensor *> &weight_bias,
                                        Pack *data, bool adam,
                                        int mini_batch_size = 64,
                                        int num_of_threads = 10, int epoch = 1);
  static void processing_per_thread(vector<Tensor *> &weight_bias,
                                    vector<Tensor *> *weight_bias_d,
                                    vector<Tensor *> *calculations,
                                    vector<Tensor *> *weight_bias_temp_d,
                                    Pack *data, int start_index, int end_index);
  static void
  create_calculaitons_computetions_matrix(vector<vector<Tensor *> *> &store,
                                          int num);
  static vector<vector<Tensor *> *> *create_adam_matrix();
};
#endif