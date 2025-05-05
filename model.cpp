#include "model.hpp"
float Model::learning_rate = 0.01;
float Model::first_moment_decay = 0.9;
float Model::second_moment_decay = 0.999;
float Model::epsilon = 1e-8f;
int Model::t_index = 1;

void Model::matmul(Tensor *input, Tensor *output, Tensor *result) {
  if (input->shape.size() != 2 || output->shape.size() != 2 ||
      result->shape.size() != 2) {
    throw std::invalid_argument(
        "All tensors must be 2D for matrix multiplication");
  }

  if (input->shape[1] != output->shape[0]) {
    throw std::invalid_argument(
        "Incompatible dimensions: input columns must match output rows");
  }

  if (result->shape[0] != input->shape[0] ||
      result->shape[1] != output->shape[1]) {
    throw std::invalid_argument("Result tensor has incorrect dimensions");
  }

  int m = input->shape[0];
  int n = input->shape[1];
  int p = output->shape[1];

  for (int i = 0; i < m; i++) {
    for (int j = 0; j < p; j++) {
      float sum = 0.0f;
      for (int k = 0; k < n; k++) {
        sum += input->at(i, k) * output->at(k, j);
      }
      result->at(i, j) = sum;
    }
  }
}
void Model::matvec_transposed(Tensor *matrix, Tensor *vector, Tensor *result) {
  if (matrix->shape.size() != 2) {
    throw std::invalid_argument(
        "Matrix must be 2D for matrix-vector multiplication");
  }

  if (vector->shape.size() != 1) {
    throw std::invalid_argument(
        "Vector must be 1D for matrix-vector multiplication");
  }

  if (result->shape.size() != 1) {
    throw std::invalid_argument(
        "Result must be 1D for matrix-vector multiplication");
  }

  int n = matrix->shape[0];
  int m = matrix->shape[1];

  for (int i = 0; i < m; i++) {
    float sum = 0.0f;
    for (int j = 0; j < n; j++) {
      sum += matrix->at_transpose(i, j) * vector->at(j);
    }
    result->at(i) = sum;
  }
}

void Model::matvec(Tensor *matrix, Tensor *vector, Tensor *result) {
  if (matrix->shape.size() != 2) {
    throw std::invalid_argument(
        "Matrix must be 2D for matrix-vector multiplication");
  }

  if (vector->shape.size() != 1) {
    throw std::invalid_argument(
        "Vector must be 1D for matrix-vector multiplication");
  }

  if (result->shape.size() != 1) {
    throw std::invalid_argument(
        "Result must be 1D for matrix-vector multiplication");
  }

  int m = matrix->shape[0];
  int n = matrix->shape[1];

  for (int i = 0; i < m; i++) {
    float sum = 0.0f;
    for (int j = 0; j < n; j++) {
      sum += matrix->at_transpose(i, j) * vector->at(j);
    }
    result->at(i) = sum;
  }
}

float Model::ReLU(float x) { return max(0.0f, x); }
float Model::dReLU(float x) {
  if (x > 0.0f) {
    return 1.0f;
  } else {
    return 0;
  }
}
void Model::softmax(Tensor *x, Tensor *y) {
  float sum = 0;
  for (int i = 0; i < x->shape[0]; i++) {
    sum += exp(x->at(i));
  }
  for (int i = 0; i < x->shape[0]; i++) {
    y->at(i) = exp(x->at(i)) / sum;
  }
}
void Model::cross_entropy(Tensor *logits, Tensor *labels) {
  float sum = 0.0f;
  for (int i = 0; i < labels->shape[0]; i++) {
    sum += log(logits->at(i)) * labels->at(i) * -1.f;
  }
}

void Model::add_bias(Tensor *input, Tensor *bias, Tensor *output) {
  if (input->size != bias->size) {
    throw std::invalid_argument("Incompatible dimensions");
  }
  for (int i = 0; i < input->size; i++) {
    output->at(i) = ReLU(input->at(i) + bias->at(i));
  }
}
void Model::activate(Tensor *input, Tensor *output) {
  for (int i = 0; i < input->size; i++) {
    output->at(i) = ReLU(input->at(i));
  }
}
int Model::forward_pass(Tensor *input, vector<Tensor *> &stored_result,
                        vector<Tensor *> &weight_bias) {

  matvec_transposed(weight_bias[0], input, stored_result[0]);
  add_bias(stored_result[0], weight_bias[1], stored_result[0]);
  activate(stored_result[0], stored_result[1]);

  matvec_transposed(weight_bias[2], stored_result[1], stored_result[2]);
  add_bias(stored_result[2], weight_bias[3], stored_result[2]);
  activate(stored_result[2], stored_result[3]);

  matvec_transposed(weight_bias[4], stored_result[3], stored_result[4]);
  add_bias(stored_result[4], weight_bias[5], stored_result[4]);

  softmax(stored_result[4], stored_result[5]);
  int index = 0;
  float value = -1;
  for (int i = 0; i < stored_result[5]->shape[0]; i++) {
    if (stored_result[5]->at(i) > value) {
      index = i;
      value = stored_result[5]->at(i);
    }
  }
  return index;
}
void Model::backward_pass(Tensor *input, vector<Tensor *> &stored_result,
                          vector<Tensor *> &weight_bias,
                          vector<Tensor *> &weight_bias_d, Tensor *labels) {

  for (int i = 0; i < 10; i++) {
    weight_bias_d[weight_bias_d.size() - 1]->at(i) =
        stored_result[stored_result.size() - 1]->at(i) - labels->at(i);
  }
  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < 10; j++) {
      weight_bias_d[weight_bias_d.size() - 2]->at(i, j) =
          weight_bias_d[weight_bias_d.size() - 1]->at(j) *
          stored_result[stored_result.size() - 2]->at(i);
      weight_bias_d[weight_bias_d.size() - 3]->at(i) +=
          weight_bias_d[weight_bias_d.size() - 1]->at(j) *
          weight_bias[weight_bias_d.size() - 2]->at(i, j);
    }
  }
  for (int i = 0; i < 64; i++) {
    weight_bias_d[weight_bias_d.size() - 3]->at(i) *=
        dReLU(stored_result[stored_result.size() - 4]->at(i));
  }

  for (int i = 0; i < 128; i++) {
    for (int j = 0; j < 64; j++) {
      weight_bias_d[weight_bias_d.size() - 4]->at(i, j) =
          weight_bias_d[weight_bias_d.size() - 3]->at(j) *
          stored_result[stored_result.size() - 5]->at(i);
      weight_bias_d[weight_bias_d.size() - 5]->at(i) +=
          weight_bias_d[weight_bias_d.size() - 3]->at(j) *
          weight_bias[weight_bias_d.size() - 4]->at(i, j);
    }
  }
  for (int i = 0; i < 128; i++) {
    weight_bias_d[weight_bias_d.size() - 5]->at(i) *=
        dReLU(stored_result[stored_result.size() - 6]->at(i));
  }
  for (int i = 0; i < 784; i++) {
    for (int j = 0; j < 128; j++) {
      weight_bias_d[weight_bias_d.size() - 6]->at(i, j) =
          weight_bias_d[weight_bias_d.size() - 5]->at(j) * input->at(i);
    }
  }
}
void Model::training(vector<Tensor *> &weight_bias, Pack *data,
                     int batch_size) {

  Tensor layer1_weight_mini_batch_d(vector<int>{784, 128});
  Tensor layer1_bias_mini_batch_d(vector<int>{128});
  Tensor layer2_weight_mini_batch_d(vector<int>{128, 64});
  Tensor layer2_bias_mini_batch_d(vector<int>{64});
  Tensor layer3_weight_mini_batch_d(vector<int>{64, 10});
  Tensor layer3_bias_mini_batch_d(vector<int>{10});

  vector<Tensor *> weight_bias_mini_batch_d;

  weight_bias_mini_batch_d.push_back(&layer1_weight_mini_batch_d);
  weight_bias_mini_batch_d.push_back(&layer1_bias_mini_batch_d);
  weight_bias_mini_batch_d.push_back(&layer2_weight_mini_batch_d);
  weight_bias_mini_batch_d.push_back(&layer2_bias_mini_batch_d);
  weight_bias_mini_batch_d.push_back(&layer3_weight_mini_batch_d);
  weight_bias_mini_batch_d.push_back(&layer3_bias_mini_batch_d);

  Tensor layer1_calculation(vector<int>{128});
  Tensor layer1_calculation_relu(vector<int>{128});
  Tensor layer2_caluculation(vector<int>{64});
  Tensor layer2_calculation_relu(vector<int>{64});
  Tensor layer3_calculation(vector<int>{10});
  Tensor layer3_softmax(vector<int>{10});

  vector<Tensor *> calculations;

  calculations.push_back(&layer1_calculation);
  calculations.push_back(&layer1_calculation_relu);
  calculations.push_back(&layer2_caluculation);
  calculations.push_back(&layer2_calculation_relu);
  calculations.push_back(&layer3_calculation);
  calculations.push_back(&layer3_softmax);

  Tensor layer1_weight_d(vector<int>{784, 128});
  Tensor layer1_bias_d(vector<int>{128});
  Tensor layer2_weight_d(vector<int>{128, 64});
  Tensor layer2_bias_d(vector<int>{64});
  Tensor layer3_weight_d(vector<int>{64, 10});
  Tensor layer3_bias_d(vector<int>{10});

  vector<Tensor *> weight_bias_d;

  weight_bias_d.push_back(&layer1_weight_d);
  weight_bias_d.push_back(&layer1_bias_d);
  weight_bias_d.push_back(&layer2_weight_d);
  weight_bias_d.push_back(&layer2_bias_d);
  weight_bias_d.push_back(&layer3_weight_d);
  weight_bias_d.push_back(&layer3_bias_d);

  for (int i = 0; i < data->train_data.size(); i++) {
    if (i % batch_size == 0) {
      update_gradient(weight_bias, weight_bias_mini_batch_d, batch_size);
      for (Tensor *t : weight_bias_mini_batch_d) {
        t->ZeroInitilize();
      }
      cout << i / batch_size << endl;
    }

    for (Tensor *t : calculations) {
      t->ZeroInitilize();
    }
    for (Tensor *t : weight_bias_d) {
      t->ZeroInitilize();
    }
    forward_pass(data->train_data[i], calculations, weight_bias);
    backward_pass(data->train_data[i], calculations, weight_bias, weight_bias_d,
                  data->train_label[i]);
    add_weight_bias(weight_bias_mini_batch_d, weight_bias_d);
  }
}
void Model::add_weight_bias(vector<Tensor *> &weight_bias_cumulative_d,
                            vector<Tensor *> &weight_bias_d) {
  for (int i = 0; i < weight_bias_cumulative_d.size(); i++) {
    for (int j = 0; j < weight_bias_cumulative_d[i]->size; j++) {
      weight_bias_cumulative_d[i]->storage[j] += weight_bias_d[i]->storage[j];
    }
  }
}

float Model::evaluate(vector<Tensor *> &weight_bias, Pack *data) {

  vector<Tensor *> calculations;
  Tensor layer1(vector<int>{128});
  Tensor layer1_relu(vector<int>{128});
  Tensor layer2(vector<int>{64});
  Tensor layer2_relu(vector<int>{64});
  Tensor layer3(vector<int>{10});
  Tensor layer4(vector<int>{10});
  calculations.push_back(&layer1);
  calculations.push_back(&layer1_relu);
  calculations.push_back(&layer2);
  calculations.push_back(&layer2_relu);
  calculations.push_back(&layer3);
  calculations.push_back(&layer4);

  int correct = 0;
  cout << "Evaluating model..." << endl;
  for (int i = 0; i < data->train_data.size(); i++) {
    correct += data->train_label[i]->at(
        forward_pass(data->train_data[i], calculations, weight_bias));
  }
  cout << "Efficiency "
       << ((float)correct / (float)data->train_data.size()) * 100 << "%"
       << endl;
  return (float)correct / (float)data->train_data.size();
}
void Model::update_gradient(vector<Tensor *> &weight_bias,
                            vector<Tensor *> &weight_bias_mini_batch_d,
                            int batch_size) {
  for (int i = 0; i < weight_bias_mini_batch_d.size(); i++) {
    for (int j = 0; j < weight_bias_mini_batch_d[i]->size; j++) {
      weight_bias[i]->storage[j] -= learning_rate *
                                    weight_bias_mini_batch_d[i]->storage[j] /
                                    (float)batch_size;
    }
  }
}
void Model::update_gradient_adam(vector<Tensor *> &weight_bias,
                                 vector<Tensor *> &weight_bias_mini_batch_d,
                                 vector<vector<Tensor *> *> *moments,
                                 int batch_size) {
  for (int i = 0; i < (*moments)[0]->size(); i++) {
    for (int j = 0; j < (*(*moments)[0])[i]->size; j++) {
      float gradient =
          weight_bias_mini_batch_d[i]->storage[j] / (float)batch_size;

      // Update first moment (m)
      (*(*moments)[0])[i]->storage[j] =
          (*(*moments)[0])[i]->storage[j] * first_moment_decay +
          (1 - first_moment_decay) * gradient;

      // Update second moment (v) - with gradient squared
      (*(*moments)[1])[i]->storage[j] =
          (*(*moments)[1])[i]->storage[j] * second_moment_decay +
          (1 - second_moment_decay) * gradient * gradient;

      // Apply bias correction and update weights
      float m_corrected = (*(*moments)[0])[i]->storage[j] /
                          (1 - pow(first_moment_decay, t_index));
      float v_corrected = (*(*moments)[1])[i]->storage[j] /
                          (1 - pow(second_moment_decay, t_index));

      weight_bias[i]->storage[j] -=
          learning_rate * m_corrected / (sqrt(v_corrected) + epsilon);
    }
  }
  t_index += 1;
}

void Model::training_multiple_threads(vector<Tensor *> &weight_bias, Pack *data,
                                      bool adam, int mini_batch_size,
                                      int num_of_threads, int epoch) {
  vector<vector<Tensor *> *> *adam_data;
  if (adam) {
    adam_data = create_adam_matrix();
  }
  ThreadPool pool(num_of_threads);
  vector<vector<Tensor *> *> store;
  create_calculaitons_computetions_matrix(store, num_of_threads);

  int no_of_mini_batches = data->train_data.size() / mini_batch_size;
  int no_of_dp_per_thread = mini_batch_size / (num_of_threads - 1);
  int last_thread = mini_batch_size % (num_of_threads - 1);

  Tensor layer1_weight_mini_batch_d(vector<int>{784, 128});
  Tensor layer1_bias_mini_batch_d(vector<int>{128});
  Tensor layer2_weight_mini_batch_d(vector<int>{128, 64});
  Tensor layer2_bias_mini_batch_d(vector<int>{64});
  Tensor layer3_weight_mini_batch_d(vector<int>{64, 10});
  Tensor layer3_bias_mini_batch_d(vector<int>{10});

  vector<Tensor *> weight_bias_mini_batch_d;

  weight_bias_mini_batch_d.push_back(&layer1_weight_mini_batch_d);
  weight_bias_mini_batch_d.push_back(&layer1_bias_mini_batch_d);
  weight_bias_mini_batch_d.push_back(&layer2_weight_mini_batch_d);
  weight_bias_mini_batch_d.push_back(&layer2_bias_mini_batch_d);
  weight_bias_mini_batch_d.push_back(&layer3_weight_mini_batch_d);
  weight_bias_mini_batch_d.push_back(&layer3_bias_mini_batch_d);

  for (int k = 0; k < epoch; k++) {
    cout << "epoch: " << k << endl;
    for (int i = 0; i < no_of_mini_batches - 5; i++) {
      pool.set_number_to_do(num_of_threads);
      int counter = 0;
      for (int j = i * mini_batch_size; j < (i + 1) * (mini_batch_size);
           j = j + no_of_dp_per_thread) {
        if (counter / 3 == (num_of_threads - 1)) {
          pool.enqueue(processing_per_thread, weight_bias, store[counter],
                       store[counter + 1], store[counter + 2], data, j,
                       (i + 1) * (mini_batch_size)-1);
          continue;
        }
        pool.enqueue(processing_per_thread, weight_bias, store[counter],
                     store[counter + 1], store[counter + 2], data, j,
                     j + no_of_dp_per_thread - 1);
        counter += 3;
      }
      pool.allTasksCompleted();
      pool.set_number_to_do(-1);
      for (int j = 0; j < num_of_threads; j++) {
        add_weight_bias(weight_bias_mini_batch_d, *store[j * 3]);
        for (Tensor *t : *store[j * 3]) {
          t->ZeroInitilize();
        }
      }
      if (adam) {
        update_gradient_adam(weight_bias, weight_bias_mini_batch_d, adam_data,
                             mini_batch_size);
      } else {
        update_gradient(weight_bias, weight_bias_mini_batch_d, mini_batch_size);
      }
      for (Tensor *t : weight_bias_mini_batch_d) {
        t->ZeroInitilize();
      }
      if (i % 100 == 0) {
        cout << "Number of processed batches: " << i << endl;
      }
    }
  }
}

void Model::processing_per_thread(vector<Tensor *> &weight_bias,
                                  vector<Tensor *> *weight_bias_d,
                                  vector<Tensor *> *calculations,
                                  vector<Tensor *> *weight_bias_temp_d,
                                  Pack *data, int start_index, int end_index) {
  for (int i = start_index; i <= end_index; i++) {

    for (Tensor *t : *calculations) {
      t->ZeroInitilize();
    }
    for (Tensor *t : *weight_bias_temp_d) {
      t->ZeroInitilize();
    }
    forward_pass(data->train_data[i], *calculations, weight_bias);
    backward_pass(data->train_data[i], *calculations, weight_bias,
                  *weight_bias_temp_d, data->train_label[i]);
    add_weight_bias(*weight_bias_d, *weight_bias_temp_d);
  }
}

void Model::create_calculaitons_computetions_matrix(
    vector<vector<Tensor *> *> &store, int num) {
  for (int i = 0; i < num; i++) {
    auto calculations = new vector<Tensor *>();
    calculations->push_back(new Tensor(vector<int>{128}));
    calculations->push_back(new Tensor(vector<int>{128}));
    calculations->push_back(new Tensor(vector<int>{64}));
    calculations->push_back(new Tensor(vector<int>{64}));
    calculations->push_back(new Tensor(vector<int>{10}));
    calculations->push_back(new Tensor(vector<int>{10}));

    auto weight_bias = new vector<Tensor *>();
    weight_bias->push_back(new Tensor(vector<int>{784, 128}));
    weight_bias->push_back(new Tensor(vector<int>{128}));
    weight_bias->push_back(new Tensor(vector<int>{128, 64}));
    weight_bias->push_back(new Tensor(vector<int>{64}));
    weight_bias->push_back(new Tensor(vector<int>{64, 10}));
    weight_bias->push_back(new Tensor(vector<int>{10}));

    auto weight_bias_temp_d = new vector<Tensor *>();
    weight_bias_temp_d->push_back(new Tensor(vector<int>{784, 128}));
    weight_bias_temp_d->push_back(new Tensor(vector<int>{128}));
    weight_bias_temp_d->push_back(new Tensor(vector<int>{128, 64}));
    weight_bias_temp_d->push_back(new Tensor(vector<int>{64}));
    weight_bias_temp_d->push_back(new Tensor(vector<int>{64, 10}));
    weight_bias_temp_d->push_back(new Tensor(vector<int>{10}));

    store.push_back(weight_bias);
    store.push_back(calculations);
    store.push_back(weight_bias_temp_d);
  }
}
vector<vector<Tensor *> *> *Model::create_adam_matrix() {
  auto store = new vector<vector<Tensor *> *>;
  for (int i = 0; i < 2; i++) {
    auto weight_bias_d = new vector<Tensor *>();
    weight_bias_d->push_back(new Tensor(vector<int>{784, 128}));
    weight_bias_d->push_back(new Tensor(vector<int>{128}));
    weight_bias_d->push_back(new Tensor(vector<int>{128, 64}));
    weight_bias_d->push_back(new Tensor(vector<int>{64}));
    weight_bias_d->push_back(new Tensor(vector<int>{64, 10}));
    weight_bias_d->push_back(new Tensor(vector<int>{10}));
    for (Tensor *t : *weight_bias_d) {
      t->ZeroInitilize();
    }
    store->push_back(weight_bias_d);
  }
  return store;
}
