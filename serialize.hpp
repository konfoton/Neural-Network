#pragma once
#include "Tensor.hpp"
#include <fstream>
#include <iostream>

void save(vector<Tensor *> &p, const std::string &filename) {
  std::ofstream ofs(filename, std::ios::binary);
  for (int i = 0; i < 6; i++) {
    int size = (p[i]->shape.size());
    ofs.write(reinterpret_cast<const char *>(&(size)), sizeof(int));
    for (int j = 0; j < p[i]->shape.size(); j++) {
      ofs.write(reinterpret_cast<const char *>(&(p[i]->shape[j])),
                sizeof(p[i]->shape[j]));
    }

    for (int j = 0; j < p[i]->size; j++) {
      ofs.write(reinterpret_cast<const char *>(p[i]->storage + j),
                sizeof(p[i]->storage[j]));
    }
  }
  ofs.close();
}

vector<Tensor *> *load(const std::string &filename) {
  auto model_object = new vector<Tensor *>();
  std::ifstream ifs(filename, std::ios::binary);
  for (int i = 0; i < 6; i++) {
    int size;
    int element;
    vector<int> shape;
    ifs.read(reinterpret_cast<char *>(&size), sizeof(int));
    for (int j = 0; j < size; j++) {
      ifs.read(reinterpret_cast<char *>(&element), sizeof(int));
      shape.push_back(element);
    }
    Tensor *p = new Tensor(shape);
    for (int j = 0; j < p->size; j++) {
      ifs.read(reinterpret_cast<char *>(p->storage + j), sizeof(p->storage[j]));
    }
    model_object->push_back(p);
  }
  ifs.close();
  return model_object;
}