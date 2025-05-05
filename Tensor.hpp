    #ifndef TENSOR_H
    #define TENSOR_H

    #include <vector>
    #include <random>
    using namespace std;
    class Tensor {
    public:
        Tensor(vector<int> shape) {
            this->shape = shape;

            int size_to_alloc = 1;
            for(int i = 0; i < shape.size(); i++) {
                size_to_alloc *= shape[i];
            }
            stride.push_back(1);
            int value = 1;
            for (int i = shape.size() - 1; i >=1 ; i--) {
                value *= shape[i];
                stride.insert(stride.begin(), value);
            }
            this->size = size_to_alloc;
            this->storage = new float[size_to_alloc];
        }

        int access(int dim_idx = 0) {
            return 0;
        }

        template <typename T, typename... Args>
        int access(int dim_idx, T first, Args... args) {

            if (dim_idx >= shape.size()) {
                throw std::out_of_range("Dimension index out of bounds");
            }
            if (first < 0 || first >= shape[dim_idx]) {
                throw std::out_of_range("Index out of bounds for dimension " + std::to_string(dim_idx));
            }

            return first * stride[dim_idx] + access(dim_idx + 1, args...);
        }

        template <typename... Args>
        float& at(Args... args) {
            int offset = access(0, args...);
            return storage[offset];
        }

        ~Tensor() {
            delete[] storage;
        }
        void HeInitilize() {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::normal_distribution<> dist(0.0, sqrt(2.0/ shape[0]));
            for (int i = 0; i < size; i++) {
                storage[i] = dist(gen);
            }
        }

        void ZeroInitilize() {
            for (int i = 0; i < size; i++) {
                storage[i] = 0;
            }
        }
        void transpose() {
            if (stride.size() != 2) {
                throw std::out_of_range("Tensor transpose requires stride vector");
            }
            vector<int> temp;

            temp.push_back(stride[1]);
            temp.push_back(stride[0]);
            stride = temp;
            temp.clear();
            temp.push_back(shape[1]);
            temp.push_back(shape[0]);
            shape = temp;

        }
        float& at_transpose(int a, int b){
            return storage[a * stride[1] + b * stride[0]];
        }

        float* storage;
        int size;
        vector<int> shape;
        vector<int> stride;
    };

#endif