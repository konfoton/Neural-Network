#include "thread_pool.hpp"

ThreadPool::ThreadPool(size_t num_threads) : stop(false) {
    for (size_t i = 0; i < num_threads; ++i) {
        workers.emplace_back([this] {
            while (true) {
                std::function<void()> task;

                {

                    std::unique_lock<std::mutex> lock(this->queue_mutex);

                    this->condition.wait(lock, [this] {
                        return this->stop || !this->tasks.empty();
                    });
                    if (this->stop && this->tasks.empty()) {
                        return;
                    }

                    task = std::move(this->tasks.front());
                    this->tasks.pop();
                }

                task();
            }
        });
    }
}

ThreadPool::~ThreadPool() {
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        stop = true;
    }

    condition.notify_all();

    for (std::thread& worker : workers) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

bool ThreadPool::allTasksCompleted() {
    std::unique_lock<std::mutex> lock(mutex_number_to_do);
    if(number_to_do != 0) {
        condition_number_to_do_zero.wait(lock, [this] { return (number_to_do == 0); });
    }
    return true;
}

void ThreadPool::set_number_to_do(int number){
  std::unique_lock<std::mutex> lock(mutex_number_to_do);
  number_to_do = number;
  condition_number_to_do.notify_all();
}

size_t ThreadPool::queueSize() const {
    std::unique_lock<std::mutex> lock(queue_mutex);
    return tasks.size();
}