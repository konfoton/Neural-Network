#ifndef THREAD_POOL_H
#define THREAD_POOL_H

#include <atomic>
#include <condition_variable>
#include <functional>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

class ThreadPool {
public:
  ThreadPool(size_t num_threads);

  ~ThreadPool();

  ThreadPool(const ThreadPool &) = delete;
  ThreadPool &operator=(const ThreadPool &) = delete;
  ThreadPool(ThreadPool &&) = delete;
  ThreadPool &operator=(ThreadPool &&) = delete;

  template <class F, class... Args>
  auto enqueue(F &&f, Args &&...args)
      -> std::future<typename std::invoke_result<F, Args...>::type>;

  bool allTasksCompleted();
  size_t queueSize() const;
  void set_number_to_do(int number);
  int number_to_do = -1;

private:
  std::vector<std::thread> workers;

  mutable std::mutex mutex_number_to_do;
  std::condition_variable condition_number_to_do;
  std::condition_variable condition_number_to_do_zero;

  std::queue<std::function<void()>> tasks;

  mutable std::mutex queue_mutex;
  std::condition_variable condition;
  bool stop;
};

template <class F, class... Args>
auto ThreadPool::enqueue(F &&f, Args &&...args)
    -> std::future<typename std::invoke_result<F, Args...>::type> {

  using return_type = typename std::invoke_result<F, Args...>::type;

  auto task = std::make_shared<std::packaged_task<return_type()>>(
      std::bind(std::forward<F>(f), std::forward<Args>(args)...));

  std::future<return_type> res = task->get_future();

  {
    std::unique_lock<std::mutex> lock1(queue_mutex);
    if (stop) {
      throw std::runtime_error("enqueue on stopped ThreadPool");
    }

    tasks.emplace([task, this]() {
      std::unique_lock<std::mutex> lock(mutex_number_to_do);
      if (number_to_do == -1) {
        condition_number_to_do.wait(lock,
                                    [this] { return number_to_do != -1; });
      }
      lock.unlock();

      (*task)();

      lock.lock();

      number_to_do--;
      if (number_to_do == 0) {
        condition_number_to_do_zero.notify_one();
      }
      lock.unlock();
    });
  }

  condition.notify_one();
  return res;
}

#endif