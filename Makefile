CXX = g++
CXXFLAGS = -std=c++20 -g

all: result

result: thread_pool.o main.o model.o
	$(CXX) $(CXXFLAGS) thread_pool.o main.o model.o -o result

thread_pool.o: thread_pool.cpp
	$(CXX) $(CXXFLAGS) -c thread_pool.cpp

main.o: main.cpp
	$(CXX) $(CXXFLAGS) -c main.cpp

model.o: model.cpp
	$(CXX) $(CXXFLAGS) -c model.cpp

clean:
	rm -f result *.o
