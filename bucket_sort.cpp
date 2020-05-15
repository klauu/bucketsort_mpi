#include <iostream>
#include <vector>
#include <algorithm>
#include <time.h>
#include <omp.h>
#include <chrono>
#include <math.h>

constexpr int VECTOR_SIZE = 100000;
constexpr int NUM_THREADS = 8;
constexpr int NUM_BUCKETS = 10000;
constexpr int BUCKETS_PER_THREAD = NUM_BUCKETS/NUM_THREADS;

using Buckets = std::vector<std::vector<double>>;

double generateRandom()
{
    static __thread std::mt19937 generator{ std::random_device{}() };
    std::uniform_real_distribution<double> distribution(0, 1);
    return distribution(generator);
}

void generateNumbers(std::vector<double>& v)
{
    int i{};
    #pragma omp for private(i) schedule(static, VECTOR_SIZE/NUM_THREADS)
    for (i = 0; i < VECTOR_SIZE; i++)
        v[i] = generateRandom();
}

bool isValueWithinThreadRange(int idx, int thread_id)
{
    return (idx * NUM_THREADS/NUM_BUCKETS) == thread_id;
}

void fillBuckets(Buckets& buckets, const std::vector<double>& values)
{
    auto thread_id = omp_get_thread_num();

    for (int i = 0; i < VECTOR_SIZE; i++)
        if(isValueWithinThreadRange(values[i] * NUM_BUCKETS, thread_id))
            buckets[values[i] * NUM_BUCKETS].push_back(values[i]);
}

void sortBuckets(Buckets& buckets)
{
    std::vector<int> sizes(NUM_BUCKETS);
    auto thread_id = omp_get_thread_num();
    for (int i = thread_id * BUCKETS_PER_THREAD; i < (thread_id + 1) * BUCKETS_PER_THREAD; i++)
        std::sort(buckets[i].begin(), buckets[i].end());
}

std::vector<int> countPositions(const Buckets& buckets)
{
    std::vector<int> positions (NUM_BUCKETS);
    positions[0] = buckets[0].size();
    for (int i = 1; i < NUM_BUCKETS; i++)
        positions[i] = positions[i-1] + buckets[i].size();
    return positions;
}

void concatenateBuckets(const Buckets& buckets, std::vector<double>& values)
{
    auto positions = countPositions(buckets);
    std::vector<double>::iterator start_position;
    auto thread_id = omp_get_thread_num();
    for (int i = thread_id * BUCKETS_PER_THREAD; i < (thread_id + 1) * BUCKETS_PER_THREAD; i++)
    {
        start_position = values.begin() + positions[i];
        std::copy(buckets[i].begin(), buckets[i].end(), start_position);
    }
}

template <typename F, typename ...Args>
double startFunctionWithMeasurements(F func, Args&&... args)
{
    auto start = std::chrono::high_resolution_clock::now();
    func(std::forward<Args>(args)...);
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = stop - start;
    return duration.count();
}

int main()
{
    omp_set_num_threads(NUM_THREADS);
    std::vector<double> v(VECTOR_SIZE);
    Buckets buckets(NUM_BUCKETS);


    for(auto& bucket: buckets)
        bucket.reserve(VECTOR_SIZE/NUM_BUCKETS);

    double generating_time, filling_time, sorting_time, concatenating_time, whole_time;

    #pragma omp parallel
    {
        whole_time = startFunctionWithMeasurements([&]()
        {
            generating_time = startFunctionWithMeasurements(generateNumbers, v);
            filling_time = startFunctionWithMeasurements(fillBuckets, buckets, v);
            sorting_time = startFunctionWithMeasurements(sortBuckets, buckets);
            concatenating_time = startFunctionWithMeasurements(concatenateBuckets, buckets, v);
        });
    }

    std::cout << "Generowanie liczb: " << generating_time << " [ms]" << std::endl;
    std::cout << "Wypelnianie kubelkow: " << filling_time << " [ms]" << std::endl;
    std::cout << "Sortowanie kubelkow: " << sorting_time << " [ms]" << std::endl;
    std::cout << "Laczenie kubelkow: " << concatenating_time << " [ms]" << std::endl;
    std::cout << "Calosc: " << whole_time << " [ms]" << std::endl;

    std::cout << "Posortowane: " << std::is_sorted(v.begin(), v.end()) << std::endl;  


    return 0;
}