/**
 * @file main.cpp
 * @author JointPoints, 2021, github.com/jointpoints
 */
#include "../vulkan-gram-schmidt/vulkan-gram-schmidt.hpp"
#include <exception>
#include <chrono>
#include <random>
#include <iostream>





double average_time_for_one_random_matrix(std::default_random_engine &generator, std::uniform_real_distribution<double> &pseudorandom, GPUGramSchmidt &vgs, uint8_t const n, uint8_t const repetitions = 10)
{
	GPUGramSchmidt::Matrix matrix(n, std::vector<double>(n, 0.0));
	auto start_time = std::chrono::high_resolution_clock::now();
	auto runtime = std::chrono::high_resolution_clock::now() - start_time;
	double total_runtime = 0.0;

	for (auto &row : matrix)
		for (uint32_t i = 0; i < n; ++i)
			row[i] = pseudorandom(generator);
	
	start_time = std::chrono::high_resolution_clock::now();
	for (uint8_t repeat_i = 0; repeat_i < repetitions; ++repeat_i)
	{
		GPUGramSchmidt::Matrix matrix_copy(matrix);
		start_time = std::chrono::high_resolution_clock::now();
		vgs.run(matrix_copy);
		runtime = std::chrono::high_resolution_clock::now() - start_time;
		total_runtime += std::chrono::duration_cast<std::chrono::microseconds>(runtime).count() * 0.000001;
	}

	return total_runtime / repetitions;
}





void benchmarking(void)
{
	// Set up a path to "shader_folder" that contains Gram-Schmidt SPIR-V compute shader
	GPUGramSchmidt::shader_folder = "../vulkan-gram-schmidt";
	// Create GPU solver
	GPUGramSchmidt vgs(true);
	// Create pseudorandom number generator
	std::default_random_engine generator(0);
	std::uniform_real_distribution<double> pseudorandom(0.001, 20.0);
	// Perform tests on random matrices of different orders
	uint8_t const orders_count = 10;
	uint16_t const orders[] = {2, 5, 10, 50, 100, 500, 1000, 5000, 10000, 50000};
	double average_runtimes[orders_count];
	for (uint8_t order_i = 0; order_i < orders_count; ++order_i)
	{
		average_runtimes[order_i] = 0.0;
		for (uint8_t matrix_i = 0; matrix_i < 50; ++matrix_i)
			average_runtimes[order_i] += average_time_for_one_random_matrix(generator, pseudorandom, vgs, orders[order_i]);
		average_runtimes[order_i] /= 50;
		std::cout << orders[order_i] << '\t' << average_runtimes[order_i] << '\n';
	}
	/*for (auto &row : matrix)
	{
		for (auto &elem : row)
			std::cout << elem << '\t';
		std::cout << '\n';
	}*/
	return;
}





int main(void)
{
	try
	{
		benchmarking();
	}
	catch (std::exception &error)
	{
		std::cout << "ERROR! " << error.what() << "\n\n";
		system("pause");
	}

	return 0;
}
