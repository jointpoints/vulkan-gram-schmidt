/**
 * @file main.cpp
 * @author JointPoints, 2021, github.com/jointpoints
 */
#include "../vulkan-gram-schmidt/vulkan-gram-schmidt.hpp"
#include <exception>
#include <iostream>





int main(void)
{
	try
	{
		GPUGramSchmidt::shader_folder = "../vulkan-gram-schmidt";
		GPUGramSchmidt vgs(true);
		std::cout << vgs.run() << "\n";
	}
	catch (std::exception &error)
	{
		std::cout << "ERROR! " << error.what() << "\n\n";
		system("pause");
	}

	return 0;
}
