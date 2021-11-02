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
		GPUGramSchmidt vgs(true);
		std::cout << "1\n";
	}
	catch (std::exception &error)
	{
		std::cout << "ERROR! " << error.what() << "\n\n";
		system("pause");
	}

	return 0;
}
