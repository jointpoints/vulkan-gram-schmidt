# Gram-Schmidt process on GPU

This repository contains an implementation of the [Gram-Schmidt process](https://en.wikipedia.org/wiki/Gram–Schmidt_process) written on C++ with Vulkan API.

## Requirements

To run this code, you'll need to have:

* C++11-compatible compiler;
* Vulkan SDK with API version 1.2 (provided by [LunarG](https://vulkan.lunarg.com/sdk/home), for example);
* A GPU capable of compute operations in double precision and having at least one partition of device memory that is both host visible and host coherent.

## Usage

Steps are as follows:
1. Include file `vulkan-gram-schmidt/vulkan-gram-schmidt.hpp` into your program.
2. Write your code.
3. During compilation, add the path to the `Include` folder of your Vulkan SDK to the include path.
4. During linking, link the static library `vulkan-1.lib` from the `Lib` folder of your Vulkan SDK.
5. Run.

If you do not need benchmarking data or a tool for benchmarking, you may delete the `benchmark` folder.

## Example

```c++
// Let's find an orthonormal basis in 2D
// with initial vectors (1, 2) and (3, 4)
#include <iostream>
#include "vulkan-gram-schmidt/vulkan-gram-schmidt.hpp"

int main(void)
{
    // Before we can use the solver, we need to specify the path to
    // the vulkan-gram-schmidt.spv file - the binary SPIR-V shader
    // that computes Gram-Schmidt. You may find an equivalent shader
    // vulkan-gram-schmidt.comp written in GLSL in the vulkan-gram-schmidt
    // folder.
    GPUGramSchmidt::shader_folder = "./vulkan-gram-schmidt";
    // Next, we create the solver itself.
    GPUGramSchmidt solver;
    
    // Imagine that you are a mathematician and you only store vector
    // coordinates as columns.
    // Matrix type here is just std::vector<std::vector<double>>
    GPUGramSchmidt::Matrix vectors{{1, 3},
                                   {2, 4}};
    
    // Run the solver
    solver.run(vectors, /*vectors_as_columns = */ true);
    
    // Print the answer!
    for (auto &row : vectors)
    {
        for (auto &elem : row)
            std::cout << elem << '\t';
        std::cout << '\n';
    }
    
    return 0;
}
```

The output will be:

    0.447214        0.894427
    0.894427        -0.447214

Thus, the resulting ortonormal basis is the first column (0.447214, 0.894427) and the second column (0.894427, -0.447214). If we set `vectors_as_columns` to `false`, then the rows of the matrix will be interpreted as the initial vectors and the resulting vectors will be written in rows as well.

## Further details

Documentation can be found in the `vulkan-gram-schmidt` folder.
