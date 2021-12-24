# Benchmarking

## Procedure

The benchmarking was performed in the following way. For the set of dimensions, namely,
_n_ = 2, 5, 10, 50, 100, 500, 1000, 5000, 10000, 50000 fifty different random _n_ x _n_ matrices
were generated which then were passed to the solver ten times each. The runtimes for each matrix
were averaged resulting in the average runtime of a single matrix (ARSM). After this, all fifty
ARSM's were averaged for each value of _n_.

## Results

This implementation has shown the following average runtime on NVIDIA GTX 1650 Ti.

| Dimension | Average runtime on 50 random matrices (s) |
|:---------:|:-----------------------------------------:|
|     2     |0.000215242|
|     5     |0.000500822|
|    10     |0.00101816|
|    50     |0.0075756|
|    100    |0.0227079|
|    500    |0.196061|
|   1000    |0.190567|
|   5000    |0.0790866|
|   10000   |0.00180731|
|   50000   |0.0175033|

![Plot](https://github.com/jointpoints/vulkan-gram-schmidt/blob/main/benchmark/Benchmark.png)
