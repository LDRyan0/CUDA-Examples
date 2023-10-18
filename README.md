# CUDA-Examples

A basic set of [CUDA](https://en.wikipedia.org/wiki/CUDA) examples for educational purposes.
- `vecAdd.cu`: [*N*]+[*N*] vector addition
- `matAdd.cu`: [*M*x*N*]+[*M*x*N*] matrix addition
- `matHad.cu`: [*M*x*N*]⊙[*M*x*N*] Hadamard (element-wise) product
- `matMul.cu`: [*M*x*N*][*N*x*P*] matrix multiplication (naive)
- `transpose.cu`: Tiled matrix transpose (with avoidance of memory bank conflicts)
- `coalesced.cu`: Coalesced memory access
- `stats.cu`: displays some GPU device information from `cudaGetDeviceProperties()`

## Build
Makefile automatically detects all `.cu` files and compiles with `nvcc`.
```
git clone https://github.com/LDRyan0/CUDA-Examples.git
cd CUDA-Examples/src/
make
```
- `make debug` to compile with `-DDEBUG` (used for checking of `cudaError_t` return types)
- `make release` adds `-O3` and `--use_fast_math`
