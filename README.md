# CUDA-Examples

A basic set of programs that perform a variety of parralelised linear algebra operations using [CUDA](https://en.wikipedia.org/wiki/CUDA).
- [*N*]+[*N*] vector addition
- [*M*x*N*]+[*M*x*N*] matrix addition
- [*M*x*N*]⊙[*M*x*N*] Hadamard (element-wise) product
- [*M*x*N*][*N*x*P*] matrix multiplication


Compile code using 
```
nvcc <filename>.cu -o <exec>
```
