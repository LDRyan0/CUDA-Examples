all: matAdd.cu
	nvcc -o matAdd matAdd.cu

clean:
	rm -f matAdd
