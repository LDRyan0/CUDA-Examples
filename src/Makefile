NVCC = nvcc
NVCCFLAGS =

# automatically find all .cu files and compile them directly into
# exectuable of the same name
SOURCES = $(shell ls *.cu)
EXECS = $(patsubst %.cu,%,$(SOURCES))

all : $(EXECS)
	
debug : NVCCFLAGS +=-g -G -DDEBUG
debug : $(EXECS)

release : NVCCFLAGS +=-O3 --use_fast_math
release: $(EXECS)

$(EXECS): %: %.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean : 
	rm $(EXECS)

