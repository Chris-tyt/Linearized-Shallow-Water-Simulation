CPP=CC
CFLAGS=-lm -fopenmp
COPTFLAGS=-Ofast -ffast-math -flto -march=native -ftree-vectorize -fprefetch-loop-arrays -funroll-loops -fno-math-errno -falign-loops=32
OPTFLAGS=-O3 -ffast-math -flto -march=native -ftree-vectorize -fprefetch-loop-arrays -funroll-loops -fno-math-errno -falign-loops=32
MPIFLAGS=-DMPI
DEBUGFLAGS=-g -pg

NVCC=nvcc
NVCCFLAGS=-DCUDA

PYTHON=python3

all: mpi gpu basic_serial

mpi: build/mpi
gpu: build/gpu
serial: build/serial
serial_omp: build/serial_omp
basic_serial: build/basic_serial

build/mpi: common/main.cpp common/scenarios.cpp mpi/mpi.cpp
	$(CPP) $^ -o $@ $(MPIFLAGS) $(CFLAGS) $(OPTFLAGS)

build/gpu: common/main.cpp common/scenarios.cpp gpu/gpu.cu
	$(NVCC) $^ -o $@ $(NVCCFLAGS)

build/serial: common/main.cpp common/scenarios.cpp serial/serial.cpp
	$(CPP) $^ -o $@ $(CFLAGS) $(COPTFLAGS)

build/serial_omp: common/main.cpp common/scenarios.cpp serial/serial_omp.cpp
	$(CPP) $^ -o $@ $(CFLAGS) $(COPTFLAGS)

build/basic_serial: common/main.cpp common/scenarios.cpp serial/basic_serial.cpp
	$(CPP) $^ -o $@ $(CFLAGS) $(COPTFLAGS)

.PHONY: clean

clean:
	rm -f *.out
	rm -f build/*
	rm -f *.gif
	rm -f *.txt
	rm -f *.ncu-rep