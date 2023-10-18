#include <iostream>

void showProperties(int devId) {
        std::cout << "Device: " << devId << "\n";;

        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, devId);

        std::cout << "Name: " << prop.name << "\n";
        std::cout << "totalGlobalMem: " << prop.totalGlobalMem / (1024 * 1024 * 1024) << "GB\n";
        std::cout << "sharedMemPerBlock: " << prop.sharedMemPerBlock / 1024 << "MB\n";
        std::cout << "regsPerBlock: " << prop.regsPerBlock << "\n";
        std::cout << "warpSize: " << prop.warpSize << "\n";
        std::cout << "maxThreadsPerBlock: " << prop.maxThreadsPerBlock << "\n";
        for(int i = 0; i < 3; ++i)
            std::printf("maxThreadsDim[%d]: %d\n", i, prop.maxThreadsDim[i]);
        for(int i = 0; i < 3; ++i)
            std::printf("maxGridSize[%d]: %d\n", i, prop.maxGridSize[i]);
        std::cout << "clockRate: " << prop.clockRate << "\n";
        std::cout << "totalConstMem: " << prop.totalConstMem << "\n";
        std::cout << "capability: " << prop.major  << "." << prop.minor << "\n";
        std::cout << "multiProcessorCount " << prop.multiProcessorCount << "\n";
        std::cout << "integrated: " << (prop.integrated ? "TRUE" : "FALSE") << "\n";
        std::cout << "concurrentKernels: " << (prop.concurrentKernels ? "TRUE" : "FALSE") << "\n";
        std::cout << "memoryClockRate:" << prop.memoryClockRate << "\n";
        std::cout << "l2CacheSize:" << prop.l2CacheSize << "\n";
        std::cout << "maxThreadsPerMultiProcessor: " << prop.maxThreadsPerMultiProcessor << "\n";
        std::cout << "globalL1CacheSupported: " << (prop.globalL1CacheSupported ? "TRUE" : "FALSE") << "\n";
        std::cout << "localL1CacheSupported: " << (prop.localL1CacheSupported ? "TRUE" : "FALSE")<< "\n";
        std::cout << "sharedMemPerMultiprocessor: " << prop.sharedMemPerMultiprocessor << "\n";
        std::cout << "regsPerMultiprocessor: " << prop.sharedMemPerMultiprocessor << "\n";
}

int main(int argc, char **argv) {
    if (argc <= 1) 
        showProperties(0);
    else {
        for(int i = 1; i < argc; i++) {
            showProperties(atoi(argv[i]));
            std::cout << "\n";
        }
    }

    return 0;
}