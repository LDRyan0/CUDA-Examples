#include <iostream>

void showProperties(int devId) {
        std::cout << "Device: " << devId << "\n";;

        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, devId);

        std::cout << "Name: " << prop.name << "\n";
        std::printf("totalGlobalMem: %.1f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        std::printf("sharedMemPerBlock: %.1f MB\n", prop.sharedMemPerBlock / 1024.0);
        std::cout << "regsPerBlock: " << prop.regsPerBlock << "\n";
        std::cout << "warpSize: " << prop.warpSize << "\n";
        std::cout << "maxThreadsPerBlock: " << prop.maxThreadsPerBlock << "\n";
        for(int i = 0; i < 3; ++i)
            std::printf("maxThreadsDim[%d]: %d\n", i, prop.maxThreadsDim[i]);
        for(int i = 0; i < 3; ++i)
            std::printf("maxGridSize[%d]: %d\n", i, prop.maxGridSize[i]);
        std::printf("clockRate: %.1f GHz\n", prop.clockRate / (1024.0*1024.0));
        std::cout << "totalConstMem: " << prop.totalConstMem << "\n";
        std::cout << "capability: " << prop.major  << "." << prop.minor << "\n";
        std::cout << "multiProcessorCount: " << prop.multiProcessorCount << "\n";
        std::cout << "integrated: " << (prop.integrated ? "TRUE" : "FALSE") << "\n";
        std::cout << "concurrentKernels: " << (prop.concurrentKernels ? "TRUE" : "FALSE") << "\n";
        std::printf("memoryClockRate: %.1f GHz\n", prop.memoryClockRate / (1024.0*1024.0));
        std::printf("l2CacheSize: %.1f MB\n", prop.l2CacheSize / (1024.0 * 1024.0));
        std::cout << "maxThreadsPerMultiProcessor: " << prop.maxThreadsPerMultiProcessor << "\n";
        std::cout << "globalL1CacheSupported: " << (prop.globalL1CacheSupported ? "TRUE" : "FALSE") << "\n";
        std::cout << "localL1CacheSupported: " << (prop.localL1CacheSupported ? "TRUE" : "FALSE")<< "\n";
        std::printf("sharedMemPerMultiprocessor: %.1f kB\n", prop.sharedMemPerMultiprocessor / 1024.0);
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