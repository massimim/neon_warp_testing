#include <cstdio>
#include <vector>

#include <dlfcn.h>

#include <cuda.h>
#include <cudaTypedefs.h>


// Warp builtins
#include <warp/native/builtin.h>

// Neon
#include <Neon/domain/dGrid.h>

using NeonDenseIdx = ::Neon::domain::details::dGrid::dIndex;


// the minimum CUDA version required from the driver
#define MIN_DRIVER_VERSION 11040

#if CUDA_VERSION < 12000
static PFN_cuGetProcAddress_v11030 pfn_cuGetProcAddress;
#else
static PFN_cuGetProcAddress_v12000 pfn_cuGetProcAddress;
#endif
static PFN_cuGetErrorName_v6000 pfn_cuGetErrorName;
static PFN_cuGetErrorString_v6000 pfn_cuGetErrorString;
static PFN_cuCtxSynchronize_v2000 pfn_cuCtxSynchronize;
static PFN_cuLaunchKernel_v4000 pfn_cuLaunchKernel;

#define check_cu(code) (check_cu_result(code, __FUNCTION__, __FILE__, __LINE__))

bool check_cu_result(CUresult result, const char* func, const char* file, int line)
{
    if (result == CUDA_SUCCESS)
        return true;

    const char* errString = NULL;
    if (pfn_cuGetErrorString)
        pfn_cuGetErrorString(result, &errString);

    if (errString)
        fprintf(stderr, "CUDA error %u: %s (in function %s, %s:%d)\n", unsigned(result), errString, func, file, line);
    else
        fprintf(stderr, "CUDA error %u (in function %s, %s:%d)\n", unsigned(result), func, file, line);

    return false;
}

static bool get_driver_entry_point(const char* name, void** pfn)
{
    if (!pfn_cuGetProcAddress || !name || !pfn)
        return false;

#if CUDA_VERSION < 12000
    CUresult r = pfn_cuGetProcAddress(name, pfn, MIN_DRIVER_VERSION, CU_GET_PROC_ADDRESS_DEFAULT);
#else
    CUresult r = pfn_cuGetProcAddress(name, pfn, MIN_DRIVER_VERSION, CU_GET_PROC_ADDRESS_DEFAULT, NULL);
#endif

    if (r != CUDA_SUCCESS)
    {
        fprintf(stderr, "CUDA error: Failed to get driver entry point '%s' (CUDA error %u)\n", name, unsigned(r));
        return false;
    }

    return true;
}

extern "C" bool init()
{
    // initialize CUDA API
    void* cuda_lib = dlopen("libcuda.so", RTLD_NOW);
    if (cuda_lib == NULL)
    {
        // WSL and possibly other systems might require the .1 suffix
        cuda_lib = dlopen("libcuda.so.1", RTLD_NOW);
        if (cuda_lib == NULL)
        {
            fprintf(stderr, "Warp CUDA error: Could not open libcuda.so.\n");
            return false;
        }
    }

    pfn_cuGetProcAddress = (PFN_cuGetProcAddress)dlsym(cuda_lib, "cuGetProcAddress");
    if (!pfn_cuGetProcAddress)
    {
        fprintf(stderr, "Warp CUDA error: Failed to get function cuGetProcAddress\n");
        return false;
    }

    get_driver_entry_point("cuGetErrorString", &(void*&)pfn_cuGetErrorString);
    get_driver_entry_point("cuGetErrorName", &(void*&)pfn_cuGetErrorName);
    get_driver_entry_point("cuLaunchKernel", &(void*&)pfn_cuLaunchKernel);
    get_driver_entry_point("cuCtxSynchronize", &(void*&)pfn_cuCtxSynchronize);

    // initialize Neon
    Neon::init();

    return true;
}

extern "C" void test_index_kernel(void* kernel)
{
    printf("==== Index let's goooo =======================================\n");

    int n = 1;

    // Warp launch bounds
    wp::launch_bounds_t bounds;
    bounds.ndim = 1;
    bounds.shape[0] = n;
    bounds.size = n;

    // index
    NeonDenseIdx index(17, 42, 99);

    // kernel args
    std::vector<void*> args;
    args.push_back(&bounds);
    args.push_back(&index);

    int block_dim = 256;
    int grid_dim = (n + block_dim - 1) / block_dim;

    // let's go
    check_cu(pfn_cuLaunchKernel(
        (CUfunction)kernel,
        grid_dim, 1, 1,
        block_dim, 1, 1,
        0, nullptr,
        args.data(), 0));

    check_cu(pfn_cuCtxSynchronize());
}

extern "C" void test_span_kernel(void* kernel)
{
    printf("==== Span let's goooo =======================================\n");

    Neon::Backend bk(1, Neon::Runtime::stream);
    Neon::index_3d dim(10, 10, 10);
    Neon::domain::Stencil d3q19 = Neon::domain::Stencil::s19_t(false);
    Neon::dGrid grid(bk, dim, [](Neon::index_3d const& /*idx*/) { return true; }, d3q19);

    auto& span = grid.getSpan(Neon::Execution::device, 0, Neon::DataView::STANDARD);

    int n = 10;

    // Warp launch bounds
    wp::launch_bounds_t bounds;
    bounds.ndim = 1;
    bounds.shape[0] = n;
    bounds.size = n;

    // kernel args
    std::vector<void*> args;
    args.push_back(&bounds);
    args.push_back((void*)&span);

    int block_dim = 256;
    int grid_dim = (n + block_dim - 1) / block_dim;

    // let's go
    check_cu(pfn_cuLaunchKernel(
        (CUfunction)kernel,
        grid_dim, 1, 1,
        block_dim, 1, 1,
        0, nullptr,
        args.data(), 0));

    check_cu(pfn_cuCtxSynchronize());
}
