# neon_warp_testing

The following lines reproduce the issue when running the mres with warp.

```bash
./clean.sh; ./clone.sh ssh; ./neon.sh build; ./warp.sh build; source env.sh export; ./xlb.sh mres ssh
cd XLB/examples/performance/
python3 ./mlups_3d_multires_solver.py 5 1000000 neon fp32/fp32 -
```

The detected error is the following:
```
[13:35:06] Neon: Exception thrown at 
Line 37 File /home/max/repos/test/neon_warp_testing/neon/libNeonPy/include/Neon/py/CudaDriver.h Function void Neon::py::CudaDriver::check_cuda_res(const CUresult&, const String&) [with String = char [15]; CUresult = cudaError_enum] 
[13:35:06] Neon: CudaDriverEntryPoint: cuLaunchKernel failed with 
701 CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES
too many resources requested for launch
terminate called after throwing an instance of 'Neon::NeonException'
  what():  cuLaunchKernel failed with 
701 CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES
too many resources requested for launch
Aborted (core dumped)

```