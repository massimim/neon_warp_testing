from time import sleep

from env_setup import update_pythonpath
update_pythonpath()

import ctypes
import warp as wp
import wpne
import os

from py_neon import Index_3d
from py_neon.dense import dSpan

# Get the path of the current script
script_path = __file__

# Get the directory containing the script
script_dir = os.path.dirname(os.path.abspath(script_path))

print(f"Directory containing the script: {script_dir}")

def example():

    wp.config.mode = "debug"
    wp.config.llvm_cuda = False
    wp.config.verbose = True
    wp.verbose_warnings = True

    wp.init()

    wp.build.set_cpp_standard("c++17")
    wp.build.add_include_directory(script_dir)
    wp.build.add_preprocessor_macro_definition('NEON_WARP_COMPILATION')

    # It's a good idea to always clear the kernel cache when developing new native or codegen features
    wp.build.clear_kernel_cache()

    # !!! DO THIS BEFORE DEFINING/USING ANY KERNELS WITH CUSTOM TYPES
    wpne.init()


    # @wp.kernel
    # def index_kernel(idx: Index_3d):
    #     wp.neon_print(idx)


    def get_index_kernel(idx: Index_3d):
        @wp.kernel
        def myKernel():
            wp.print("JJJJJJJJJJ")
            wp.neon_print(idx)
        return myKernel

    @wp.kernel
    def span_kernel(span: dSpan):
        is_valid = wp.bool(True)
        idx = wp.neon_set(span, is_valid)
        if is_valid:
            wp.neon_print(idx)

    @wp.kernel
    def ciao_kernel():
        wp.print("ciao")


    # returns the CUfunction that can be launched with cuLaunchKernel()
    def get_kernel_hook(kernel, dev_str):
        device = wp.get_device(dev_str)
        module = kernel.module
        module.load(device)
        return module.get_kernel_hooks(kernel, device).forward



    # build and load Warp this module
    # wp.load_module("__main__", device=device)

    # Build and load the native lib that will launch kernels

    src_path = os.path.join(script_dir, "native", "test_native_launch.cpp")
    dll_path = os.path.join(script_dir, "native", "test_native_launch.so")

    # build if necessary
    if not os.path.exists(dll_path) or os.path.getmtime(dll_path) < os.path.getmtime(src_path):
        import subprocess

        print("\nBuilding C++ lib...")

        build_script_path = os.path.join(script_dir, "native", "build.sh")

        subprocess.run([build_script_path], check=True)

    # load
    dll = ctypes.CDLL(dll_path)

    dll.init.argtypes = []
    dll.init.restype = ctypes.c_bool
    dll.test_index_kernel.argtypes = [ctypes.c_void_p]
    dll.test_index_kernel.restype = None
    dll.test_span_kernel.argtypes = [ctypes.c_void_p]
    dll.test_span_kernel.restype = None

    # initialize native lib
    if not dll.init():
        raise RuntimeError("Failed to initialize library")

    # get kernel hooks (CUfunction) and launch them from native code

    print("\nLaunching from C++...")

    print()
    index_kernel = get_index_kernel(Index_3d(1, 2, 3))
    k_index = get_kernel_hook(index_kernel,"cuda:0")
    k_span = get_kernel_hook(span_kernel, "cuda:0")

    wp.launch(get_index_kernel(Index_3d(1,1,1)), dim=1, device=wp.get_device("cuda:0"))
    #
    print(f"index_kernel: {hex(k_index)}")
    dll.test_index_kernel(k_index)
    #
    # print()
    # dll.test_span_kernel(k_span)
    sleep(10)
if __name__ == "__main__":
    example()