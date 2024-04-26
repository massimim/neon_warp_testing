import os

import warp as wp
import warp.config

import wpne
import py_neon as ne

import sys

# Get the path of the current script
script_path = __file__

# Get the directory containing the script
script_dir = os.path.dirname(os.path.abspath(script_path))

print(f"Directory containing the script: {script_dir}")


@wp.func
def user_foo(idx: wpne.Dense_idx):
    # this is a Warp array which wraps the image data
    wp.myPrint(idx)

@wp.kernel
def neon_kernel_test(span: wpne.Dense_span):
    # this is a Warp array which wraps the image data
    myIdx = wp.Dense_span_set_and_validata(span)
    user_foo(myIdx)
    # user_foo(idx)


wp.config.llvm_cuda = False
warp.config.verbose = True
wp.config.verbose = True
wp.verbose_warnings = True

wp.init()
wp.config.verbose = True
wp.build.set_cpp_standard("c++17")
wp.build.add_include_directory(script_dir)
wp.build.add_preprocessor_macro_definition('NEON_WARP_COMPILATION')

# It's a good idea to always clear the kernel cache when developing new native or codegen features
wp.build.clear_kernel_cache()

# !!! DO THIS BEFORE LOADING MODULES OR LAUNCHING KERNELS
wpne.init()

with wp.ScopedDevice("cuda:0"):
    # print image info
    print("===== Image info:")
    idx = wpne.Dense_idx(1, 2, 33)

    grid = wpne.Dense_grid()
    span_device_id0_standard = grid.get_span(ne.Execution.device(),
                                             0,
                                             ne.Data_view.standard)
    print(span_device_id0_standard)

    wp.launch(neon_kernel_test, dim=10, inputs=[span_device_id0_standard])

wp.synchronize()
