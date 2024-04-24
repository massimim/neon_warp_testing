import warp as wp
import wpne
import warp.config

import os
import py_neon as ne

# Get the path of the current script
script_path = __file__

# Get the directory containing the script
script_dir = os.path.dirname(os.path.abspath(script_path))

print(f"Directory containing the script: {script_dir}")


# print some info about an image
# @wp.function
# def neon_kernel_test(idx: wpne.DenseIndex):
#     # this is a Warp array which wraps the image data
#     wp.myPrint(idx)

# @wp.function
# def user_foo(idx:wpne.dIndex):
#     wp.myPrint(idx)

@wp.kernel
def neon_kernel_test(span: wpne.dSpan):
    # this is a Warp array which wraps the image data
    idx = wp.dSpan_set_and_validata(span)
    wp.myPrint(idx)


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
wpne.register()

with wp.ScopedDevice("cuda:0"):
    # print image info
    print("===== Image info:")
    idx = wpne.DenseIndex(1, 2, 33)

    grid = ne.DGrid()
    span_device_id0_standard = grid.get_span(ne.Execution.device,
                                             0,
                                             ne.Data_view.standard)
    print(span_device_id0_standard)

    wp.launch(neon_kernel_test, dim=10, inputs=[span_device_id0_standard])

wp.synchronize()
