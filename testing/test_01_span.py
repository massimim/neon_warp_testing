from env_setup import update_pythonpath
update_pythonpath()

import os

import warp as wp

import wpne

import py_neon as ne
from py_neon import Index_3d
from py_neon.dense import dSpan


# Get the path of the current script
script_path = __file__
# Get the directory containing the script
script_dir = os.path.dirname(os.path.abspath(script_path))


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


@wp.func
def user_foo(idx: Index_3d):
    wp.neon_print(idx)


@wp.kernel
def neon_kernel_test(span: dSpan):
    # this is a Warp array which wraps the image data
    is_valid = wp.bool(True)
    myIdx = wp.NeonDenseSpan_set_idx(span, is_valid)
    if is_valid:
        user_foo(myIdx)


with wp.ScopedDevice("cuda:0"):

    grid = ne.dense.dGrid()
    span_device_id0_standard = grid.get_span(ne.Execution.device(),
                                             0,
                                             ne.DataView.standard())
    print(span_device_id0_standard)

    wp.launch(neon_kernel_test, dim=10, inputs=[span_device_id0_standard])

    wp.synchronize_device()
