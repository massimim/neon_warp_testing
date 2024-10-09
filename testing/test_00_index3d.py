from env_setup import update_pythonpath
update_pythonpath()

import warp as wp
import wpne
import os

from py_neon import Index_3d

def test_00_index3d():
    # Get the path of the current script
    script_path = __file__

    # Get the directory containing the script
    script_dir = os.path.dirname(os.path.abspath(script_path))

    print(f"Directory containing the script: {script_dir}")


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


    @wp.kernel
    def index_print_kernel(idx: Index_3d):
        wp.neon_print(idx)


    @wp.kernel
    def index_create_kernel():
        idx = wp.neon_idx_3d(17, 42, 99)
        wp.neon_print(idx)


    with wp.ScopedDevice("cuda:0"):
        # pass index to a kernel
        idx = Index_3d(11, 22, 33)
        wp.launch(index_print_kernel, dim=1, inputs=[idx])

        # create index in a kernel
        wp.launch(index_create_kernel, dim=1, inputs=[])

        wp.synchronize_device()

if __name__ == "__main__":
    test_00_index3d()