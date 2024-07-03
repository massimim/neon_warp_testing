from env_setup import update_pythonpath
update_pythonpath()

import warp as wp
import wpne
import os

from py_neon import DataView

def test_02_data_view():


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
    def print_kernel(a: DataView, b: DataView, c: DataView):
        wp.NeonDataView_print(a)
        wp.NeonDataView_print(b)
        wp.NeonDataView_print(c)


    with wp.ScopedDevice("cuda:0"):

        d0 = DataView(DataView.Values.standard)
        d1 = DataView(DataView.Values.internal)
        d2 = DataView(DataView.Values.boundary)

        wp.launch(print_kernel, dim=1, inputs=[d0, d1, d2])

        wp.synchronize_device()

    print("Done.")


test_02_data_view()
