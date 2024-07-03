from env_setup import update_pythonpath
update_pythonpath()

import os

import warp as wp

import wpne
import py_neon as ne
from py_neon import Index_3d
from py_neon.dense import Span
from py_neon.dense.partition import PartitionInt

def test_03_field_int():

    # Get the path of the current script
    script_path = __file__
    # Get the directory containing the script
    script_dir = os.path.dirname(os.path.abspath(script_path))


    def conainer_kernel_generator(field):
        partition = field.get_partition(ne.Execution.device(), 0, ne.DataView.standard())
        print(f"?????? partition {id(partition)}, {type(partition)}")
        # from wpne.dense.partition import NeonDensePartitionInt
        # print(f"?????? NeonDensePartitionInt {id(NeonDensePartitionInt)}, {type(NeonDensePartitionInt)}, {partition.get_my_name()}")

        @wp.func
        def user_foo(idx: Index_3d):
            wp.NeonDenseIdx_print(idx)
            value = wp.NeonDensePartitionInt_read(partition, idx, 0)
            print(33)

        @wp.kernel
        def neon_kernel_test(span: Span):
            is_valid = wp.bool(True)
            myIdx = wp.NeonDenseSpan_set_idx(span, is_valid)
            if is_valid:
                user_foo(myIdx)

        return neon_kernel_test


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

    with wp.ScopedDevice("cuda:0"):

        grid = ne.dense.Grid()
        span_device_id0_standard = grid.get_span(ne.Execution.device(),
                                                 0,
                                                 ne.DataView.standard())
        print(span_device_id0_standard)

        field = grid.new_field()

        container = conainer_kernel_generator(field)
        wp.launch(container, dim=1, inputs=[span_device_id0_standard])

    wp.synchronize()


test_03_field_int()
