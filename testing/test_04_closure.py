from env_setup import update_pythonpath
update_pythonpath()

import warp as wp

import wpne

import py_neon
from py_neon import Index_3d, DataView
from py_neon.dense import Span
from py_neon.dense.partition import PartitionInt

import os

def test_04_closure():

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


    def create_kernel():

        # not closure
        @wp.kernel
        def kernel():
            wp.neon_print(wp.NeonDenseIdx_create(11, 22, 33))

        return kernel


    def create_kernel_closure(value: Index_3d):

        # closure
        @wp.kernel
        def kernel():
            wp.neon_print(value)

        return kernel


    def create_fk():

        # not closure
        @wp.func
        def functional():
            wp.neon_print(wp.NeonDenseIdx_create(11, 22, 33))

        # not closure
        @wp.kernel
        def kernel():
            functional()

        return functional, kernel


    def create_fk_closure(value: Index_3d):

        # closure
        @wp.func
        def functional():
            wp.neon_print(value)

        # closure
        @wp.kernel
        def kernel():
            functional()

        return functional, kernel


    # manually generate unique functions and kernels
    class Generator:
        def __init__(self):
            self.count = 0

        def create_fk(self, value: Index_3d):

            def functional():
                wp.neon_print(value)

            f_key = f"{wp.codegen.make_full_qualified_name(functional)}_{self.count}"
            functional = wp.Function(functional, f_key, "")

            def kernel():
                functional()

            k_key = f"{wp.codegen.make_full_qualified_name(kernel)}_{self.count}"
            kernel = wp.Kernel(kernel, key=k_key)

            self.count += 1

            return functional, kernel


    # test whether capturing Python custom types is working
    def create_closure_all_types(idx: Index_3d, data_view: DataView, span: Span, partition: PartitionInt):

        # closure captures variables by value
        @wp.kernel
        def kernel():
            wp.neon_print(idx)
            wp.NeonDataView_print(data_view)
            wp.NeonDenseSpan_print(span)
            wp.neon_print(partition)

        return kernel


    with wp.ScopedDevice("cuda:0"):
        print("\n===== Test kernel =========================================================================")

        kernel1 = create_kernel()
        kernel2 = create_kernel()

        wp.launch(kernel1, dim=1, inputs=[])
        wp.launch(kernel2, dim=1, inputs=[])

        wp.synchronize_device()

        print("\n===== Test kernel closure =================================================================")

        kernel3 = create_kernel_closure(Index_3d(-1, -2, -3))
        kernel4 = create_kernel_closure(Index_3d(17, 42, 99))

        wp.launch(kernel3, dim=1, inputs=[])
        wp.launch(kernel4, dim=1, inputs=[])

        wp.synchronize_device()

        print("\n===== Test functional + kernel ============================================================")

        f1, k1 = create_fk()
        f2, k2 = create_fk()

        wp.launch(k1, dim=1, inputs=[])
        wp.launch(k2, dim=1, inputs=[])

        wp.synchronize_device()

        print("\n===== Test functional + kernel closures ===================================================")

        f3, k3 = create_fk_closure(Index_3d(-1, -2, -3))
        f4, k4 = create_fk_closure(Index_3d(17, 42, 99))

        wp.launch(k3, dim=1, inputs=[])
        wp.launch(k4, dim=1, inputs=[])

        wp.synchronize_device()

        print("\n===== Test aggregate kernel ===============================================================")

        @wp.kernel
        def aggregate_kernel():
            f1()
            f2()
            f3()
            f4()

        wp.launch(aggregate_kernel, dim=1, inputs=[])

        wp.synchronize_device()

        print("\n===== Test manual generator ===============================================================")

        generator = Generator()

        f1, k1 = generator.create_fk(Index_3d(-1, -2, -3))
        f2, k2 = generator.create_fk(Index_3d(17, 42, 99))
        wp.launch(k1, dim=1, inputs=[])
        wp.launch(k2, dim=1, inputs=[])

        wp.synchronize_device()

        print("\n===== Test all types ===============================================================")

        idx = Index_3d(3, 2, 1)
        data_view = DataView(DataView.Values.boundary)

        span = Span()
        span.dataView = DataView(DataView.Values.internal)
        span.z_ghost_radius = 17
        span.z_boundary_radius = 42
        span.max_z_in_domain = 99
        span.span_dim = Index_3d(2, 4, 6)

        grid = py_neon.dense.Grid()
        span_device_id0_standard = grid.get_span(py_neon.Execution.device(),
                                                 0,
                                                 py_neon.DataView.standard())
        # print(span_device_id0_standard)

        field = grid.new_field()
        partition = field.get_partition(py_neon.Execution.device(), 0, py_neon.DataView.standard())

        k = create_closure_all_types(idx, data_view, span, partition)

        wp.launch(k, dim=1)
        wp.synchronize_device()
