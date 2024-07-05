from env_setup import update_pythonpath

update_pythonpath()

import os

import warp as wp

import wpne
import py_neon as ne
from py_neon import Index_3d
from py_neon.dense import dSpan


class Loader:
    def __init__(self, gpu_id):
        self.gpu_id = gpu_id
        self.execution = None
        self.data_view = None

    def _set_execution(self, execution: ne.Execution):
        self.execution = execution

    def _set_data_view(self, data_view: ne.DataView):
        self.data_view = data_view

    def get_read_handel(self, data_set):
        partition = data_set.get_partition(
            self.execution,
            self.gpu_id,
            self.data_view)
        return partition


class Container:
    def __init__(self,
                 data_set=None,
                 loading_lambda=None):
        self.data_set = data_set
        self.loading_lambda = loading_lambda

    def _get_kernel(self, gpu_idx: int, dataview):
        span = self.data_set.get_span(gpu_idx, dataview)
        loader = Loader(gpu_idx)
        self.loading_lambda(loader)
        compute_lambda = loader.get_compute_lambda(loader)

        @wp.kernel
        def kernel():
            is_valid = wp.bool(True)
            myIdx = wp.NeonDenseSpan_set_idx(span, is_valid)
            if is_valid:
                compute_lambda(myIdx)

        return kernel

    def run(self, stream_idx: int, dataview: ne.DataView = ne.DataView.standard()):
        kernel = self._get_kernel(stream_idx, dataview)
        wp.launch(kernel, dim=1, inputs=[stream_idx])

    # def _set_data(self, data_set):
    #     self.data_set = data_set
    #
    # def _set_loading_lambda(self, loading_lambda):
    #     self.loading_lambda = loading_lambda


def _field_int():
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

    def container_decorator(loading_lambda_generator):
        def container_generator(*args, **kwargs):
            try:
                target_grid = kwargs['container_grid']
            except:
                # throw expection
                raise 'continer_grid parameter is missing'
            loading_lambda = loading_lambda_generator(*args, **kwargs)
            container = Container(data_set=target_grid,
                                  loading_lambda=loading_lambda)
            return container

    @container_decorator
    def user_code(field, container_grid):
        def loading(loader: Loader):
            f_read = loader.get_read_handel(field)

            @wp.func
            def foo(idx: wp.any):
                wp.NeonDenseIdx_print(idx)
                value = wp.NeonDensePartitionInt_read(f_read, idx, 0)
                print(33)

            loader.set_user_kernel(foo)

        return loading

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
    dev_idx = 0
    with wp.ScopedDevice(f"cuda:{dev_idx}"):
        bk = ne.Backend(runtime=ne.Backend.Runtime.stream,
                        dev_idx_list=[dev_idx])

        grid = ne.dense.dGrid(bk, Index_3d(10, 10, 10))
        span_device_id0_standard = grid.get_span(ne.Execution.device(),
                                                 0,
                                                 ne.DataView.standard())
        print(span_device_id0_standard)

        field = grid.new_field()

        container = conainer_kernel_generator(field)
        wp.launch(container, dim=1, inputs=[span_device_id0_standard])

    wp.synchronize()


if __name__ == "__main__":
    _field_int()
