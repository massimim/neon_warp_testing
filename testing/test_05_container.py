from env_setup import update_pythonpath

update_pythonpath()

import os

import warp as wp
from enum import Enum
import wpne
import py_neon as ne
from py_neon import Index_3d
from py_neon.dense import dSpan
import typing


class Loader:
    class ParsingTarget(Enum):
        grid = 0
        partitions = 1
        data_dependencies = 2

    def __init__(self,
                 execution: ne.Execution,
                 gpu_id: int,
                 data_view: ne.DataView):
        self.execution = execution
        self.gpu_id = gpu_id
        self.data_view = data_view

        self.kernel = None
        self.data_set = None

    def get_read_handel(self, data_set):
        partition = data_set.get_partition(
            self.execution,
            self.gpu_id,
            self.data_view)
        return partition

    def declare_execution_scope(self, grid):
        self.data_set = grid

    def _retrieve_grid(self):
        return self.data_set

    def declare_kernel(self, kernel):
        self.kernel = kernel

    def _retrieve_compute_lambda(self):
        return self.kernel


class Container:
    def __init__(self, loading_lambda=None):
        # creating a dummy loader to retrieve the grid for the thread scope
        loader: Loader = Loader(ne.Execution.host(),
                                0,
                                ne.DataView.standard())
        if loading_lambda is not None:
            self.loading_lambda = loading_lambda
            self.loading_lambda(loader)
            self.data_set = loader._retrieve_grid()
            return

        self.loading_lambda = None
        self.data_set = None

    def _get_kernel(self,
                    execution: ne.Execution,
                    gpu_id: int,
                    data_view: ne.DataView):
        span = self.data_set.get_span(execution=execution,
                                      dev_idx=gpu_id,
                                      data_view=data_view)
        loader: Loader = Loader(ne.Execution.host(),
                                0,
                                ne.DataView.standard())

        self.loading_lambda(loader)
        compute_lambda = loader._retrieve_compute_lambda()

        @wp.kernel
        def kernel():
             = wp.tid()
            wp.printf("From WP: x,y,z %d %d %d\n", x, y, z)
            is_valid = wp.bool(True)
            myIdx = wp.NeonDenseSpan_set_idx(span, x, y, z, is_valid)
            print("kernel - myIdx: ")
            wp.NeonDenseIdx_print(myIdx)

            if is_valid:
                compute_lambda(myIdx)

        return kernel

    def run(self,
            execution: ne.Execution,
            stream_idx: int,
            dataview: ne.DataView = ne.DataView.standard()):

        bk = self.data_set.get_backend()
        n_devices = bk.get_num_devices()
        wp_device_name: str = bk.get_warp_device_name()

        for dev_idx in range(n_devices):
            wp_device = f"{wp_device_name}:{dev_idx}"
            span = self.data_set.get_span(execution=execution,
                                          dev_idx=dev_idx,
                                          data_view=dataview)
            thread_space = span.get_thread_space()
            kernel = self._get_kernel(execution, dev_idx, dataview)
            wp_kernel_dim = thread_space.to_wp_kernel_dim()
            wp.launch(kernel, dim=(2, 1, 3), device=wp_device)
            # TODO@Max - WARNING - the following synchronization is temporary
            wp.synchronize_device(wp_device)

    # def _set_data(self, data_set):
    #     self.data_set = data_set
    #
    # def _set_loading_lambda(self, loading_lambda):
    #     self.loading_lambda = loading_lambda


def container_decorator(loading_lambda_generator):
    print(":container_decorator")

    def container_generator(*args, **kwargs):
        loading_lambda = loading_lambda_generator(*args, **kwargs)
        container = Container(loading_lambda=loading_lambda)
        return container

    return container_generator


@container_decorator
def declaring_my_contaier(field):
    def loading(loader: Loader):
        loader.declare_execution_scope(field.get_grid())
        f_read = loader.get_read_handel(field)

        @wp.func
        def foo(idx: typing.Any):
            wp.NeonDenseIdx_print(idx)
            # value = wp.NeonDensePartitionInt_read(f_read, idx, 0)
            print(int(33))

        loader.declare_kernel(foo)

    return loading


def _container_int():
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

    bk = ne.Backend(runtime=ne.Backend.Runtime.stream,
                    dev_idx_list=[0])

    grid = ne.dense.dGrid(bk, Index_3d(1, 1, 3))
    field = grid.new_field()

    my_contaier = declaring_my_contaier(field)
    my_contaier.run(ne.Execution.device(), 0, ne.DataView.standard())
    wp.synchronize()


if __name__ == "__main__":
    _container_int()
