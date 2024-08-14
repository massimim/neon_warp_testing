from time import sleep

from env_setup import update_pythonpath

update_pythonpath()

import os
import warp as wp
import wpne
import py_neon as ne
from py_neon import Index_3d
from py_neon.dense import dSpan
import typing


@wpne.Container.factory
def get_solver_operator_container(field):
    def setup(loader: wpne.Loader):
        loader.declare_execution_scope(field.get_grid())

        # f_read = loader.get_read_handel(field)

        @wp.func
        def foo(idx: typing.Any):
            wp.print("::::::::::::::::::::::::::::::::::::::")
            wp.neon_print(idx)
            # wp.neon_print(f_read)
            # value = wp.neon_read(f_read, idx, 0)
            # print(value)

        loader.declare_kernel(foo)

    return setup


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

    dim = Index_3d(1, 1, 3)
    grid = ne.dense.dGrid(bk, dim)
    field = grid.new_field(cardinality=1)

    # for z in range(0, dim.z):
    #     for y in range(0, dim.y):
    #         for x in range(0, dim.x):
    #             field.write(idx=Index_3d(x, y, z),
    #                         cardinality=0,
    #                         newValue=x + y + z)

    # field.updateDeviceData(0)

    solver_operator = get_solver_operator_container(field)
    solver_operator.run(
        stream_idx=0,
        data_view=ne.DataView.standard(),
        container_runtime=wpne.Container.ContainerRuntime.warp)
    print('=====================')
    solver_operator.run(
        stream_idx=0,
        data_view=ne.DataView.standard(),
        container_runtime=wpne.Container.ContainerRuntime.neon)
    field.updateDeviceData(0)
    wp.synchronize()
    pass


if __name__ == "__main__":
    _container_int()
