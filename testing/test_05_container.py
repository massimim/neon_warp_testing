from env_setup import update_pythonpath

update_pythonpath()

import os
import warp as wp
import wpne
import py_neon as ne
from py_neon import Index_3d
import typing


@wpne.Container.factory
def get_solver_operator_container(field):
    def setup(loader: wpne.Loader):
        loader.declare_execution_scope(field.get_grid())

        f_read = loader.get_read_handel(field)

        @wp.func
        def foo(idx: typing.Any):
            wp.neon_print(idx)
            # wp.neon_print(f_read)
            value = wp.neon_read(f_read, idx, 0)
            cartesianIdx = wp.neon_global_idx(f_read, idx)
            extra = wp.neon_get_x(cartesianIdx) + wp.neon_get_y(cartesianIdx) + wp.neon_get_z(cartesianIdx)
            wp.printf("Position (%d %d %d) read %d extra %d\n", wp.neon_get_x(cartesianIdx), wp.neon_get_y(cartesianIdx), wp.neon_get_z(cartesianIdx), value, extra)
            value = value + extra
            wp.print(value)

            # value = value + int(idx.x)
            wp.neon_write(f_read, idx, 0, value)

            # print(value)

        loader.declare_kernel(foo)

    return setup


def test_container_int():
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
    field = grid.new_field(cardinality=1, dtype='int32')

    def set_value(idx: Index_3d):
        return idx.x + idx.y + idx.z

    for z in range(0, dim.z):
        for y in range(0, dim.y):
            for x in range(0, dim.x):
                idx = Index_3d(x, y, z)
                newValue = set_value(idx)
                print(f"Init@({x},{y},{z}): [value]{newValue} ")
                field.write(idx=idx,
                            cardinality=0,
                            newValue=newValue)

    field.updateDeviceData(0)
    wp.synchronize()

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

    field.updateHostData(0)
    wp.synchronize()

    for z in range(0, dim.z):
        for y in range(0, dim.y):
            for x in range(0, dim.x):
                idx = Index_3d(x, y, z)
                newValue = set_value(idx)
                newValue = newValue*3
                newValueRead = field.read(idx=idx,
                                          cardinality=0)
                different = (newValue ) - newValueRead
                if different != 0:
                    print(f"Error@({x},{y},{z}): [expected]{newValue} != {newValueRead}[read], {different}")

    pass


if __name__ == "__main__":
    test_container_int()
