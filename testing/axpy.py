from env_setup import update_pythonpath

update_pythonpath()

import os
import warp as wp
import wpne
import py_neon as ne
from py_neon import Index_3d
from py_neon.dense import dSpan
import typing
from typing import Any


@wp.kernel
def warp_AXPY(
        x: wp.array4d(dtype=Any),
        y: wp.array4d(dtype=Any),
        alpha: Any):
    i, j, k = wp.tid()
    for c in range(x.shape[0]):
        y[c, k, j, i] = x[c, k, j, i] + alpha * y[c, k, j, i]


@wpne.Container.factory
def get_AXPY(f_X, f_Y, alpha: Any):
    def axpy(loader: wpne.Loader):
        loader.declare_execution_scope(f_Y.get_grid())

        f_x = loader.get_read_handel(f_X)
        f_y = loader.get_read_handel(f_Y)

        @wp.func
        def foo(idx: typing.Any):
            #            wp.neon_print(idx)
            # wp.neon_print(f_read)
            for c in range(wp.neon_cardinality(f_x)):
                x = wp.neon_read(f_x, idx, c)
                y = wp.neon_read(f_y, idx, c)
                axpy = x + alpha * y
                wp.neon_write(f_y, idx, c, axpy)

            # print(value)

        loader.declare_kernel(foo)

    return axpy


def execution(nun_devs: int,
              num_card: int,
              dim: ne.Index_3d,
              dtype,
              container_runtime: wpne.Container.ContainerRuntime):
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

    dev_idx_list = list(range(nun_devs))
    bk = ne.Backend(runtime=ne.Backend.Runtime.stream,
                    dev_idx_list=dev_idx_list)

    grid = ne.dense.dGrid(bk, dim)
    field_X = grid.new_field(cardinality=num_card, dtype=dtype)
    field_Y = grid.new_field(cardinality=num_card, dtype=dtype)
    field_result = grid.new_field(cardinality=num_card, dtype=dtype)

    # Initialize two 4D arrays on the CPU
    cpu_warp_X = wp.empty(shape=(num_card, dim.z, dim.y, dim.x), dtype=dtype, device="cpu")
    cpu_warp_Y = wp.empty(shape=(num_card, dim.z, dim.y, dim.x), dtype=dtype, device="cpu")
    gpu_warp_X = wp.empty(shape=(num_card, dim.z, dim.y, dim.x), dtype=dtype, device="cuda")
    gpu_warp_Y = wp.empty(shape=(num_card, dim.z, dim.y, dim.x), dtype=dtype, device="cuda")

    alpha = dtype(2)

    def golden_axpy(index: ne.Index_3d, cardinality: int):
        def golden_axpy_input(idx: ne.Index_3d, cardinality: int):
            return (dtype(idx.x + idx.y + idx.z + cardinality), dtype(idx.x + idx.y + idx.z + cardinality + 1))

        x, y = golden_axpy_input(index, cardinality)
        return x, y, x + dtype(alpha) * y

    np_array_X = cpu_warp_X.numpy()
    np_array_Y = cpu_warp_Y.numpy()

    for zi in range(0, dim.z):
        for yi in range(0, dim.y):
            for xi in range(0, dim.x):
                for c in range(0, num_card):
                    idx = Index_3d(xi, yi, zi)
                    # print(f"idx {idx}")
                    x, y, result = golden_axpy(idx, c)
                    field_X.write(idx=idx,
                                  cardinality=c,
                                  newValue=dtype(x))
                    field_Y.write(idx=idx,
                                  cardinality=c,
                                  newValue=dtype(y))
                    field_result.write(idx,
                                       cardinality=c,
                                       newValue=dtype(result))

                    np_array_X[c, zi, yi, xi] = x
                    np_array_Y[c, zi, yi, xi] = y

    field_X.updateDeviceData(0)
    field_Y.updateDeviceData(0)
    field_result.updateDeviceData(0)

    cpu_warp_X = wp.array4d(np_array_X, dtype=dtype, device="cpu")
    cpu_warp_Y = wp.array4d(np_array_Y, dtype=dtype, device="cpu")

    wp.synchronize()

    gpu_warp_X = wp.array4d(cpu_warp_X, dtype=dtype, device="cuda")
    gpu_warp_Y = wp.array4d(cpu_warp_Y, dtype=dtype, device="cuda")
    wp.synchronize()

    axpy = get_AXPY(f_X=field_X, f_Y=field_Y, alpha=alpha)

    axpy.run(
        stream_idx=0,
        data_view=ne.DataView.standard(),
        container_runtime=container_runtime)

    wp.synchronize()
    wp.launch(
        warp_AXPY,
        dim=(dim.x, dim.y, dim.z),
        inputs=[
            gpu_warp_X,
            gpu_warp_Y,
            alpha
        ]
    )

    wp.synchronize()

    field_Y.updateHostData(0)
    wp.copy(cpu_warp_Y, gpu_warp_Y)

    wp.synchronize()
    np_array_Y = cpu_warp_Y.numpy()

    for zi in range(0, dim.z):
        for yi in range(0, dim.y):
            for xi in range(0, dim.x):
                for c in range(0, num_card):
                    idx = Index_3d(xi, yi, zi)
                    _, _, expected = golden_axpy(idx, c)
                    computed = field_Y.read(idx=idx,
                                            cardinality=c)
                    if expected != computed:
                        print(f'neon error at {xi},{yi},{zi} :{expected} cvs {computed}')
                    assert expected == computed

                    wp_res = np_array_Y[c, zi, yi, xi]
                    if wp_res != expected:
                        print(f'warp error at {xi},{yi},{zi} :{expected} cvs {wp_res}')
                    assert expected == computed


def gpu1_int(dimx, neon_ngpus:int=1):
    execution(nun_devs=neon_ngpus, num_card=1, dim=ne.Index_3d(dimx, dimx, dimx), dtype=int,
              container_runtime=wpne.Container.ContainerRuntime.neon)


def gpu1_float(dimx, neon_ngpus:int=1):
    execution(nun_devs=neon_ngpus, num_card=1, dim=ne.Index_3d(dimx, dimx, dimx), dtype=float,
              container_runtime=wpne.Container.ContainerRuntime.neon)


# def gpu1_float():
#     execution(nun_devs=1, num_card=1, dim=ne.Index_3d(10, 10, 10), dtype=ctypes.c_float,
#               container_runtime=wpne.Container.ContainerRuntime.neon)

if __name__ == "__main__":
    # gpu1_int()
    # gpu1_int()
    gpu1_float(100, 2)
