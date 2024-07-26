import ctypes
from enum import Enum

import nvtx
import warp as wp

import py_neon
from py_neon import Py_neon
from .loader import Loader


class Container:
    # define an enum class
    class ContainerRuntime(Enum):
        warp = 1
        neon = 2

    def __init__(self,
                 loading_lambda=None,
                 execution: py_neon.Execution = py_neon.Execution.device()):

        # creating a dummy loader to retrieve the grid for the thread scope
        dummy_loader: Loader = Loader(execution,
                                      0,
                                      py_neon.DataView.standard())

        try:
            self.py_neon: Py_neon = Py_neon()
        except Exception as e:
            self.handle: ctypes.c_uint64 = ctypes.c_uint64(0)
            raise Exception('Failed to initialize PyNeon: ' + str(e))
        self.help_load_api()

        self.loading_lambda = None
        self.data_set = None
        self.backend = None

        if loading_lambda is not None:
            self.loading_lambda = loading_lambda
            self.loading_lambda(dummy_loader)
            self.data_set = dummy_loader._retrieve_grid()
            self.backend = self.data_set.get_backend()

            n_devices = self.backend.get_num_devices()  # rows
            n_data_views = 3  # columns
            # Create a NumPy array of object dtype
            k_2Darray = (ctypes.c_void_p * (n_data_views * n_devices))()


            for dev_idx in range(n_devices):
                for dw_idx in range(n_data_views):
                    k = self._get_kernel(execution, dev_idx, py_neon.DataView.from_int(dw_idx), self.backend)
                    offset = dev_idx * n_data_views + dw_idx
                    k_2Darray[offset] = ctypes.pointer(k)



            self.container_handle = self.py_neon.handle_type(0)
            block_size = py_neon.Index_3d(0, 0, 0)
            self.py_neon.lib.warp_dgrid_container_new(ctypes.byref(self.container_handle),
                                                      execution,
                                                      self.backend.cuda_driver_handle,
                                                      self.data_set.get_handle(),
                                                      k_2Darray,
                                                      block_size)

    def help_load_api(self):
        # ------------------------------------------------------------------
        # backend_new
        self.py_neon.lib.warp_dgrid_container_new.argtypes = [self.py_neon.handle_type,
                                                              py_neon.Execution,
                                                              self.py_neon.handle_type,
                                                              self.py_neon.handle_type,
                                                              ctypes.c_void_p,
                                                              ctypes.c_void_p,
                                                              ctypes.c_void_p,
                                                              ctypes.POINTER(py_neon.Index_3d)]
        self.py_neon.lib.warp_dgrid_container_new.restype = None
        # ------------------------------------------------------------------
        # warp_container_delete
        self.py_neon.lib.dBackend_delete.argtypes = [self.py_neon.handle_type]
        self.py_neon.lib.dBackend_delete.restype = None
        # ------------------------------------------------------------------
        # warp_dgrid_container_new
        self.py_neon.lib.warp_dgrid_container_run.argtypes = [self.py_neon.handle_type,
                                                              ctypes.c_int,
                                                              py_neon.DataView]
        self.py_neon.lib.warp_dgrid_container_run.restype = None

        # TODOMATT get num devices
        # TODOMATT get device type

    def _get_kernel(self,
                    container_runtime: ContainerRuntime,
                    execution: py_neon.Execution,
                    gpu_id: int,
                    data_view: py_neon.DataView,
                    runtime):
        span = self.data_set.get_span(execution=execution,
                                      dev_idx=gpu_id,
                                      data_view=data_view)
        loader: Loader = Loader(execution=execution,
                                gpu_id=gpu_id,
                                data_view=data_view)

        self.loading_lambda(loader)
        compute_lambda = loader._retrieve_compute_lambda()

        if container_runtime == Container.ContainerRuntime.warp:
            @wp.kernel
            def kernel():
                x, y, z = wp.tid()
                wp.printf("kernel - tid: %d %d %d\n", x, y, z)
                myIdx = wp.neon_set(span, x, y, z)
                print("kernel - myIdx: ")
                wp.neon_print(myIdx)
                compute_lambda(myIdx)

            return kernel

        elif container_runtime == Container.ContainerRuntime.neon:
            @wp.kernel
            def kernel():
                wp.printf("kernel - tid: %d %d %d\n", x, y, z)
                myIdx = wp.neon_set(span)
                print("kernel - myIdx: ")
                wp.neon_print(myIdx)
                compute_lambda(myIdx)

            return kernel

    def run(self,
            stream_idx: int,
            data_view: py_neon.DataView = py_neon.DataView.standard(),
            container_runtime: ContainerRuntime = ContainerRuntime.warp):
        if container_runtime == Container.ContainerRuntime.warp:
            with nvtx.annotate("wpne-container", color="green"):
                bk = self.data_set.get_backend()
                n_devices = bk.get_num_devices()
                wp_device_name: str = bk.get_warp_device_name()

                for dev_idx in range(n_devices):
                    wp_device = f"{wp_device_name}:{dev_idx}"
                    span = self.data_set.get_span(execution=self.execution,
                                                  dev_idx=dev_idx,
                                                  data_view=data_view)
                    thread_space = span.get_thread_space()
                    kernel = self._get_kernel(self.execution, dev_idx, data_view)
                    wp_kernel_dim = thread_space.to_wp_kernel_dim()
                    wp.launch(kernel, dim=wp_kernel_dim, device=wp_device)
                    # TODO@Max - WARNING - the following synchronization is temporary
                    wp.synchronize_device(wp_device)
        elif container_runtime == Container.ContainerRuntime.neon:
            self.py_neon.lib.warp_dgrid_container_run(self.container_handle,
                                                      stream_idx,
                                                      data_view)

    # def _set_data(self, data_set):
    #     self.data_set = data_set
    #
    # def _set_loading_lambda(self, loading_lambda):
    #     self.loading_lambda = loading_lambda
    @staticmethod
    def factory(loading_lambda_generator):
        def container_generator(*args, **kwargs):
            loading_lambda = loading_lambda_generator(*args, **kwargs)
            container = Container(loading_lambda=loading_lambda)
            return container

        return container_generator
