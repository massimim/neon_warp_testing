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

    # This is a set of compiled executable modules loaded by Warp.
    # When getting kernel hooks, we can retain the module references here
    # to prevent them from being unloaded prematurely.

    def __init__(self,
                 loading_lambda=None,
                 execution: py_neon.Execution = py_neon.Execution.device()):

        # creating a dummy loader to retrieve the grid for the thread scope
        self.execution = execution
        dummy_loader: Loader = Loader(execution=execution,
                                      gpu_id=0,
                                      data_view=py_neon.DataView.standard())

        try:
            self.py_neon: Py_neon = Py_neon()
        except Exception as e:
            self.handle: ctypes.c_uint64 = ctypes.c_void_p
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
            # Setting up the information of the Neon container for Neon runtime
            n_devices = self.backend.get_num_devices()  # rows
            self.retained_executable_modules = [set() for _ in range(n_devices)]

            n_data_views = 3  # columns
            # Create a NumPy array of object dtype
            self.k_2Darray = (ctypes.c_void_p * (n_data_views * n_devices))()

            for dev_idx in range(n_devices):
                for dw_idx in range(n_data_views):
                    k = self._get_kernel(execution=execution,
                                         gpu_id=dev_idx,
                                         data_view=py_neon.DataView.from_int(dw_idx),
                                         container_runtime=Container.ContainerRuntime.neon)
                    # using self.k for debugging
                    offset = dev_idx * n_data_views + dw_idx
                    dev_str = self.backend.get_device_name(dev_idx)
                    k_hook = self._get_kernel_hook(k, dev_str, dev_idx)
                    # print(f"hook {hex(k_hook)}, device {dev_idx}, data_view {dw_idx}")

                    self.k_2Darray[offset] = k_hook

            # debug = True
            # if debug:
            #     print("k_2Darray")
            #     for i in range(n_devices):
            #         for j in range(n_data_views):
            #             print(f"Device {i}, DataView {j} hook {hex(k_2Darray[i * n_data_views + j])}")

            self.container_handle = self.py_neon.handle_type(0)
            block_size = py_neon.Index_3d(128, 0, 0)
            self.py_neon.lib.warp_dgrid_container_new(ctypes.pointer(self.container_handle),
                                                      execution,
                                                      self.backend.cuda_driver_handle,
                                                      self.data_set.get_handle(),
                                                      self.k_2Darray,
                                                      block_size)

    def _get_kernel_hook(self, kernel, decvice_str, dev_idx):
        """
         decvice_str = "cuda:0"
        :param kernel:
        :param device_str:
        :return:
        """

        device = wp.get_device(decvice_str)
        # compile and load the executable module
        module_exec = kernel.module.load(device)
        if module_exec is None:
            raise RuntimeError(f"Failed to load module for kernel {kernel.key}")
        self.retained_executable_modules[dev_idx].add(module_exec)
        return module_exec.get_kernel_hooks(kernel).forward

    def help_load_api(self):
        # ------------------------------------------------------------------
        # backend_new
        self.py_neon.lib.warp_dgrid_container_new.argtypes = [ctypes.POINTER(self.py_neon.handle_type),
                                                              py_neon.Execution,
                                                              self.py_neon.handle_type,
                                                              self.py_neon.handle_type,
                                                              ctypes.POINTER(ctypes.c_void_p),
                                                              ctypes.POINTER(py_neon.Index_3d)]
        self.py_neon.lib.warp_dgrid_container_new.restype = None
        # ------------------------------------------------------------------
        # warp_container_delete
        self.py_neon.lib.warp_container_delete.argtypes = [self.py_neon.handle_type]
        self.py_neon.lib.warp_container_delete.restype = None
        # ------------------------------------------------------------------
        # warp_dgrid_container_run
        self.py_neon.lib.warp_container_run.argtypes = [self.py_neon.handle_type,
                                                        ctypes.c_int,
                                                        py_neon.DataView]
        self.py_neon.lib.warp_container_run.restype = None

        # TODOMATT get num devices
        # TODOMATT get device type

    def _get_kernel(self,
                    container_runtime: ContainerRuntime,
                    execution: py_neon.Execution,
                    gpu_id: int,
                    data_view: py_neon.DataView):
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
                # wp.printf("WARP my kernel - tid: %d %d %d\n", x, y, z)
                myIdx = wp.neon_set(span, x, y, z)
                # print("my kernel - myIdx: ")
                # wp.neon_print(myIdx)
                compute_lambda(myIdx)

            return kernel

        elif container_runtime == Container.ContainerRuntime.neon:
            @wp.kernel
            def kernel():
                is_active = wp.bool(False)
                myIdx = wp.neon_set(span, is_active)
                if is_active:
                    # print("NEON-RUNTIME kernel - myIdx: ")
                    # wp.neon_print(myIdx)
                    compute_lambda(myIdx)

            return kernel

    def _run_warp(
            self,
            stream_idx: int,
            data_view: py_neon.DataView):
        """
        Executing a container in the warp backend.
        :param stream_idx:
        :param data_view:
        :return:
        """
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
                kernel = self._get_kernel(
                    container_runtime=Container.ContainerRuntime.warp,
                    execution=self.execution,
                    gpu_id=dev_idx,
                    data_view=data_view)

                wp_kernel_dim = thread_space.to_wp_kernel_dim()
                wp.launch(kernel, dim=wp_kernel_dim, device=wp_device)
                # TODO@Max - WARNING - the following synchronization is temporary
                wp.synchronize_device(wp_device)

    def _run_neon(
            self,
            stream_idx: int,
            data_view: py_neon.DataView):
        self.py_neon.lib.warp_container_run(self.container_handle,
                                            stream_idx,
                                            data_view)

    def run(self,
            stream_idx: int,
            data_view: py_neon.DataView = py_neon.DataView.standard(),
            container_runtime: ContainerRuntime = ContainerRuntime.warp):
        if container_runtime == Container.ContainerRuntime.warp:
            self._run_warp(stream_idx=stream_idx,
                           data_view=data_view)
        elif container_runtime == Container.ContainerRuntime.neon:
            self._run_neon(stream_idx=stream_idx,
                           data_view=data_view)

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
