import ctypes
from enum import Enum

import py_neon
import py_neon as ne


class Loader:

    def __init__(self,
                 execution: ne.Execution,
                 gpu_id: int,
                 data_view: ne.DataView):
        self.execution = execution
        self.gpu_id = gpu_id
        self.data_view = data_view

        self.kernel = None
        self.data_set = None

    def help_load_api(self):
        # ------------------------------------------------------------------
        # warp_loader_new
        self.py_neon.lib.warp_loader_new.argtypes = [ctypes.POINTER(self.py_neon.handle_type),
                                                     ctypes.POINTER(self.py_neon.handle_type), # the container pointer
                                                     py_neon.Execution,
                                                     py_neon.DataView,
                                                     ctypes.c_int]
        self.py_neon.lib.warp_dgrid_container_new.restype = None
        # ------------------------------------------------------------------
        # warp_loader_new
        self.py_neon.lib.warp_loader_delete.argtypes = [ctypes.POINTER(self.py_neon.handle_type),]
        self.py_neon.lib.warp_loader_delete.restype = None

    # def get_loader_handel(self, data_set):
    #     self.
    #     return partition

    def get_read_handel(self, data_set):
        partition = data_set.get_partition(
            self.execution,
            self.gpu_id,
            self.data_view)
        return partition

    def get_write_read(self, data_set):
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
