from enum import Enum
import py_neon as ne


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
