import nvtx
import warp as wp

import py_neon as ne
from .loader import Loader


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
        loader: Loader = Loader(execution=execution,
                                gpu_id=gpu_id,
                                data_view=data_view)

        self.loading_lambda(loader)
        compute_lambda = loader._retrieve_compute_lambda()

        @wp.kernel
        def kernel():
            x, y, z = wp.tid()
            wp.printf("kernel - tid: %d %d %d\n", x, y, z)
            myIdx = wp.neon_set(span, x, y, z)
            print("kernel - myIdx: ")
            wp.neon_print(myIdx)
            compute_lambda(myIdx)

        return kernel

    def run(self,
            execution: ne.Execution,
            stream_idx: int,
            data_view: ne.DataView = ne.DataView.standard()):
        with nvtx.annotate("wpne-container", color="green"):
            bk = self.data_set.get_backend()
            n_devices = bk.get_num_devices()
            wp_device_name: str = bk.get_warp_device_name()

            for dev_idx in range(n_devices):
                wp_device = f"{wp_device_name}:{dev_idx}"
                span = self.data_set.get_span(execution=execution,
                                              dev_idx=dev_idx,
                                              data_view=data_view)
                thread_space = span.get_thread_space()
                kernel = self._get_kernel(execution, dev_idx, data_view)
                wp_kernel_dim = thread_space.to_wp_kernel_dim()
                wp.launch(kernel, dim=wp_kernel_dim, device=wp_device)
                # TODO@Max - WARNING - the following synchronization is temporary
                wp.synchronize_device(wp_device)

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
