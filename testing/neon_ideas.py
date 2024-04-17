import neon as ne
import warp as wp

"""
 Field = [Partition, Partition, Partition, ..., Partition]
 Field used during initialization and io
 Partition used in device functions  
 """

## This would be the function the user writed to create a neon container.
def myAdd(a, b) -> ne.Container:
    grid = a.get_grig()

    # this function gets a loader object and returns a warp device function
    def loading(loader : ne.Loader):
        aP = loader(a, ne.map_op) # convert the field into a partition type
        bP = loader(b, ne.map_op) # convert the field into a partition type

        @wp.function
        def compute_fun(idx: grid.Idx):
            aP[idx, 0] + bP[idx, 0]
        return compute_fun

    c = grid.new_container(loading, a, b)

    return c


grid = ne.new_grid(...)
aField = grid.new_field(dtype=wp.float64, num_components=1)
bField = grid.new_field(dtype=wp.float64, num_components=1)
c: ne.Contianer = myAdd(aField, bField)
c.run(0)


class DeviceSet:
    def __init__(self, name: str, device_set: ne.DeviceSet, gridType: str):
        self.name = name
        self.numDevices = device_set
        self.gridType = gridType


class Grid:
    def new_container(self, loading_fun, A, B):
        # FOR EACH DEVICE p
        # 1. Convert A and B to Ap and Bp
        # 2. Get the Span_p from grid.
        # 3. Get CUDA grid and block size from grid for partition p
        # 4. Generate a wp kernel of type void (*)(Span, A, B) with a body like this:
        #       void kernel(Span s, A a, B b) {
        #           idx = s.get_idx(threadIdx.x, threadIdx.y, threadIdx.z)
        #           if idx.is_valid() {
        #               s[idx] = compute_fun(idx, a, b)
        #           }
        #       }
        cuda_launch_info_list = []
        kernel_params_list = []
        kernel_list = []
        for p in self.devices:
            aP = A.get_partition(p)
            aP = B.get_partition(p)
            spanP = self.get_span(p)
            kernel_params_list.append([spanP, Ap, Bp])
            grid_size = self.get_grid_size(p)
            block_size = self.get_block_size(p)
            cuda_launch_info_list.append([grid_size, block_size])
            # how can we generate the kernel?
            kernel = generate_kernel()
            kernel_list.append(kernel)
            pass
        c = ne.Container(cuda_launch_info_list,
                         kernel_params_list,
                         kernel_list)
        return c


class Container:
    def __init__(self, cuda_launch_info_list,
                 kernel_params_list,
                 kernel_list):
        self.cuda_launch_info_list = cuda_launch_info_list
        self.kernel_params_list = kernel_params_list
        self.kernel_list = kernel_list

    def run(self, stream):
        for idx, device_id in enumerate(self.devices):
            wp.set_device(device_id)
            stream = wp.Stream("cuda:0")
            with wp.ScopedStream(stream):
                wp.launch(kernel=self.kernel_list[idx],  # kernel to launch
                          # we would need direct access to cuda launch
                          cuda_launch_info=self.cuda_launch_info_list[i],  # number of threads
                          inputs=self.kernel_params_list[idx],  # parameters
                          device="cuda")
