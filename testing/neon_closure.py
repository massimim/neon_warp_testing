from env_setup import update_pythonpath
update_pythonpath()

import warp as wp

import wpne
from wpne import NeonDenseIdx

import os

# Get the path of the current script
script_path = __file__

# Get the directory containing the script
script_dir = os.path.dirname(os.path.abspath(script_path))

print(f"Directory containing the script: {script_dir}")


# wp.config.llvm_cuda = False
# wp.config.verbose = True
# wp.verbose_warnings = True

wp.init()
wp.build.set_cpp_standard("c++17")
wp.build.add_include_directory(script_dir)
wp.build.add_preprocessor_macro_definition('NEON_WARP_COMPILATION')

# It's a good idea to always clear the kernel cache when developing new native or codegen features
wp.build.clear_kernel_cache()

# !!! DO THIS BEFORE LOADING MODULES OR LAUNCHING KERNELS
wpne.init()


def create_kernel():

    # not closure
    @wp.kernel
    def kernel():
        wp.myPrint(wp.NeonDenseIdx_(11, 22, 33))
    
    return kernel


def create_kernel_closure(value: NeonDenseIdx):

    # closure
    @wp.kernel
    def kernel():
        wp.myPrint(value)
    
    return kernel


def create_fk():

    # not closure
    @wp.func
    def functional():
        wp.myPrint(wp.NeonDenseIdx_(11, 22, 33))

    # not closure
    @wp.kernel
    def kernel():
        functional()

    return functional, kernel


def create_fk_closure(value: NeonDenseIdx):

    # closure
    @wp.func
    def functional():
        wp.myPrint(value)

    # closure
    @wp.kernel
    def kernel():
        functional()

    return functional, kernel


# manually generate unique functions and kernels
class Generator:
    def __init__(self):
        self.count = 0

    def create_fk(self, value: NeonDenseIdx):

        def functional():
            wp.myPrint(value)

        f_key = f"{wp.codegen.make_full_qualified_name(functional)}_{self.count}"
        functional = wp.Function(functional, f_key, "")

        def kernel():
            functional()

        k_key = f"{wp.codegen.make_full_qualified_name(kernel)}_{self.count}"
        kernel = wp.Kernel(kernel, key=k_key)

        self.count += 1

        return functional, kernel


with wp.ScopedDevice("cuda:0"):
    print("\n===== Test kernel =========================================================================")

    kernel1 = create_kernel()
    kernel2 = create_kernel()

    wp.launch(kernel1, dim=1, inputs=[])
    wp.launch(kernel2, dim=1, inputs=[])

    wp.synchronize_device()

    print("\n===== Test kernel closure =================================================================")

    kernel3 = create_kernel_closure(NeonDenseIdx(-1, -2, -3))
    kernel4 = create_kernel_closure(NeonDenseIdx(17, 42, 99))

    wp.launch(kernel3, dim=1, inputs=[])
    wp.launch(kernel4, dim=1, inputs=[])

    wp.synchronize_device()

    print("\n===== Test functional + kernel ============================================================")

    f1, k1 = create_fk()
    f2, k2 = create_fk()

    wp.launch(k1, dim=1, inputs=[])
    wp.launch(k2, dim=1, inputs=[])

    wp.synchronize_device()

    print("\n===== Test functional + kernel closures ===================================================")

    f3, k3 = create_fk_closure(NeonDenseIdx(-1, -2, -3))
    f4, k4 = create_fk_closure(NeonDenseIdx(17, 42, 99))

    wp.launch(k3, dim=1, inputs=[])
    wp.launch(k4, dim=1, inputs=[])

    wp.synchronize_device()

    print("\n===== Test aggregate kernel ===============================================================")

    @wp.kernel
    def aggregate_kernel():
        f1()
        f2()
        f3()
        f4()

    wp.launch(aggregate_kernel, dim=1, inputs=[])

    wp.synchronize_device()

    print("\n===== Test manual generator ===============================================================")

    generator = Generator()

    f1, k1 = generator.create_fk(NeonDenseIdx(-1, -2, -3))
    f2, k2 = generator.create_fk(NeonDenseIdx(17, 42, 99))
    wp.launch(k1, dim=1, inputs=[])
    wp.launch(k2, dim=1, inputs=[])

    wp.synchronize_device()
