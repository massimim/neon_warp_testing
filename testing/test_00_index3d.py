from env_setup import update_pythonpath
update_pythonpath()

import warp as wp
import warp.config
import wpne
import os

# Get the path of the current script
script_path = __file__

# Get the directory containing the script
script_dir = os.path.dirname(os.path.abspath(script_path))

print(f"Directory containing the script: {script_dir}")


# print some info about an image
@wp.kernel
def neon_kernel_test(idx: wpne.NeonDenseIdx):
    # this is a Warp array which wraps the image data
    wp.myPrint(idx)


wp.config.llvm_cuda = False
warp.config.verbose = True
wp.config.verbose = True
wp.verbose_warnings = True



wp.init()
wp.config.verbose = True
wp.build.set_cpp_standard("c++17")
wp.build.add_include_directory(script_dir)
wp.build.add_preprocessor_macro_definition('NEON_WARP_COMPILATION')

# It's a good idea to always clear the kernel cache when developing new native or codegen features
wp.build.clear_kernel_cache()

# !!! DO THIS BEFORE LOADING MODULES OR LAUNCHING KERNELS
wpne.init()

with wp.ScopedDevice("cuda:0"):
    # print image info
    print("===== Image info:")
    idx = wpne.NeonDenseIdx(1, 2, 33)
    wp.launch(neon_kernel_test, dim=1, inputs=[idx])

wp.synchronize()