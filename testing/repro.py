import warp as wp

wp.init()
wp.build.clear_kernel_cache()

@wp.func
def foo():
    print(42)

@wp.kernel
def k1():
    foo()

# wp.launch(k1, dim=1, inputs=[])

@wp.func
def foo():
    print(17)

@wp.kernel
def k2():
    foo()

wp.launch(k1, dim=1, inputs=[])
wp.launch(k2, dim=1, inputs=[])

wp.synchronize_device()
