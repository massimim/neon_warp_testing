import ctypes
import os
import warp as wp


class DenseIndex:
    # # define variables accessible in kernels (e.g., coord.x)
    vars = {
        "x": wp.codegen.Var("x", int),
        "y": wp.codegen.Var("y", int),
        "z": wp.codegen.Var("z", int),
    }

    # struct that corresponds to the native Coord type
    # - used when packing arguments for kernels (pass-by-value)
    # - binary layout of fields must match native type
    class _type_(ctypes.Structure):
        _fields_ = [
            ("x", ctypes.c_int),
            ("y", ctypes.c_int),
            ("z", ctypes.c_int),
        ]

        def __init__(self, coord):
            self.x = coord.x
            self.y = coord.y
            self.z = coord.z

    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z

    # HACK: used when packing kernel argument as `arg_type._type_(value.value)` in `pack_arg()` during `wp.launch()`
    @property
    def value(self):
        return self


def _add_header(path):
    include_directive = f"#include \"{path}\"\n"
    # add this header for all native modules
    wp.codegen.cpu_module_header += include_directive
    wp.codegen.cuda_module_header += include_directive


def _register_headers():
    include_path = os.path.abspath(os.path.dirname(__file__))
    _add_header(f"{include_path}/neon_warp.h")


def _register_builtins():
    # Coord constructor
    wp.context.add_builtin(
        "DenseIndex_",
        input_types={"x": int, "y": int, "z": int},
        value_type=DenseIndex,
        missing_grad=True,
    )

    # Color addition
    wp.context.add_builtin(
        "myPrint",
        input_types={"a": DenseIndex},
        value_type=None,
        missing_grad=True,
    )
    #
    # # Color scaling
    # wp.context.add_builtin(
    #     "mul",
    #     input_types={"s": float, "c": Color},
    #     value_type=Color,
    #     missing_grad=True,
    # )
    #
    # # get image width
    # wp.context.add_builtin(
    #     "img_width",
    #     input_types={"img": Image},
    #     value_type=int,
    #     missing_grad=True,
    # )
    #
    # # get image height
    # wp.context.add_builtin(
    #     "img_height",
    #     input_types={"img": Image},
    #     value_type=int,
    #     missing_grad=True,
    # )
    #
    # # get image data as a Warp array
    # wp.context.add_builtin(
    #     "img_data",
    #     input_types={"img": Image},
    #     value_type=wp.array2d(dtype=wp.vec3f),
    #     missing_grad=True,
    # )
    #
    # # get pixel
    # wp.context.add_builtin(
    #     "img_get_pixel",
    #     input_types={"img": Image, "coord": Coord},
    #     value_type=Color,
    #     missing_grad=True,
    # )
    #
    # # set pixel
    # wp.context.add_builtin(
    #     "img_set_pixel",
    #     input_types={"img": Image, "coord": Coord, "color": Color},
    #     value_type=None,
    #     missing_grad=True,
    # )


def register():
    _register_headers()
    _register_builtins()
