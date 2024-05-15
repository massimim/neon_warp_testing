import ctypes
import os
import warp as wp



class NeonDenseIdx:
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

    @staticmethod
    def register_builtins():
        # Coord constructor
        wp.context.add_builtin(
            "Idx_",
            input_types={"x": int, "y": int, "z": int},
            value_type=NeonDenseIdx,
            missing_grad=True,
        )

        # Color addition
        wp.context.add_builtin(
            "myPrint",
            input_types={"a": NeonDenseIdx},
            value_type=None,
            missing_grad=True,
        )

