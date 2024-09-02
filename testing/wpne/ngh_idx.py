import warp as wp

from py_neon import Ngh_idx
import ctypes


def register_builtins():
    # register type
    wp.types.add_type(Ngh_idx, native_name="NeonNghIdx")

    # create dense index
    wp.context.add_builtin(
        "neon_ngh_idx",
        input_types={"x": wp.int8, "y": wp.int8, "z": wp.int8},
        value_type=Ngh_idx,
        missing_grad=True,
    )

    # create dense index
    wp.context.add_builtin(
        "neon_ngh_idx",
        input_types={"idx": Ngh_idx, "x": wp.int8, "y": wp.int8, "z": wp.int8},
        value_type=None,
        missing_grad=True,
    )

    wp.context.add_builtin(
        "neon_ngh_idx",
        input_types={"idx": Ngh_idx},
        value_type=wp.int8,
        missing_grad=True,
    )

    wp.context.add_builtin(
        "neon_get_y",
        input_types={"idx": Ngh_idx},
        value_type=wp.int8,
        missing_grad=True,
    )

    wp.context.add_builtin(
        "neon_get_z",
        input_types={"idx": Ngh_idx},
        value_type=wp.int8,
        missing_grad=True,
    )

    # print dense index
    wp.context.add_builtin(
        "neon_print",
        input_types={"a": Ngh_idx},
        value_type=None,
        missing_grad=True,
    )
