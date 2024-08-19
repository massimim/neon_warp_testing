import warp as wp

from py_neon import Index_3d


def register_builtins():

    # register type
    wp.types.add_type(Index_3d, native_name="NeonDenseIdx")

    # create dense index
    wp.context.add_builtin(
        "neon_idx_3d",
        input_types={"x": int, "y": int, "z": int},
        value_type=Index_3d,
        missing_grad=True,
    )

    # create dense index
    wp.context.add_builtin(
        "neon_init",
        input_types={"idx":Index_3d, "x": int, "y": int, "z": int},
        value_type=None,
        missing_grad=True,
    )

    wp.context.add_builtin(
        "neon_get_x",
        input_types={"idx":Index_3d},
        value_type=int,
        missing_grad=True,
    )

    wp.context.add_builtin(
        "neon_get_y",
        input_types={"idx":Index_3d},
        value_type=int,
        missing_grad=True,
    )

    wp.context.add_builtin(
        "neon_get_z",
        input_types={"idx":Index_3d},
        value_type=int,
        missing_grad=True,
    )

    # print dense index
    wp.context.add_builtin(
        "neon_print",
        input_types={"a": Index_3d},
        value_type=None,
        missing_grad=True,
    )
