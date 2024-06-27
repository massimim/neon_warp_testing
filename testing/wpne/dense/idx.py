import warp as wp

from py_neon import Index_3d


def register_builtins():

    # register type
    wp.types.add_type(Index_3d, native_name="NeonDenseIdx")

    # create dense index
    wp.context.add_builtin(
        "NeonDenseIdx_create",
        input_types={"x": int, "y": int, "z": int},
        value_type=Index_3d,
        missing_grad=True,
    )

    # print dense index
    wp.context.add_builtin(
        "NeonDenseIdx_print",
        input_types={"a": Index_3d},
        value_type=None,
        missing_grad=True,
    )
