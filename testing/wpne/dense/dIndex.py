import warp as wp

from py_neon import dIndex
from py_neon.dense.dIndex import dIndex

def register_builtins():

    # register type
    wp.types.add_type(dIndex, native_name="NeonDenseIdx")

    # # create dense index
    # wp.context.add_builtin(
    #     "neon_idx_3d",
    #     input_types={"x": int, "y": int, "z": int},
    #     value_type=dIndex,
    #     missing_grad=True,
    # )
    #
    # # create dense index
    # wp.context.add_builtin(
    #     "neon_init",
    #     input_types={"idx":dIndex, "x": int, "y": int, "z": int},
    #     value_type=None,
    #     missing_grad=True,
    # )

    wp.context.add_builtin(
        "neon_get_x",
        input_types={"idx":dIndex},
        value_type=int,
        missing_grad=True,
    )

    wp.context.add_builtin(
        "neon_get_y",
        input_types={"idx":dIndex},
        value_type=int,
        missing_grad=True,
    )

    wp.context.add_builtin(
        "neon_get_z",
        input_types={"idx":dIndex},
        value_type=int,
        missing_grad=True,
    )

    # print dense index
    wp.context.add_builtin(
        "neon_print",
        input_types={"a": dIndex},
        value_type=None,
        missing_grad=True,
    )
