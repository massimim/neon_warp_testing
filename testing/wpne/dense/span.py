import warp as wp

from py_neon import Index_3d
from py_neon.dense import dSpan


def register_builtins():
    # register type
    wp.types.add_type(dSpan, native_name="NeonDenseSpan", has_binary_ctor=True)

    # print
    wp.context.add_builtin(
        "NeonDenseSpan_print",
        input_types={"a": dSpan},
        value_type=None,
        missing_grad=True,
    )

    wp.context.add_builtin(
        "NeonDenseSpan_set_idx",
        input_types={"span": dSpan, "is_valid": wp.bool},
        value_type=Index_3d,
        missing_grad=True,
    )

    wp.context.add_builtin(
        "NeonDenseSpan_set_idx",
        input_types={"span": dSpan, 'x': int, 'y': int, 'z': int},
        value_type=Index_3d,
        missing_grad=True,
    )

    wp.context.add_builtin(
        "neon_set",
        input_types={"span": dSpan, 'x': int, 'y': int, 'z': int},
        value_type=Index_3d,
        missing_grad=True,
    )