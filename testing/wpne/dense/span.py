import warp as wp

from py_neon import Index_3d
from py_neon.dense import Span


def register_builtins():

    # register type
    wp.types.add_type(Span, native_name="NeonDenseSpan", has_binary_ctor=True)

    # print
    wp.context.add_builtin(
        "NeonDenseSpan_print",
        input_types={"a": Span},
        value_type=None,
        missing_grad=True,
    )

    wp.context.add_builtin(
        "NeonDenseSpan_set_idx",
        input_types={"span": Span, "is_valid": wp.bool},
        value_type=Index_3d,
        missing_grad=True,
    )
