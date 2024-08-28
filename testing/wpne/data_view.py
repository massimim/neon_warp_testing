import warp as wp

from py_neon import DataView


def register_builtins():

    # register type
    wp.types.add_type(DataView, native_name="NeonDataView")

    # print
    wp.context.add_builtin(
        "NeonDataView_print",
        input_types={"a": DataView},
        value_type=None,
        missing_grad=True,
    )
