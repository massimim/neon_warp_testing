import warp as wp

from py_neon import Index_3d
from py_neon.dense.dPartition import dPartitionInt


def register_builtins():

    # register type
    wp.types.add_type(dPartitionInt, native_name="NeonDensePartitionInt", has_binary_ctor=True)

    # print
    wp.context.add_builtin(
        "neon_print",
        input_types={"p": dPartitionInt},
        value_type=None,
        missing_grad=True,
    )

    wp.context.add_builtin(
        "neon_read",
        input_types={"partition": dPartitionInt, 'idx': Index_3d, "card": int},
        value_type=int,
        missing_grad=True,
    )

    wp.context.add_builtin(
        "NeonDensePartitionInt_write",
        input_types={"partition": dPartitionInt, 'idx': Index_3d, "card": int, "value": int},
        value_type=None,
        missing_grad=True,
    )
