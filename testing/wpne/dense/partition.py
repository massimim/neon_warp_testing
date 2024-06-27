import warp as wp

from py_neon import Index_3d
from py_neon.dense.partition import PartitionInt


def register_builtins():

    # register type
    wp.types.add_type(PartitionInt, native_name="NeonDensePartitionInt", has_binary_ctor=True)

    # print
    wp.context.add_builtin(
        "NeonDensePartitionInt_print",
        input_types={"p": PartitionInt},
        value_type=None,
        missing_grad=True,
    )

    wp.context.add_builtin(
        "NeonDensePartitionInt_read",
        input_types={"partition": PartitionInt, 'idx': Index_3d, "card": int},
        value_type=int,
        missing_grad=True,
    )

    wp.context.add_builtin(
        "NeonDensePartitionInt_write",
        input_types={"partition": PartitionInt, 'idx': Index_3d, "card": int, "value": int},
        value_type=None,
        missing_grad=True,
    )
