import warp as wp

import py_neon.dense.dPartition as dPartition
from py_neon import Index_3d


def register_builtins():
    supported_types = [dPartition.dPartitionInt,
                       dPartition.dPartitionFloat,
                       dPartition.dPartitionDouble]

    for Partition in supported_types:
        # register type
        wp.types.add_type(Partition, native_name="NeonDensePartitionInt", has_binary_ctor=True)
        # wp.types.add_type(dPartitionDouble, native_name="NeonDensePartitionDouble", has_binary_ctor=True)
        # wp.types.add_type(dPartitionFloat, native_name="NeonDensePartitionFloat", has_binary_ctor=True)

        # print
        wp.context.add_builtin(
            "neon_print",
            input_types={"p": Partition},
            value_type=None,
            missing_grad=True,
        )

        wp.context.add_builtin(
            "neon_read",
            input_types={"partition": Partition, 'idx': Index_3d, "card": int},
            value_type=int,
            missing_grad=True,
        )

        wp.context.add_builtin(
            "neon_write",
            input_types={"partition": Partition, 'idx': Index_3d, "card": int, "value": int},
            value_type=None,
            missing_grad=True,
        )

        wp.context.add_builtin(
            "neon_cardinality",
            input_types={"partition": Partition},
            value_type=int,
            missing_grad=True,
        )
