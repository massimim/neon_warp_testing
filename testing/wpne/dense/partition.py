import ctypes

import warp as wp

import py_neon.dense.dPartition as dPartition
from py_neon import Index_3d
from py_neon import Ngh_idx


def register_builtins():
    supported_types = [(dPartition.dPartitionInt, 'Int', wp.int32),
                       (dPartition.dPartitionFloat, 'Float', wp.float32),
                       (dPartition.dPartitionDouble, 'Double', wp.float64)]

    for Partition, suffix, Type in supported_types:
        # register type
        wp.types.add_type(Partition, native_name=f"NeonDensePartition{suffix}", has_binary_ctor=True)

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
            value_type=Type,
            missing_grad=True,
        )

        wp.context.add_builtin(
            "neon_write",
            input_types={"partition": Partition, 'idx': Index_3d, "card": int, "value": Type},
            value_type=None,
            missing_grad=True,
        )

        wp.context.add_builtin(
            "neon_cardinality",
            input_types={"partition": Partition},
            value_type=int,
            missing_grad=True,
        )

        wp.context.add_builtin(
            "neon_ngh_data",
            input_types={"partition": Partition,
                         'idx': Index_3d,
                         'ngh_idx': Ngh_idx,
                         "card": wp.int32,
                         "alternative": Type,
                         'is_valid': wp.bool},
            value_type=Type,
            missing_grad=True,
        )
