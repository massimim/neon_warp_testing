import ctypes

import warp as wp

import py_neon.dense.dPartition as dPartition
import py_neon.dense.dIndex as dIndex
from py_neon import Index_3d
from py_neon import Ngh_idx


def register_builtins():
    supported_types = [(dPartition.dPartition_int8, 'int8', wp.int8),
                       (dPartition.dPartition_uint8, 'uint8', wp.uint8),

                       (dPartition.dPartition_int32, 'int32', wp.int32),
                       (dPartition.dPartition_uint32, 'uint32', wp.uint32),

                       (dPartition.dPartition_int64, 'int64', wp.int64),
                       (dPartition.dPartition_uint64, 'uint64', wp.uint64),

                       (dPartition.dPartition_float32, 'float32', wp.float32),
                       (dPartition.dPartition_float64, 'float64', wp.float64)]

    for Partition, suffix, Type in supported_types:
        # register type
        wp.types.add_type(Partition, native_name=f"NeonDensePartition_{suffix}", has_binary_ctor=True)

        # print
        wp.context.add_builtin(
            "neon_print_dbg",
            input_types={"p": Partition},
            value_type=None,
            missing_grad=True,
        )

        wp.context.add_builtin(
            "neon_read",
            input_types={"partition": Partition, 'idx': dIndex, "card": int},
            value_type=Type,
            missing_grad=True,
        )

        wp.context.add_builtin(
            "neon_write",
            input_types={"partition": Partition, 'idx': dIndex, "card": int, "value": Type},
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
                         'idx': dIndex,
                         'ngh_idx': Ngh_idx,
                         "card": wp.int32,
                         "alternative": Type,
                         'is_valid': wp.bool},
            value_type=Type,
            missing_grad=True,
        )
        wp.context.add_builtin(
            "neon_partition_id",
            input_types={"partition": Partition},
            value_type=int,
            missing_grad=True,
        )

        wp.context.add_builtin(
            "neon_device_id",
            input_types={"partition": Partition},
            value_type=int,
            missing_grad=True,
        )

        wp.context.add_builtin(
            "neon_global_idx",
            input_types={"partition": Partition, 'idx': dIndex},
            value_type=Index_3d,
            missing_grad=True,
        )
