import ctypes
import os
import warp as wp

from .idx import NeonDenseIdx
from py_neon.dataview import DataView as NeDataView
from py_neon.index_3d import Index_3d as NeIndex_3d
from py_neon.dense.partition import PartitionInt as NePartitionInt


class NeonDensePartitionInt:
    # # define variables accessible in kernels (e.g., coord.x)
    vars = {
    }

    # struct that corresponds to the native Coord type
    # - used when packing arguments for kernels (pass-by-value)
    # - binary layout of fields must match native type
    class _type_(ctypes.Structure):
        _fields_ = [
            ("mDataView", NeDataView),
            ("mMem", ctypes.POINTER(ctypes.c_int)),
            ("mDim", NeIndex_3d),
            ("mZHaloRadius", ctypes.c_int),
            ("mZBoundaryRadius", ctypes.c_int),
            ("mPitch1", ctypes.c_uint64),
            ("mPitch2", ctypes.c_uint64),
            ("mPitch3", ctypes.c_uint64),
            ("mPitch4", ctypes.c_uint64),
            ("mPrtID", ctypes.c_uint64),
            ("mOrigin", NeIndex_3d),
            ("mCardinality", ctypes.c_int),
            ("mFullGridSize", NeIndex_3d),
            ("mPeriodicZ", ctypes.c_bool),
            ("mStencil", ctypes.POINTER(ctypes.c_int)),
        ]

        def __init__(self, partition):
            self.mDataView = partition.mDataView
            self.mMem = partition.mMemc
            self.mDim = partition.mDim
            self.mZHaloRadius = partition.mZHaloRadius
            self.mZBoundaryRadius = partition.mZBoundaryRadius
            self.mPitch1 = partition.mPitch1
            self.mPitch2 = partition.mPitch2
            self.mPitch3 = partition.mPitch3
            self.mPitch4 = partition.mPitch4
            self.mPrtID = partition.mPrtID
            self.mOrigin = partition.mOrigin
            self.mCardinality = partition.mCardinality
            self.mFullGridSize = partition.mFullGridSize
            self.mPeriodicZ = partition.mPeriodicZ
            self.mStencil = partition.mStencil


    def __init__(self, partition: NePartitionInt):
        self.mDataView = partition.mDataView
        self.mMem = partition.mMem
        self.mDim = partition.mDim
        self.mZHaloRadius = partition.mZHaloRadius
        self.mZBoundaryRadius = partition.mZBoundaryRadius
        self.mPitch1 = partition.mPitch1
        self.mPitch2 = partition.mPitch2
        self.mPitch3 = partition.mPitch3
        self.mPitch4 = partition.mPitch4
        self.mPrtID = partition.mPrtID
        self.mOrigin = partition.mOrigin
        self.mCardinality = partition.mCardinality
        self.mFullGridSize = partition.mFullGridSize
        self.mPeriodicZ = partition.mPeriodicZ
        self.mStencil = partition.mStencil

    # HACK: used when packing kernel argument as `arg_type._type_(value.value)` in `pack_arg()` during `wp.launch()`
    @property
    def value(self):
        return self

    @staticmethod
    def _register_builtins():

        print(f"?????? NeonDenseIdx {id(NeonDenseIdx)}")
        from wpne.dense.idx import NeonDenseIdx as wpne_dense_NeonDenseIdx
        print(f"?????? wpne_dense_NeonDenseIdx {id(wpne_dense_NeonDenseIdx)}")

        wp.context.add_builtin(
            "NeonDensePartitionInt_read",
            input_types={"partition": NeonDensePartitionInt, 'idx': NeonDenseIdx, "value": ctypes.c_int},
            value_type=wp.bool,
            missing_grad=True,
        )

        wp.context.add_builtin(
            "NeonDensePartitionInt_write",
            input_types={"partition": NeonDensePartitionInt, 'idx': NeonDenseIdx, "value": ctypes.c_int},
            value_type=wp.bool,
            missing_grad=True,
        )
