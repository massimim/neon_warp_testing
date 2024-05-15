import ctypes
import os
import warp as wp
import py_neon as ne

from .idx import NeonDenseIdx


class NeonDenseSpan:
    # # define variables accessible in kernels (e.g., coord.x)
    vars = {
    }

    # struct that corresponds to the native Coord type
    # - used when packing arguments for kernels (pass-by-value)
    # - binary layout of fields must match native type
    class _type_(ctypes.Structure):
        _fields_ = ne.dense.Span.fields_()

        def __init__(self, span):
            self.dataView = span.dataView
            self.z_ghost_radius = span.z_ghost_radius
            self.z_boundary_radius = span.z_boundary_radius
            self.max_z_in_domain = span.max_z_in_domain
            self.span_dim = span.span_dim

    def __init__(self, s: ne.dense.Span):
        """
         s is the binding of dSpon fron Neon
        """
        # here we could use a pointer to copy the data with memcpy
        # the copy function should be providded by the neon bindings
        # Q: how do we get the correct pointer to this object?
        # Q: is the python layout the same the C++ ? My feeling is that there are actually 2 layours
        # - one for the python object (dSpan)
        # - one for the C++ object (_type_)
        # Can we use only __type__? I.e. dSpan becomes _type_ and we add vars to _type_
        self.dataView = s.dataView
        self.z_ghost_radius = s.z_ghost_radius
        self.z_boundary_radius = s.z_boundary_radius
        self.max_z_in_domain = s.max_z_in_domain
        self.span_dim = s.span_dim

    # HACK: used when packing kernel argument as `arg_type._type_(value.value)` in `pack_arg()` during `wp.launch()`
    @property
    def value(self):
        return self

    @staticmethod
    def _register_builtins():
        wp.context.add_builtin(
            "NeonDenseSpan_set_idx",
            input_types={"span": NeonDenseSpan, "is_valid": wp.bool},
            value_type=NeonDenseIdx,
            missing_grad=True,
        )
