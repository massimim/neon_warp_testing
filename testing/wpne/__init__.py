import ctypes
import os
import warp as wp

from .dense.idx import NeonDenseIdx
from .dense.span import NeonDenseSpan

def _add_header(path):
    include_directive = f"#include \"{path}\"\n"
    # add this header for all native modules
    wp.codegen.cpu_module_header += include_directive
    wp.codegen.cuda_module_header += include_directive


def _register_dense_headers():
    include_path = os.path.abspath(os.path.dirname(__file__))
    _add_header(f"{include_path}/dense/dIdx.h")
    _add_header(f"{include_path}/dense/dSpan.h")

def _register_dense_builtins():
    NeonDenseIdx.register_builtins()
    NeonDenseSpan._register_builtins()

def init():
    _register_dense_headers()
    _register_dense_builtins()
    # dense.span._register_builtins()


# print(f"?????? {id(ne.dense.Span)}")