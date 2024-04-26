import ctypes
import os
import warp as wp

from wpne.wpne_dense_index import DenseIndex
from wpne.wpne_dense_span import dSpan

from py_neon import *

def _add_header(path):
    include_directive = f"#include \"{path}\"\n"
    # add this header for all native modules
    wp.codegen.cpu_module_header += include_directive
    wp.codegen.cuda_module_header += include_directive


def _register_headers():
    include_path = os.path.abspath(os.path.dirname(__file__))
    _add_header(f"{include_path}/neon_warp.h")
    _add_header(f"{include_path}/dSpan.h")

def init():
    import wpne.wpne_dense_index as dense_index
    _register_headers()
    dense_index._register_builtins()
    wpne_dense_span._register_builtins()
