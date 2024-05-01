import ctypes
import os
import warp as wp

import dense
import py_neon as ne

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
    _register_headers()
    from wpne.dense.idx import register_builtins
    register_builtins()
    dense.span._register_builtins()
