import os
import warp as wp

from .container import Container
from .loader import Loader


def _add_header(path):
    include_directive = f"#include \"{path}\"\n"
    # add this header for all native modules
    wp.codegen.cpu_module_header += include_directive
    wp.codegen.cuda_module_header += include_directive

def _register_base_headers():
    include_path = os.path.abspath(os.path.dirname(__file__))
    _add_header(f"{include_path}/Index_3d.h")
    _add_header(f"{include_path}/dDataView.h")


def _register_dense_headers():
    include_path = os.path.abspath(os.path.dirname(__file__))
    _add_header(f"{include_path}/dense/dSpan.h")
    _add_header(f"{include_path}/dense/dPartition.h")

def _register_base_builtins():
    from wpne import idx_3d, data_view

    idx_3d.register_builtins()
    data_view.register_builtins()


def _register_dense_builtins():
    from .dense import span, partition

    span.register_builtins()
    partition.register_builtins()


def init():
    _register_base_headers()
    _register_dense_headers()

    _register_base_builtins()
    _register_dense_builtins()

