import os
import warp as wp


def _add_header(path):
    include_directive = f"#include \"{path}\"\n"
    # add this header for all native modules
    wp.codegen.cpu_module_header += include_directive
    wp.codegen.cuda_module_header += include_directive


def _register_dense_headers():
    include_path = os.path.abspath(os.path.dirname(__file__))
    _add_header(f"{include_path}/dense/dIdx.h")
    _add_header(f"{include_path}/dense/dDataView.h")
    _add_header(f"{include_path}/dense/dSpan.h")
    _add_header(f"{include_path}/dense/dPartition.h")


def _register_dense_builtins():
    from .dense import idx, data_view, span, partition

    idx.register_builtins()
    data_view.register_builtins()
    span.register_builtins()
    partition.register_builtins()


def init():
    _register_dense_headers()
    _register_dense_builtins()
