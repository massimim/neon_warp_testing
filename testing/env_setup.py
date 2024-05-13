import os
import sys

def update_pythonpath():
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    sys.path.insert(0, script_dir+'/../neon_py_bindings/py_neon/')
    # print PYTHONPATH
    print(f"PYTHONPATH: {sys.path}")