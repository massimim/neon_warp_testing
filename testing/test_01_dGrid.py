
from env_setup import update_pythonpath
update_pythonpath()
import py_neon as ne

def test_dGrid_allocations():
    print("test 1")
    allocationCounter = ne.allocationCounter()
    def local_scope():
        assert allocationCounter.get_allocation_count() == 0, "allocation count should be 0"
        backend = ne.Backend(1, ne.Backend.Runtime.openmp)
        assert allocationCounter.get_allocation_count() == 1, "allocation count should be 1"
        grid = ne.dGrid(backend, ne.Index_3d(10,10,10))
        assert allocationCounter.get_allocation_count() == 2, "allocation count should be 2"
        span_device_id0_standard = grid.get_span(ne.Execution.device(),
                                            0,
                                            ne.DataView.standard())
        assert allocationCounter.get_allocation_count() == 2, "allocation count should be 2"
        field = grid.new_field()
        assert allocationCounter.get_allocation_count() == 3, "allocation count should be 3"
    local_scope()
    assert allocationCounter.get_allocation_count() == 0, "allocation count should be 0"


def test_dGrid_dSpan_offsets():
    grid = ne.dGrid(ne.Backend(1, ne.Backend.Runtime.openmp), ne.Index_3d(10,10,10))
    span = ne.dSpan()
    assert span.get_offsets() == span.get_member_field_offsets(), "python and cpp offsets do not match."

# Need Index_3d to work to implement the following test
# def test_dGrid_dimensions():
#     grid = ne.dGrid(ne.Backend(1, ne.Backend.Runtime.openmp), ne.Index_3d(10,10,10))
#     assert grid.dimensions() == ne.Index_3d(10,10,10), "grid dimensions do not match"

# def main():
#     print("main begins:")
#     allocationCounter = ne.allocationCounter()
#     def local_scope():
#         print(allocationCounter.get_allocation_count() == 0)
#         backend = ne.Backend(1, ne.Backend.Runtime.openmp)
#         print(allocationCounter.get_allocation_count() == 1)
#         grid = ne.dGrid(backend, ne.Index_3d(10,10,10))
#         print(allocationCounter.get_allocation_count() == 2)
#         span_device_id0_standard = grid.get_span(ne.Execution.device(),
#                                             0,
#                                             ne.DataView.standard())
#         print(allocationCounter.get_allocation_count() == 3)
#         field = grid.new_field()
#         print(allocationCounter.get_allocation_count() == 3)
#     local_scope()
#     print(allocationCounter.get_allocation_count() == 0, "allocation count should be 0")

# if __name__ == "__main__":
#     main()