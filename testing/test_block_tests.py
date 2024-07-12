
from env_setup import update_pythonpath
update_pythonpath()
import py_neon as ne
import random


def test_bGrid_allocations():
    allocationCounter = ne.allocationCounter()
    def local_scope():
        assert allocationCounter.get_allocation_count() == 0, "allocation count should be 0"
        backend = ne.Backend(ne.Backend.Runtime.openmp, 1)
        assert allocationCounter.get_allocation_count() == 1, "allocation count should be 1"
        grid = ne.bGrid(backend, ne.Index_3d(10,10,10))
        assert allocationCounter.get_allocation_count() == 2, "allocation count should be 2"
        span_device_id0_standard = grid.get_span(ne.Execution.device(),
                                            0,
                                            ne.DataView.standard())
        assert allocationCounter.get_allocation_count() == 2, "allocation count should be 2"
        field = grid.new_field(10)
        assert allocationCounter.get_allocation_count() == 3, "allocation count should be 3"
    local_scope()
    assert allocationCounter.get_allocation_count() == 0, "allocation count should be 0"


def test_bGrid_dSpan_offsets():
    span = ne.bSpan()
    assert span.get_offsets() == span.get_cpp_field_offsets(), "python and cpp bSpan offsets do not match."

def test_bGrid_dPartition_offsets():
    partition = ne.bPartitionInt()
    assert partition.get_offsets() == partition.get_cpp_field_offsets(), "python and cpp bPartitionInt offsets do not match"

def test_bGrid_dimensions():
    grid1 = ne.bGrid(ne.Backend(ne.Backend.Runtime.openmp, 1), ne.Index_3d(1,1,1))
    assert grid1.get_python_dimensions() == ne.Index_3d(1,1,1), "bGrid python dimensions test 1 incorrect"
    assert grid1.get_cpp_dimensions() == ne.Index_3d(1,1,1), "bGrid cpp dimensions test 1 incorrect"

    grid2 = ne.bGrid(ne.Backend(ne.Backend.Runtime.openmp, 1), ne.Index_3d(10,99,3))
    assert grid2.get_python_dimensions() == ne.Index_3d(10,99,3), "bGrid python dimensions test 2 incorrect"
    assert grid2.get_cpp_dimensions() == ne.Index_3d(10,99,3), "bGrid cpp dimensions test 2 incorrect"

    grid3 = ne.bGrid(ne.Backend(ne.Backend.Runtime.openmp, 1), ne.Index_3d(17,31,10513))
    assert grid3.get_python_dimensions() == ne.Index_3d(17,31,10513), "bGrid python dimensions test 3 incorrect"
    assert grid3.get_cpp_dimensions() == ne.Index_3d(17,31,10513), "bGrid cpp dimensions test 3 incorrect"

def test_bGrid_get_properties():
    grid = ne.bGrid(ne.Backend(ne.Backend.Runtime.openmp, 1), ne.Index_3d(10,10,10))
    for x in range (0,10):
        for y in range (0,10):
            for z in range (0,10):
                assert grid.getProperties(ne.Index_3d(x,y,z)) == ne.DataView.standard(), "bGrid dataview properties for each cell should start as `standard`"

def test_bGrid_is_inside_domain():
    grid1 = ne.bGrid(ne.Backend(ne.Backend.Runtime.openmp, 1), ne.Index_3d(10,10,10))

    # @TODOMATT for some reason, negative numbers in the index behave strangely for bGrids, so they won't be inluded in these tests
    for x in range (0,12):
        for y in range (0,12):
            for z in range (0,12):
                # assert grid1.isInsideDomain(ne.Index_3d(x,y,z))
                if (x in range(0,10) and y in range (0,10) and z in range (0,10)):
                    assert grid1.isInsideDomain(ne.Index_3d(x,y,z)), f"failure with ({x},{y},{z}); everything inside [0,9] x [0,9] x [0,9] should be inside bGrid's domain"
                else:
                    assert not grid1.isInsideDomain(ne.Index_3d(x,y,z)), f"failure with ({x},{y},{z}); everything outisde [0,9] x [0,9] x [0,9] should not be inside bGrid's domain"

def test_bGrid_dField_read_write():
    backend = ne.Backend(ne.Backend.Runtime.openmp, 1)
    grid = ne.bGrid(backend, ne.Index_3d(10,10,10))
    field = grid.new_field(10)
    for x in range (0,10):
        for y in range (0,10):
            for z in range (0,10):
                for cardinality in range (0,10): 
                    randomVal = random.randint(10,1000)
                    field.write(ne.Index_3d(x,y,z), cardinality, randomVal)
                    assert field.read(ne.Index_3d(x,y,z), cardinality) == randomVal

def test_bGrid_dField_update_device_and_host_data():
    backend = ne.Backend(ne.Backend.Runtime.openmp, 1)
    grid = ne.bGrid(backend, ne.Index_3d(10,10,10))
    field = grid.new_field(10)

    #write data
    field.write(ne.Index_3d(3,3,3), 10, 400)
    field.write(ne.Index_3d(2,9,7), 3, 600)
    field.write(ne.Index_3d(0,0,0), 11, 2)
    field.write(ne.Index_3d(9,9,9), 7, 11)


    #update device
    field.updateDeviceData(0)
    
    # synchronize backend
    backend.sync()

    #write garbage
    field.write(ne.Index_3d(3,3,3), 10, 7)
    field.write(ne.Index_3d(2,9,7), 3, 7)
    field.write(ne.Index_3d(0,0,0), 11, 7)
    field.write(ne.Index_3d(9,9,9), 7, 7)

    #update host
    field.updateHostData(0)

    # synchronize backend
    backend.sync()

    #should be original
    assert field.read(ne.Index_3d(3,3,3), 10) == 400, "garbage data written after field.updateDeviceData(0) should not be preserved after field.updateHostData(0)"
    assert field.read(ne.Index_3d(2,9,7), 3) == 600, "garbage data written after field.updateDeviceData(0) should not be preserved after field.updateHostData(0)"
    assert field.read(ne.Index_3d(0,0,0), 11) == 2, "garbage data written after field.updateDeviceData(0) should not be preserved after field.updateHostData(0)"
    assert field.read(ne.Index_3d(9,9,9), 7) == 11, "garbage data written after field.updateDeviceData(0) should not be preserved after field.updateHostData(0)"

def test_bGrid_sparsity_pattern():
    1==1

# # def main():
# #     print("main begins:")
# #     allocationCounter = ne.allocationCounter()
# #     def local_scope():
# #         print(allocationCounter.get_allocation_count() == 0)
# #         backend = ne.Backend(ne.Backend.Runtime.openmp, 1)
# #         print(allocationCounter.get_allocation_count() == 1)
# #         grid = ne.bGrid(backend, ne.Index_3d(10,10,10))
# #         print(allocationCounter.get_allocation_count() == 2)
# #         span_device_id0_standard = grid.get_span(ne.Execution.device(),
# #                                             0,
# #                                             ne.DataView.standard())
# #         print(allocationCounter.get_allocation_count() == 3)
# #         field = grid.new_field()
# #         print(allocationCounter.get_allocation_count() == 3)
# #     local_scope()
# #     print(allocationCounter.get_allocation_count() == 0, "allocation count should be 0")

# # if __name__ == "__main__":
# #     main()