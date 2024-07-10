
from env_setup import update_pythonpath
update_pythonpath()
import py_neon as ne
import random


def test_mGrid_allocations():
    allocationCounter = ne.allocationCounter()
    def local_scope():
        assert allocationCounter.get_allocation_count() == 0, "allocation count should be 0"
        backend = ne.Backend(ne.Backend.Runtime.openmp, 1)
        assert allocationCounter.get_allocation_count() == 1, "allocation count should be 1"
        grid = ne.mGrid(backend, ne.Index_3d(10,10,10))
        assert allocationCounter.get_allocation_count() == 2, "allocation count should be 2"
        span_device_id0_standard = grid.get_span(ne.Execution.device(),
                                            0,
                                            ne.DataView.standard())
        assert allocationCounter.get_allocation_count() == 2, "allocation count should be 2"
        field = grid.new_field(10)
        assert allocationCounter.get_allocation_count() == 3, "allocation count should be 3"
    local_scope()
    assert allocationCounter.get_allocation_count() == 0, "allocation count should be 0"


def test_mGrid_mPartition_offsets():
    partition = ne.mPartitionInt()
    assert partition.get_offsets() == partition.get_cpp_field_offsets(), "python and cpp mPartitionInt offsets do not match"

def test_mGrid_dimensions():
    grid1 = ne.mGrid(ne.Backend(ne.Backend.Runtime.openmp, 1), ne.Index_3d(1,1,1))
    assert grid1.get_python_dimensions() == ne.Index_3d(1,1,1), "mGrid python dimensions test 1 incorrect"
    assert grid1.get_cpp_dimensions() == ne.Index_3d(1,1,1), "mGrid cpp dimensions test 1 incorrect"

    grid2 = ne.mGrid(ne.Backend(ne.Backend.Runtime.openmp, 1), ne.Index_3d(10,99,3))
    assert grid2.get_python_dimensions() == ne.Index_3d(10,99,3), "mGrid python dimensions test 2 incorrect"
    assert grid2.get_cpp_dimensions() == ne.Index_3d(10,99,3), "mGrid cpp dimensions test 2 incorrect"

    grid3 = ne.mGrid(ne.Backend(ne.Backend.Runtime.openmp, 1), ne.Index_3d(17,31,10513))
    assert grid3.get_python_dimensions() == ne.Index_3d(17,31,10513), "mGrid python dimensions test 3 incorrect"
    assert grid3.get_cpp_dimensions() == ne.Index_3d(17,31,10513), "mGrid cpp dimensions test 3 incorrect"

def test_mGrid_get_properties():
    grid = ne.mGrid(ne.Backend(ne.Backend.Runtime.openmp, 1), ne.Index_3d(10,10,10))
    for x in range (0,10):
        for y in range (0,10):
            for z in range (0,10):
                assert grid.getProperties(ne.Index_3d(x,y,z)) == ne.DataView.standard(), "mGrid dataview properties for each cell should start as `standard`"

def test_mGrid_is_inside_domain():
    grid1 = ne.mGrid(ne.Backend(ne.Backend.Runtime.openmp, 1), ne.Index_3d(10,10,10))
    for x in range (-4,12):
        for y in range (-4,12):
            for z in range (-4,12):
                if (x in range(0,10) and y in range (0,10) and z in range (0,10)):
                    assert grid1.isInsideDomain(ne.Index_3d(x,y,z)), "everything inside [0,9] x [0,9] x [0,9] should be inside mGrid's domain"
                else:
                    assert not grid1.isInsideDomain(ne.Index_3d(x,y,z)), "everything outisde [0,9] x [0,9] x [0,9] should not be inside mGrid's domain"
                    
# def test_mGrid_mField_read():
#     backend = ne.Backend(ne.Backend.Runtime.openmp, 1)
#     grid = ne.mGrid(backend, ne.Index_3d(10,10,10))
#     field = grid.new_field(10)
#     for x in range (-1,11):
#         for y in range (-1,11):
#             for z in range (-1,11):
#                 for cardinality in range (0,10):
#                     if (x in range(0,10) and y in range (0,10) and z in range (0,10)):
#                         assert field.read(ne.Index_3d(x,y,z), cardinality) == 0
#                     #     assert grid1.isInsideDomain(ne.Index_3d(x,y,z)), "everything inside [0,9] x [0,9] x [0,9] should be inside mGrid's domain"
#                     # else:
#                     #     assert not grid1.isInsideDomain(ne.Index_3d(x,y,z)), "everything outisde [0,9] x [0,9] x [0,9] should not be inside mGrid's domain"


def test_mGrid_mField_read_write():
    backend = ne.Backend(ne.Backend.Runtime.openmp, 1)
    grid = ne.mGrid(backend, ne.Index_3d(10,10,10))
    field = grid.new_field(10)
    for x in range (0,10):
        for y in range (0,10):
            for z in range (0,10):
                for cardinality in range (0,10): # @TODOMATT why is it stuck in this range??? For some reason there is a double free error if we expand the range naything beyond (2,13), but it should be (0,19).
                    # 1==1
                    randomVal = random.randint(10,1000)
                    field.write(ne.Index_3d(x,y,z), cardinality, randomVal)
                    assert field.read(ne.Index_3d(x,y,z), cardinality) == randomVal

def test_mGrid_mField_update_device_and_host_data():
    backend = ne.Backend(ne.Backend.Runtime.openmp, 1)
    grid = ne.mGrid(backend, ne.Index_3d(10,10,10))
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



# def main():
#     print("main begins:")
#     allocationCounter = ne.allocationCounter()
#     def local_scope():
#         print(allocationCounter.get_allocation_count() == 0)
#         backend = ne.Backend(ne.Backend.Runtime.openmp, 1)
#         print(allocationCounter.get_allocation_count() == 1)
#         grid = ne.mGrid(backend, ne.Index_3d(10,10,10))
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