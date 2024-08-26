#pragma once

#include <cstdio>

#include "Neon/domain/details/dGrid/dSpan.h"
#include "./dIdx.h"

// TODO: currently, all types and builtins need to be in the wp:: namespace
namespace wp
{
// using NeonDenseSpan = ::Neon::domain::details::dGrid::dSpan;

class NeonDenseSpan : public ::Neon::domain::details::dGrid::dSpan
{
public:

    // NOTE: The dSpan class is private and is missing a suitable constructor/setters
    // NeonDenseSpan(Neon::DataView dataView,
    //               int zGhostRadius,
    //               int zBoundaryRadius,
    //               int maxZInDomain,
    //               const Neon::index_3d& spanDim)
    // {
    // }

    // ... that's why we need to initialize it from bytes
    NeonDenseSpan(const char* bytes, size_t n)
    {
        assert(n == sizeof(*this));
        memcpy(this, bytes, n);
    }

    // NOTE: need default constructor for adjoint vars
    NeonDenseSpan()
    {
    }
};

// print
CUDA_CALLABLE inline auto neon_print(const NeonDenseSpan& a) -> void
{
    Neon::index_3d dim = a.helpGetDim();
//    printf("NeonDenseSpan(%d, %d, %d, {%d, %d, %d})\n",
//        int(a.helpGetDataView()),
//        a.helpGetZHaloRadius(),
//        a.helpGetZBoundaryRadius(),
//        dim.x, dim.y, dim.z);
}

CUDA_CALLABLE inline auto neon_set(NeonDenseSpan& span, bool& is_valid)
 -> NeonDenseIdx
{
    NeonDenseIdx index;
    using DummyType = int;
    is_valid = span.template setAndValidate_warp<DummyType>(index);
    return index;
}

CUDA_CALLABLE inline auto neon_set(NeonDenseSpan& span, int x, int y, int z)
 -> NeonDenseIdx
{
    NeonDenseIdx index;
    span.setAndValidate_warp(index, x,y,z);
    return index;
}


}
