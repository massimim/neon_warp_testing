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
CUDA_CALLABLE inline auto NeonDenseSpan_print(const NeonDenseSpan& a) -> void
{
    Neon::index_3d dim = a.helpGetDim();
    printf("NeonDenseSpan(%d, %d, %d, {%d, %d, %d})\n",
        int(a.helpGetDataView()),
        a.helpGetZHaloRadius(),
        a.helpGetZBoundaryRadius(),
        dim.x, dim.y, dim.z);
}

CUDA_CALLABLE inline auto NeonDenseSpan_set_idx(NeonDenseSpan& span, bool& is_valid)
 -> NeonDenseIdx
{
    NeonDenseIdx index;
    is_valid = span.setAndValidate(index, threadIdx.x, threadIdx.y, threadIdx.z);
    return index;
}

CUDA_CALLABLE inline auto NeonDenseSpan_set_idx(NeonDenseSpan& span, int x, int y, int z)
 -> NeonDenseIdx
{
    NeonDenseIdx index;
    span.setAndValidate(index, threadIdx.x, threadIdx.y, threadIdx.z);
    return index;
}

}
