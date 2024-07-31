#pragma once

#include "Neon/domain/details/dGrid/dIndex.h"

// TODO: currently, all types and builtins need to be in the wp:: namespace
namespace wp
{

// import types into this namespace
using NeonDenseIdx = ::Neon::domain::details::dGrid::dIndex;

// create dense index
CUDA_CALLABLE inline auto NeonDenseIdx_create(int x, int y, int z) -> NeonDenseIdx
{
    return NeonDenseIdx(x, y, z);
}

// print dense index
CUDA_CALLABLE inline auto neon_print(const NeonDenseIdx& a) -> void
{
    printf("neon_print - NeonDenseIdx(%d, %d, %d)\n", a.getLocation().x,  a.getLocation().y, a.getLocation().z);
}

}
