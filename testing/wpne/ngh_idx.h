#pragma once

#include "Neon/domain/details/dGrid/dIndex.h"

// TODO: currently, all types and builtins need to be in the wp:: namespace
namespace wp
{

// import types into this namespace
using NeonNghIdx = Neon::int8_3d;

// create dense index
CUDA_CALLABLE inline auto neon_ngh_idx(int8_t x, int8_t y, int8_t z) -> NeonNghIdx
{
    return NeonNghIdx(x, y, z);
}

CUDA_CALLABLE inline auto neon_init(NeonNghIdx& idx, int8_t x, int8_t y, int8_t z) -> void
{
    idx.x= x;
    idx.y= y;
    idx.z= z;
}

CUDA_CALLABLE inline auto neon_get_x(NeonNghIdx& idx) -> int
{
    return idx.x  ;
}

CUDA_CALLABLE inline auto neon_get_y(NeonNghIdx& idx) -> int
{
    return idx.y;
}

CUDA_CALLABLE inline auto neon_get_z(NeonNghIdx& idx) -> int
{
    return idx.z;
}

// print dense index
CUDA_CALLABLE inline auto neon_print(const NeonNghIdx& a) -> void
{
    printf("neon_print - NeonNghIdx(%d, %d, %d)\n", int(a.x),  int(a.y), int(a.z));
}

}
