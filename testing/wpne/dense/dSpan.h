#pragma once

#include "Neon/domain/details/dGrid/dSpan.h"
#include "./dIdx.h"
// TODO: currently, all types and builtins need to be in the wp:: namespace
namespace wp
{
using NeonDenseSpan = ::Neon::domain::details::dGrid::dSpan;

// Coord constructor exposed as a free function
CUDA_CALLABLE inline auto NeonDenseSpan_set_idx(NeonDenseSpan& span, bool& is_valid)
 -> NeonDenseIdx
{
    NeonDenseIdx index;
    is_valid = span.setAndValidate(index, threadIdx.x, threadIdx.y, threadIdx.z);
    return index;
}
}