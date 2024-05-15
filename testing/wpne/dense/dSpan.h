#pragma once

#include "Neon/domain/details/dGrid/dSpan.h"
#include "./dIdx.h"
// TODO: currently, all types and builtins need to be in the wp:: namespace
namespace wp
{
using NeonDenseSpan = ::Neon::domain::details::dGrid::dSpan;

// Coord constructor exposed as a free function
CUDA_CALLABLE inline auto Dense_span_set_and_validata(NeonDenseSpan& span)
 -> NeonDenseIdx
{
    NeonDenseIdx index;
    span.setAndValidate(index, threadIdx.x, threadIdx.y, threadIdx.z);
    return index;
}
}