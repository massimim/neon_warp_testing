#pragma once

#include "Neon/domain/details/dGrid/dSpan.h"
#include "./neon_warp.h"
// TODO: currently, all types and builtins need to be in the wp:: namespace
namespace wp
{
namespace dense{
    using Span = ::Neon::domain::details::dGrid::dSpan;
    using Idx = ::Neon::domain::details::dGrid::dIndex;
}

// import types into this namespace
using dIndex = ::Neon::domain::details::dGrid::dIndex;

// Coord constructor exposed as a free function
CUDA_CALLABLE inline auto Dense_span_set_and_validata(dense.span.Span& span) ->dIndex
{
    dIndex index;
    span.setAndValidate(index, threadIdx.x, threadIdx.y, threadIdx.z);
    return index;
}
}