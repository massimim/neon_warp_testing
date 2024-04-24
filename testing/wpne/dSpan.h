#pragma once

// TODO: may need to add a mechanism for include paths
#include "Neon/domain/details/dGrid/dSpan.h"
#include "./neon_warp.h"
// TODO: currently, all types and builtins need to be in the wp:: namespace
namespace wp
{

// import types into this namespace
using dSpan = ::Neon::domain::details::dGrid::dSpan;
using dIndex = ::Neon::domain::details::dGrid::dIndex;

// Coord constructor exposed as a free function
CUDA_CALLABLE inline auto dSpan_set_and_validata(dSpan& span) ->dIndex
{
    dIndex index;
    span.setAndValidate(index, threadIdx.x, threadIdx.y, threadIdx.z);
    return index;
}
}