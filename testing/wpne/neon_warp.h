#pragma once

// TODO: may need to add a mechanism for include paths
#include "Neon/domain/details/dGrid/dIndex.h"

// TODO: currently, all types and builtins need to be in the wp:: namespace
namespace wp
{

// import types into this namespace
using Dense_idx = ::Neon::domain::details::dGrid::dIndex;

// Coord constructor exposed as a free function
CUDA_CALLABLE inline Dense_idx Dense_idx_(int x, int y, int z)
{
    return Dense_idx(x, y, z);
}

// overload operator+ for colors
CUDA_CALLABLE inline void myPrint(const Dense_idx& a)
{
    printf("Dense_idx %d %d %d\n", a.getLocation().x,  a.getLocation().y, a.getLocation().z);
}

}
