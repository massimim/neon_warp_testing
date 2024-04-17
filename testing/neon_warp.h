#pragma once

// TODO: may need to add a mechanism for include paths
#include "Neon/domain/details/dGrid/dIndex.h"

// TODO: currently, all types and builtins need to be in the wp:: namespace
namespace wp
{

// import types into this namespace
using DenseIndex = ::Neon::domain::details::dGrid::dIndex;

// Coord constructor exposed as a free function
CUDA_CALLABLE inline DenseIndex DenseIndex_(int x, int y, int z)
{
    return DenseIndex(x, y, z);
}

// overload operator+ for colors
CUDA_CALLABLE inline void myPrint(const DenseIndex& a)
{
    printf("DenseIndex %d %d %d\n", a.getLocation().x,  a.getLocation().y, a.getLocation().z);
}

}
