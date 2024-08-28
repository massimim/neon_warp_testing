#pragma once

//#include "Neon/domain/details/dGrid/dIndex.h"

#include "Neon/core/types/DataView.h"

// TODO: currently, all types and builtins need to be in the wp:: namespace
namespace wp
{

// import types into this namespace
using NeonDataView = ::Neon::DataView;

// print dense index
CUDA_CALLABLE inline auto NeonDataView_print(const NeonDataView& a) -> void
{
    printf("NeonDataView(%d)\n", int(a));
}

}
