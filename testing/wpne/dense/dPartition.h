#pragma once

#include "Neon/domain/details/dGrid/dPartition.h"
#include "./dIdx.h"

 namespace wp
{
using NeonDensePartitionInt = ::Neon::domain::details::dGrid::Partition<int,0>;

// Coord constructor exposed as a free function
CUDA_CALLABLE inline auto NeonDensePartition_read(
   NeonDensePartitionInt& p,
   NeonDenseIdx const & idx,
   int card,
   int  & value)
 -> void
{
   value = p(idx, card) ;
}

CUDA_CALLABLE inline auto NeonDensePartition_write(
   NeonDensePartitionInt& p,
   NeonDenseIdx const & idx,
   int card,
   int  const& value)
 -> void
{
    p(idx, card) = value;
}
}