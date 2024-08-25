#pragma once

#include "Neon/domain/details/dGrid/dPartition.h"
#include "./dIdx.h"

// NOTE: we need this header to avoid errors about missing copy constructor for Pitch (Vec_4d)
#include "Neon/core/types/vec/vec4d_integer.timp.h"

namespace wp
{
// using NeonDensePartitionInt = ::Neon::domain::details::dGrid::dPartition<int,0>;

// NOTE: We create a subclass so that we can add a custom constructor
template <typename T>
class NeonDensePartition : public ::Neon::domain::details::dGrid::dPartition<T,0>
{
public:

   // // NOTE: the constructor must accepts all members in declaration order!
   // NeonDensePartitionInt(Neon::DataView dataView,
   //                       Neon::index_3d dim,
   //                       int*           mem,
   //                       int            zHaloRadius,
   //                       int            zBoundaryRadius,
   //                       Pitch          pitch,
   //                       int            prtID,
   //                       Neon::index_3d origin,
   //                       int            cardinality,
   //                       Neon::index_3d fullGridSize,
   //                       bool           periodicZ,
   //                       NghIdx*        stencil)
   //    : ::Neon::domain::details::dGrid::dPartition<int,0>(
   //          dataView,
   //          mem,
   //          dim,
   //          zHaloRadius,
   //          zBoundaryRadius,
   //          pitch,
   //          prtID,
   //          origin,
   //          cardinality,
   //          fullGridSize,
   //          stencil)
   // {
   //    // Note: enablePeriodicAlongZ() is currently excluded by NEON_WARP_COMPILATION (host only)
   //    // if (periodicZ)
   //    //    enablePeriodicAlongZ();
   // }

   // initialize from bytes
   NeonDensePartition(const char* bytes, size_t n)
   {
      assert(n == sizeof(*this));
      memcpy(this, bytes, n);
   }

   // NOTE: need default constructor for adjoint vars
   NeonDensePartition()
   {
   }
};

using NeonDensePartitionInt = NeonDensePartition<int>;
using NeonDensePartitionDouble = NeonDensePartition<double>;
using NeonDensePartitionFloat = NeonDensePartition<float>;

// print
template<typename T>
CUDA_CALLABLE inline auto neon_print_generic(const NeonDensePartition<T>& p) -> void
{
   const Neon::index_3d& dim = p.dim();
   const Neon::index_3d& halo = p.halo();
   const Neon::index_3d& origin = p.origin();

   printf("NeonDensePartitionInt(dim={%d, %d, %d}, halo={%d, %d, %d}, origin={%d, %d, %d}, mem=%p)\n",
      dim.x, dim.y, dim.z,
      halo.x, halo.y, halo.z,
      origin.x, origin.y, origin.z,
      p.mem()
   );
}

template<typename T>
CUDA_CALLABLE inline auto neon_read(
   NeonDensePartition<T>& p,
   NeonDenseIdx const & idx,
   int card)
 -> T
{
   return p(idx, card);
}

template<typename T>
CUDA_CALLABLE inline auto neon_write(
   NeonDensePartition<T>& p,
   NeonDenseIdx const & idx,
   int card,
   T  const& value)
 -> void
{
    p(idx, card) = value;
}
}
