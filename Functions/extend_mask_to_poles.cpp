#include <math.h>
#include <algorithm>
#include <vector>
#include <cassert>
#include "../functions.hpp"
#include "../constants.hpp"


void extend_mask_to_poles(
        std::vector<bool> & mask_to_extend,
        const dataset & source_data,
        const std::vector<double> & extended_latitude,
        const int Ilat_start,
        const bool extend_val
        ) {

    // Get grid sizes
    const size_t    Ntime   = source_data.Ntime,
                    Ndepth  = source_data.Ndepth,
                    Nlat    = source_data.Nlat,
                    Nlon    = source_data.Nlon;

    const size_t extended_Nlat = extended_latitude.size();

    const size_t size_to_extend = mask_to_extend.size(),
                 extended_size  = Ntime * Ndepth * extended_Nlat * Nlon;

    // Start extended mask if 'land' everywhere if using land,
    //  otherwise zero-velocity water
    std::vector<bool> extended_mask( extended_size, extend_val );

    int Itime, Idepth, Ilat, Ilon;
    size_t extended_index;

    // Copy from the original mask into a padded one
    for ( size_t index = 0; index < size_to_extend; index++ ) {
        Index1to4( index, Itime, Idepth, Ilat, Ilon, Ntime, Ndepth, Nlat, Nlon );

        extended_index = Index( Itime, Idepth, Ilat + Ilat_start, Ilon, Ntime, Ndepth, extended_Nlat, Nlon );

        extended_mask.at( extended_index ) = mask_to_extend.at( index );
    }

    // Now swap out the contents of the extended mask 
    //  into the mask_to_extend
    mask_to_extend.swap( extended_mask );

}

